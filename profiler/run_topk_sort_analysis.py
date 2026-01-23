#!/usr/bin/env python3
"""
Run comprehensive profile + analyze on the TopK & Sort module.

Uses the generic profiler utility to demonstrate its reusability.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from profiler.generic_profiler import GenericProfiler

# This is the profiling script that will run inside Docker
TOPK_SORT_PROFILE_SCRIPT = '''#!/usr/bin/env python3
"""Profile TopK & Sort Module - Outputs JSON for analysis."""
import torch
import triton
import triton.language as tl
import numpy as np
import json
import os
os.environ['TRITON_PRINT_AUTOTUNING'] = '0'

device = "cuda"

from aiter import biased_grouped_topk, moe_sorting_fwd

# Configuration
M = 4  # Batch size
K = 8  # TopK
S = 1  # Shared experts
NUM_EXPERTS = 256
TOTAL_EXPERTS = NUM_EXPERTS + S
NUM_GROUPS = 8
TOPK_GROUP = 4
UNIT_SIZE = 64
MAX_PAD = M * (K+S) + TOTAL_EXPERTS * UNIT_SIZE

print("TopK & Sort Module Profiling")
print("=" * 60)

# Define the Triton kernel ONCE at module level for proper JIT caching
@triton.jit
def fast_append_kernel(
    ids_in, w_in, ids_out, w_out,
    num_experts, scale,
    K: tl.constexpr, S: tl.constexpr, M: tl.constexpr
):
    for token in range(M):
        in_off = token * K
        out_off = token * (K + S)
        offs = tl.arange(0, K)
        ids = tl.load(ids_in + in_off + offs)
        ws = tl.load(w_in + in_off + offs)
        tl.store(ids_out + out_off + offs, ids)
        tl.store(w_out + out_off + offs, ws)
        tl.store(ids_out + out_off + K, num_experts)
        tl.store(w_out + out_off + K, scale)

# Allocate tensors
torch.manual_seed(42)
gating = torch.randn(M, NUM_EXPERTS, dtype=torch.bfloat16, device=device)
bias = torch.randn(NUM_EXPERTS, dtype=torch.bfloat16, device=device)

topk_w = torch.empty((M, K), dtype=torch.float32, device=device)
topk_ids = torch.empty((M, K), dtype=torch.int32, device=device)
ids_out = torch.empty((M, K+S), dtype=torch.int32, device=device)
w_out = torch.empty((M, K+S), dtype=torch.float32, device=device)
sorted_ids = torch.empty(MAX_PAD, dtype=torch.int32, device=device)
sorted_w = torch.empty(MAX_PAD, dtype=torch.float32, device=device)
sorted_exp = torch.empty(MAX_PAD, dtype=torch.int32, device=device)
num_valid = torch.empty(1, dtype=torch.int32, device=device)
moe_buf = torch.empty(TOTAL_EXPERTS + 1, dtype=torch.int32, device=device)

# Pre-compile Triton kernel by calling it
print("Pre-compiling Triton kernel...")
fast_append_kernel[(1,)](topk_ids, topk_w, ids_out, w_out, NUM_EXPERTS, 1.0, K, S, M)
torch.cuda.synchronize()

# Heavy warmup
print("Heavy warmup (2000 iterations)...")
for i in range(2000):
    biased_grouped_topk(gating, bias.to(gating.dtype), topk_w, topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    fast_append_kernel[(1,)](topk_ids, topk_w, ids_out, w_out, NUM_EXPERTS, 1.0, K, S, M)
    moe_sorting_fwd(ids_out, w_out, sorted_ids, sorted_w, sorted_exp, num_valid, moe_buf, TOTAL_EXPERTS, UNIT_SIZE)
torch.cuda.synchronize()
print("Warmup complete.")

NUM_ITERS = 1000

# Profile each component
results = {
    "config": {"M": M, "K": K, "S": S, "num_experts": NUM_EXPERTS, "total_experts": TOTAL_EXPERTS, "unit_size": UNIT_SIZE},
    "components": {},
}

print("\\nProfiling components...")

# 1. HIP biased_grouped_topk
print("  [1/4] HIP topk...")
hip_times = []
for _ in range(NUM_ITERS):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    biased_grouped_topk(gating, bias.to(gating.dtype), topk_w, topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    end.record()
    torch.cuda.synchronize()
    hip_times.append(start.elapsed_time(end) * 1000)

results["components"]["hip_topk"] = {
    "type": "HIP",
    "mean_us": float(np.mean(hip_times)),
    "std_us": float(np.std(hip_times)),
    "min_us": float(np.min(hip_times)),
    "max_us": float(np.max(hip_times)),
    "p50_us": float(np.percentile(hip_times, 50)),
    "p95_us": float(np.percentile(hip_times, 95)),
    "p99_us": float(np.percentile(hip_times, 99)),
    "bytes_transferred": M * NUM_EXPERTS * 2 + NUM_EXPERTS * 2 + M * K * 4 + M * K * 4,
    "flops_estimated": M * NUM_EXPERTS * 2,
}

# 2. Triton fast_append
print("  [2/4] Triton append...")
triton_times = []
for _ in range(NUM_ITERS):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fast_append_kernel[(1,)](topk_ids, topk_w, ids_out, w_out, NUM_EXPERTS, 1.0, K, S, M)
    end.record()
    torch.cuda.synchronize()
    triton_times.append(start.elapsed_time(end) * 1000)

results["components"]["triton_append"] = {
    "type": "Triton",
    "mean_us": float(np.mean(triton_times)),
    "std_us": float(np.std(triton_times)),
    "min_us": float(np.min(triton_times)),
    "max_us": float(np.max(triton_times)),
    "p50_us": float(np.percentile(triton_times, 50)),
    "p95_us": float(np.percentile(triton_times, 95)),
    "p99_us": float(np.percentile(triton_times, 99)),
    "bytes_transferred": M * K * 4 + M * K * 4 + M * (K+S) * 4 + M * (K+S) * 4,
    "flops_estimated": 0,  # Pure memory operations
}

# 3. CK moe_sorting
print("  [3/4] CK sorting...")
ck_times = []
for _ in range(NUM_ITERS):
    num_valid.zero_()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    moe_sorting_fwd(ids_out, w_out, sorted_ids, sorted_w, sorted_exp, num_valid, moe_buf, TOTAL_EXPERTS, UNIT_SIZE)
    end.record()
    torch.cuda.synchronize()
    ck_times.append(start.elapsed_time(end) * 1000)

results["components"]["ck_sorting"] = {
    "type": "CK",
    "mean_us": float(np.mean(ck_times)),
    "std_us": float(np.std(ck_times)),
    "min_us": float(np.min(ck_times)),
    "max_us": float(np.max(ck_times)),
    "p50_us": float(np.percentile(ck_times, 50)),
    "p95_us": float(np.percentile(ck_times, 95)),
    "p99_us": float(np.percentile(ck_times, 99)),
    "bytes_transferred": M * (K+S) * 4 * 2 + MAX_PAD * 4 * 3 + TOTAL_EXPERTS * 4,
    "flops_estimated": int(M * (K+S) * np.log2(TOTAL_EXPERTS)),
}

# 4. Full pipeline
print("  [4/4] Full pipeline...")
full_times = []
for _ in range(NUM_ITERS):
    num_valid.zero_()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    biased_grouped_topk(gating, bias.to(gating.dtype), topk_w, topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    fast_append_kernel[(1,)](topk_ids, topk_w, ids_out, w_out, NUM_EXPERTS, 1.0, K, S, M)
    moe_sorting_fwd(ids_out, w_out, sorted_ids, sorted_w, sorted_exp, num_valid, moe_buf, TOTAL_EXPERTS, UNIT_SIZE)
    end.record()
    torch.cuda.synchronize()
    full_times.append(start.elapsed_time(end) * 1000)

results["full_pipeline"] = {
    "mean_us": float(np.mean(full_times)),
    "std_us": float(np.std(full_times)),
    "min_us": float(np.min(full_times)),
    "max_us": float(np.max(full_times)),
    "p50_us": float(np.percentile(full_times, 50)),
    "p95_us": float(np.percentile(full_times, 95)),
    "p99_us": float(np.percentile(full_times, 99)),
}

# Print summary
print("\\n" + "=" * 60)
print("PROFILING SUMMARY")
print("=" * 60)
total = results["full_pipeline"]["mean_us"]
for name, data in results["components"].items():
    pct = data["mean_us"] / total * 100
    print(f"  {name:<20}: {data['mean_us']:>8.2f} ± {data['std_us']:>5.2f} us ({pct:>5.1f}%)")
print("-" * 60)
print(f"  {'Full Pipeline':<20}: {total:>8.2f} ± {results['full_pipeline']['std_us']:>5.2f} us (100.0%)")

# Save results
with open("/workspace/results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\\nResults saved to /workspace/results.json")
'''


def main():
    print("=" * 80)
    print("TOPK & SORT MODULE - GENERIC PROFILER ANALYSIS")
    print("=" * 80)
    print("\nUsing the generic profiler utility from the agent framework.")
    print("This demonstrates how the profiler can be reused for any module.\n")
    
    # Create profiler
    profiler = GenericProfiler(
        gpu_id=3,  # Using GPU 3
        work_dir="/home/sapmajum/kernel_optimization_framework/profiler/topk_sort_workspace"
    )
    
    try:
        # Run profiling
        print("Running profiling in Docker container...")
        print("(This includes heavy warmup to ensure accurate measurements)\n")
        
        results = profiler.profile_script(
            script_content=TOPK_SORT_PROFILE_SCRIPT,
            output_json="results.json",
            timeout=900
        )
        
        # Analyze results
        analysis = profiler.analyze_results(results, module_name="TopK & Sort Module")
        
        # Print detailed analysis
        profiler.print_analysis(analysis)
        
        return analysis
        
    finally:
        pass  # Don't cleanup so we can inspect results


if __name__ == "__main__":
    analysis = main()

