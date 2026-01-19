#!/usr/bin/env python3
"""
Full Profile + Analyze for TopK & Sort Module

This script demonstrates the generic profiler utility on the topk & sort module,
showing complete bottleneck analysis and optimization suggestions.
"""

import subprocess
from pathlib import Path
import json
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from profiler.bottleneck_analyzer import BottleneckAnalyzer, BottleneckType

WORKSPACE = Path("/home/sapmajum/kernel_optimization_framework/profiler/test_workspace")
WORKSPACE.mkdir(exist_ok=True)

# GPU specs for MI355X (gfx950)
GPU_SPECS = {
    "peak_tflops_fp16": 1600.0,
    "peak_tflops_fp32": 800.0,
    "peak_bandwidth_gbps": 8000.0,
    "num_cus": 304,
    "simd_width": 64,
    "waves_per_cu": 32,
    "max_waves": 304 * 32,
    "lds_per_cu_kb": 64,
    "clock_mhz": 2500,
}


def create_comprehensive_profile_script() -> str:
    """Create a comprehensive profiling script for topk & sort."""
    return '''#!/usr/bin/env python3
"""
Comprehensive TopK & Sort Module Profiling with Bottleneck Analysis

This script profiles each component of the topk & sort pipeline and
provides detailed metrics for bottleneck analysis.
"""
import torch
import triton
import triton.language as tl
import numpy as np
import json
import os
import time
os.environ['TRITON_PRINT_AUTOTUNING'] = '0'

device = "cuda"

# Import aiter kernels
from aiter import biased_grouped_topk, moe_sorting_fwd

print("=" * 80)
print("COMPREHENSIVE TOPK & SORT MODULE PROFILING")
print("=" * 80)

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

print(f"\\nConfiguration:")
print(f"  Batch size (M): {M}")
print(f"  TopK (K): {K}")
print(f"  Shared experts (S): {S}")
print(f"  Num experts: {NUM_EXPERTS}")
print(f"  Total experts: {TOTAL_EXPERTS}")

# Optimized Triton append kernel (from our optimization work)
@triton.jit
def fast_append_kernel(
    ids_in, w_in, ids_out, w_out,
    num_experts, scale,
    K: tl.constexpr, S: tl.constexpr, M: tl.constexpr
):
    """Single-block kernel that processes all tokens."""
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

def run_pipeline():
    biased_grouped_topk(gating, bias.to(gating.dtype), topk_w, topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    fast_append_kernel[(1,)](topk_ids, topk_w, ids_out, w_out, NUM_EXPERTS, 1.0, K, S, M)
    moe_sorting_fwd(ids_out, w_out, sorted_ids, sorted_w, sorted_exp, num_valid, moe_buf, TOTAL_EXPERTS, UNIT_SIZE)

# Heavy warmup to ensure JIT compilation is complete
print("\\n" + "=" * 80)
print("PHASE 1: JIT WARMUP (1000 iterations)")
print("=" * 80)
for i in range(1000):
    run_pipeline()
    if (i+1) % 200 == 0:
        torch.cuda.synchronize()
        print(f"  Warmup progress: {i+1}/1000")
torch.cuda.synchronize()
print("JIT compilation complete.")

# Additional warmup per component
print("\\nComponent-specific warmup...")
for _ in range(200):
    biased_grouped_topk(gating, bias.to(gating.dtype), topk_w, topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
torch.cuda.synchronize()

for _ in range(200):
    fast_append_kernel[(1,)](topk_ids, topk_w, ids_out, w_out, NUM_EXPERTS, 1.0, K, S, M)
torch.cuda.synchronize()

for _ in range(200):
    num_valid.zero_()
    moe_sorting_fwd(ids_out, w_out, sorted_ids, sorted_w, sorted_exp, num_valid, moe_buf, TOTAL_EXPERTS, UNIT_SIZE)
torch.cuda.synchronize()

print("Component warmup complete.")

# Detailed profiling
print("\\n" + "=" * 80)
print("PHASE 2: DETAILED PROFILING (500 iterations each)")
print("=" * 80)

results = {
    "config": {
        "M": M, "K": K, "S": S,
        "num_experts": NUM_EXPERTS,
        "total_experts": TOTAL_EXPERTS,
        "unit_size": UNIT_SIZE,
    },
    "components": {},
    "memory_analysis": {},
}

NUM_ITERS = 500

# 1. HIP biased_grouped_topk
print("\\n[1/4] Profiling HIP biased_grouped_topk...")
hip_times = []
for _ in range(NUM_ITERS):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    biased_grouped_topk(gating, bias.to(gating.dtype), topk_w, topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    end.record()
    torch.cuda.synchronize()
    hip_times.append(start.elapsed_time(end) * 1000)  # us

hip_bytes = (M * NUM_EXPERTS * 2 +  # gating input (bf16)
             NUM_EXPERTS * 2 +       # bias input (bf16)
             M * K * 4 +             # topk_w output (fp32)
             M * K * 4)              # topk_ids output (int32)
hip_flops = M * NUM_EXPERTS * 2  # Softmax + TopK selection

results["components"]["hip_topk"] = {
    "type": "HIP",
    "mean_us": float(np.mean(hip_times)),
    "std_us": float(np.std(hip_times)),
    "min_us": float(np.min(hip_times)),
    "max_us": float(np.max(hip_times)),
    "p50_us": float(np.percentile(hip_times, 50)),
    "p95_us": float(np.percentile(hip_times, 95)),
    "p99_us": float(np.percentile(hip_times, 99)),
    "bytes_transferred": hip_bytes,
    "flops_estimated": hip_flops,
    "achieved_bandwidth_gbps": hip_bytes / (np.mean(hip_times) * 1e3),
    "arithmetic_intensity": hip_flops / hip_bytes,
}
print(f"  Mean: {np.mean(hip_times):.2f} ± {np.std(hip_times):.2f} us")
print(f"  P50/P95/P99: {np.percentile(hip_times, 50):.2f} / {np.percentile(hip_times, 95):.2f} / {np.percentile(hip_times, 99):.2f} us")

# 2. Triton fast_append
print("\\n[2/4] Profiling Triton fast_append...")
triton_times = []
for _ in range(NUM_ITERS):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fast_append_kernel[(1,)](topk_ids, topk_w, ids_out, w_out, NUM_EXPERTS, 1.0, K, S, M)
    end.record()
    torch.cuda.synchronize()
    triton_times.append(start.elapsed_time(end) * 1000)

triton_bytes = (M * K * 4 +       # ids input (int32)
                M * K * 4 +       # weights input (fp32)
                M * (K+S) * 4 +   # ids output (int32)
                M * (K+S) * 4)    # weights output (fp32)
triton_flops = 0  # Just memory operations

results["components"]["triton_append"] = {
    "type": "Triton",
    "mean_us": float(np.mean(triton_times)),
    "std_us": float(np.std(triton_times)),
    "min_us": float(np.min(triton_times)),
    "max_us": float(np.max(triton_times)),
    "p50_us": float(np.percentile(triton_times, 50)),
    "p95_us": float(np.percentile(triton_times, 95)),
    "p99_us": float(np.percentile(triton_times, 99)),
    "bytes_transferred": triton_bytes,
    "flops_estimated": triton_flops,
    "achieved_bandwidth_gbps": triton_bytes / (np.mean(triton_times) * 1e3),
    "arithmetic_intensity": 0,  # Pure memory kernel
}
print(f"  Mean: {np.mean(triton_times):.2f} ± {np.std(triton_times):.2f} us")
print(f"  P50/P95/P99: {np.percentile(triton_times, 50):.2f} / {np.percentile(triton_times, 95):.2f} / {np.percentile(triton_times, 99):.2f} us")

# 3. CK moe_sorting (2 phases)
print("\\n[3/4] Profiling CK moe_sorting...")
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

ck_bytes = (M * (K+S) * 4 +       # ids input
            M * (K+S) * 4 +       # weights input
            MAX_PAD * 4 +         # sorted_ids output
            MAX_PAD * 4 +         # sorted_w output
            MAX_PAD * 4 +         # sorted_exp output
            TOTAL_EXPERTS * 4)    # moe_buf
ck_flops = M * (K+S) * np.log2(TOTAL_EXPERTS)  # Approximate sorting FLOPs

results["components"]["ck_sorting"] = {
    "type": "CK",
    "mean_us": float(np.mean(ck_times)),
    "std_us": float(np.std(ck_times)),
    "min_us": float(np.min(ck_times)),
    "max_us": float(np.max(ck_times)),
    "p50_us": float(np.percentile(ck_times, 50)),
    "p95_us": float(np.percentile(ck_times, 95)),
    "p99_us": float(np.percentile(ck_times, 99)),
    "bytes_transferred": ck_bytes,
    "flops_estimated": ck_flops,
    "achieved_bandwidth_gbps": ck_bytes / (np.mean(ck_times) * 1e3),
    "arithmetic_intensity": ck_flops / ck_bytes,
}
print(f"  Mean: {np.mean(ck_times):.2f} ± {np.std(ck_times):.2f} us")
print(f"  P50/P95/P99: {np.percentile(ck_times, 50):.2f} / {np.percentile(ck_times, 95):.2f} / {np.percentile(ck_times, 99):.2f} us")

# 4. Full pipeline
print("\\n[4/4] Profiling Full Pipeline...")
full_times = []
for _ in range(NUM_ITERS):
    num_valid.zero_()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    run_pipeline()
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
print(f"  Mean: {np.mean(full_times):.2f} ± {np.std(full_times):.2f} us")
print(f"  P50/P95/P99: {np.percentile(full_times, 50):.2f} / {np.percentile(full_times, 95):.2f} / {np.percentile(full_times, 99):.2f} us")

# Memory analysis
print("\\n" + "=" * 80)
print("PHASE 3: MEMORY & BOTTLENECK ANALYSIS")
print("=" * 80)

# GPU specs for MI355X
PEAK_BW_GBPS = 8000.0
PEAK_TFLOPS_FP16 = 1600.0
RIDGE_POINT = (PEAK_TFLOPS_FP16 * 1e12) / (PEAK_BW_GBPS * 1e9)

print(f"\\nGPU Specs (MI355X / gfx950):")
print(f"  Peak Memory BW: {PEAK_BW_GBPS:.0f} GB/s")
print(f"  Peak FP16 TFLOPS: {PEAK_TFLOPS_FP16:.0f}")
print(f"  Roofline Ridge Point: {RIDGE_POINT:.2f} FLOP/byte")

for name, data in results["components"].items():
    ai = data["arithmetic_intensity"]
    bw_util = data["achieved_bandwidth_gbps"] / PEAK_BW_GBPS * 100
    
    if ai < RIDGE_POINT / 10:
        bottleneck = "MEMORY (very low AI)"
    elif ai < RIDGE_POINT:
        bottleneck = "MEMORY"
    elif ai > RIDGE_POINT * 10:
        bottleneck = "COMPUTE"
    else:
        bottleneck = "BALANCED"
        
    # Short kernels are latency bound
    if data["mean_us"] < 5:
        bottleneck = "LATENCY (kernel too short)"
    
    data["bottleneck"] = bottleneck
    data["bandwidth_utilization_pct"] = bw_util
    
    print(f"\\n{name} ({data['type']}):")
    print(f"  Latency: {data['mean_us']:.2f} us")
    print(f"  Arithmetic Intensity: {ai:.4f} FLOP/byte")
    print(f"  Achieved Bandwidth: {data['achieved_bandwidth_gbps']:.2f} GB/s ({bw_util:.2f}%)")
    print(f"  Bottleneck: {bottleneck}")

# Summary
print("\\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

total = results['full_pipeline']['mean_us']
print(f"\\nFull Pipeline Latency: {total:.2f} us")
print("\\nComponent Breakdown:")
comp_total = 0
for name, data in results["components"].items():
    pct = data['mean_us'] / total * 100
    comp_total += data['mean_us']
    print(f"  {name:<20} {data['mean_us']:>8.2f} us ({pct:>5.1f}%)")

overhead = total - comp_total
print(f"  {'Kernel launch overhead':<20} {overhead:>8.2f} us ({overhead/total*100:>5.1f}%)")

print("\\n" + "=" * 80)
print("OPTIMIZATION RECOMMENDATIONS")
print("=" * 80)

# Find the bottleneck component
sorted_components = sorted(results["components"].items(), key=lambda x: x[1]["mean_us"], reverse=True)
top_comp = sorted_components[0]

print(f"\\nPrimary bottleneck: {top_comp[0]} ({top_comp[1]['mean_us']:.2f} us, {top_comp[1]['mean_us']/total*100:.1f}% of total)")
print(f"Bottleneck type: {top_comp[1]['bottleneck']}")

print("\\nRecommendations:")
bottleneck = top_comp[1]['bottleneck']
if "MEMORY" in bottleneck:
    print("  1. [HIGH] Vectorize memory accesses (use float4)")
    print("  2. [HIGH] Improve memory coalescing patterns")
    print("  3. [MED] Consider fusing with adjacent operations")
    print("  4. [MED] Use shared memory for repeated accesses")
elif "LATENCY" in bottleneck:
    print("  1. [HIGH] Fuse with adjacent kernels to reduce launch overhead")
    print("  2. [HIGH] Use persistent kernels for small workloads")
    print("  3. [MED] Batch multiple invocations together")
elif "COMPUTE" in bottleneck:
    print("  1. [HIGH] Use tensor cores (MFMA instructions)")
    print("  2. [MED] Reduce unnecessary computation")
    print("  3. [LOW] Already memory-efficient, focus on algorithm")
else:
    print("  1. [MED] Profile at lower level to identify specific bottleneck")
    print("  2. [MED] Consider both compute and memory optimizations")

# Save full results
with open("/workspace/topk_sort_full_analysis.json", "w") as f:
    json.dump(results, f, indent=2)

print("\\n" + "=" * 80)
print("Full results saved to topk_sort_full_analysis.json")
print("=" * 80)
'''


def run_full_analysis():
    """Run comprehensive profiling and analysis."""
    print("=" * 80)
    print("TOPK & SORT MODULE - FULL PROFILE + ANALYZE")
    print("=" * 80)
    print("\nRunning comprehensive profiling in Docker container...")
    print("(This includes heavy JIT warmup to ensure accurate measurements)")
    print()
    
    # Write script
    script = create_comprehensive_profile_script()
    script_path = WORKSPACE / "profile_full.py"
    script_path.write_text(script)
    
    # Run in Docker
    cmd = [
        "docker", "run", "--rm",
        "--device=/dev/kfd", "--device=/dev/dri", "--ipc=host",
        "-e", "HIP_VISIBLE_DEVICES=3",
        "-v", f"{WORKSPACE}:/workspace",
        "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x",
        "python3", "/workspace/profile_full.py"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    print(result.stdout)
    
    if result.stderr:
        # Filter out JIT compilation messages
        stderr_lines = result.stderr.split('\n')
        important_errors = [l for l in stderr_lines if not l.startswith('[aiter]') and l.strip()]
        if important_errors:
            print("\nNotes/Warnings:")
            for line in important_errors[:10]:
                print(f"  {line}")
    
    # Load and analyze results
    results_file = WORKSPACE / "topk_sort_full_analysis.json"
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        
        # Run local bottleneck analysis
        print("\n" + "=" * 80)
        print("LOCAL BOTTLENECK ANALYZER OUTPUT")
        print("=" * 80)
        
        analyzer = BottleneckAnalyzer(gpu_arch="gfx950")
        
        for name, data in results["components"].items():
            print(f"\n{'='*40}")
            print(f"Component: {name}")
            print(f"{'='*40}")
            
            # Create counter dict (simulated since we don't have hardware counters)
            counters = {
                "SQ_WAVES": 1000,  # Placeholder
                "SQ_INSTS_VALU": int(data.get("flops_estimated", 0) / 64),
                "SQ_INSTS_SALU": 100,
            }
            
            report = analyzer.analyze(
                duration_us=data["mean_us"],
                counters=counters,
                memory_bytes=data["bytes_transferred"],
                flops=data.get("flops_estimated", 0),
            )
            
            print(report.analysis_text)
        
        return results
    else:
        print("ERROR: Results file not found!")
        return None


if __name__ == "__main__":
    results = run_full_analysis()

