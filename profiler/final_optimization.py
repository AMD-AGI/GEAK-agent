#!/usr/bin/env python3
"""
Final TopK & Sort Module Optimization - Clean Benchmark

Based on comprehensive profiling analysis.
"""

import subprocess
from pathlib import Path
import json

WORKSPACE = Path("/home/sapmajum/kernel_optimization_framework/profiler/final_ws")
WORKSPACE.mkdir(exist_ok=True)

FINAL_SCRIPT = '''#!/usr/bin/env python3
"""
FINAL TOPK & SORT OPTIMIZATION BENCHMARK

Based on profiling analysis showing Triton append kernel launch overhead
is 60% of total pipeline latency (34 us for 544 bytes of data).

Optimization: Replace Triton kernel with pre-filled buffers + copy.
"""
import torch
import triton
import triton.language as tl
import numpy as np
from scipy import stats
import json
import os
os.environ['TRITON_PRINT_AUTOTUNING'] = '0'

from aiter import biased_grouped_topk, moe_sorting_fwd

device = "cuda"

print("=" * 80)
print("FINAL TOPK & SORT OPTIMIZATION BENCHMARK")
print("Based on Profiling Analysis")
print("=" * 80)

# Configuration
M, K, S = 4, 8, 1
NUM_EXPERTS = 256
TOTAL_EXPERTS = NUM_EXPERTS + S
NUM_GROUPS, TOPK_GROUP = 8, 4
UNIT_SIZE = 64
MAX_PAD = M * (K+S) + TOTAL_EXPERTS * UNIT_SIZE

print(f"\\nConfiguration: M={M}, K={K}, S={S}, NUM_EXPERTS={NUM_EXPERTS}")

@triton.jit
def fast_append(ids_in, w_in, ids_out, w_out, num_experts, scale, K: tl.constexpr, S: tl.constexpr, M: tl.constexpr):
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

# Fixed seed for reproducibility
torch.manual_seed(42)
gating = torch.randn(M, NUM_EXPERTS, dtype=torch.bfloat16, device=device)
bias = torch.randn(NUM_EXPERTS, dtype=torch.bfloat16, device=device)

# ============================================================================
# PHASE 1: ISOLATED CORRECTNESS VERIFICATION
# ============================================================================
print("\\n" + "=" * 80)
print("PHASE 1: CORRECTNESS VERIFICATION")
print("=" * 80)

# Baseline tensors
b_topk_w = torch.empty((M, K), dtype=torch.float32, device=device)
b_topk_ids = torch.empty((M, K), dtype=torch.int32, device=device)
b_ids_out = torch.empty((M, K+S), dtype=torch.int32, device=device)
b_w_out = torch.empty((M, K+S), dtype=torch.float32, device=device)
b_sorted_ids = torch.empty(MAX_PAD, dtype=torch.int32, device=device)
b_sorted_w = torch.empty(MAX_PAD, dtype=torch.float32, device=device)
b_sorted_exp = torch.empty(MAX_PAD, dtype=torch.int32, device=device)
b_num_valid = torch.empty(1, dtype=torch.int32, device=device)
b_moe_buf = torch.empty(TOTAL_EXPERTS + 1, dtype=torch.int32, device=device)

# Optimized tensors
o_topk_w = torch.empty((M, K), dtype=torch.float32, device=device)
o_topk_ids = torch.empty((M, K), dtype=torch.int32, device=device)
o_ids_out = torch.empty((M, K+S), dtype=torch.int32, device=device)
o_w_out = torch.empty((M, K+S), dtype=torch.float32, device=device)
o_sorted_ids = torch.empty(MAX_PAD, dtype=torch.int32, device=device)
o_sorted_w = torch.empty(MAX_PAD, dtype=torch.float32, device=device)
o_sorted_exp = torch.empty(MAX_PAD, dtype=torch.int32, device=device)
o_num_valid = torch.empty(1, dtype=torch.int32, device=device)
o_moe_buf = torch.empty(TOTAL_EXPERTS + 1, dtype=torch.int32, device=device)

# Pre-fill shared expert columns ONCE
o_ids_out[:, K] = NUM_EXPERTS
o_w_out[:, K] = 1.0

# Run baseline
biased_grouped_topk(gating, bias.to(gating.dtype), b_topk_w, b_topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
fast_append[(1,)](b_topk_ids, b_topk_w, b_ids_out, b_w_out, NUM_EXPERTS, 1.0, K, S, M)
b_num_valid.zero_()
moe_sorting_fwd(b_ids_out, b_w_out, b_sorted_ids, b_sorted_w, b_sorted_exp, b_num_valid, b_moe_buf, TOTAL_EXPERTS, UNIT_SIZE)
torch.cuda.synchronize()

baseline_valid = b_num_valid.item()
baseline_ids = b_ids_out.clone()
baseline_sorted = b_sorted_ids[:baseline_valid].clone()

# Run optimized
biased_grouped_topk(gating, bias.to(gating.dtype), o_topk_w, o_topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
o_ids_out[:, :K] = o_topk_ids
o_w_out[:, :K] = o_topk_w
o_num_valid.zero_()
moe_sorting_fwd(o_ids_out, o_w_out, o_sorted_ids, o_sorted_w, o_sorted_exp, o_num_valid, o_moe_buf, TOTAL_EXPERTS, UNIT_SIZE)
torch.cuda.synchronize()

opt_valid = o_num_valid.item()

# Verify
correct = (
    baseline_valid == opt_valid and
    torch.equal(baseline_ids, o_ids_out) and
    torch.equal(baseline_sorted, o_sorted_ids[:opt_valid])
)

print(f"  Baseline num_valid: {baseline_valid}")
print(f"  Optimized num_valid: {opt_valid}")
print(f"  ids_out match: {torch.equal(baseline_ids, o_ids_out)}")
print(f"  sorted_ids match: {torch.equal(baseline_sorted, o_sorted_ids[:opt_valid]) if baseline_valid == opt_valid else False}")
print(f"  OVERALL CORRECT: {correct}")

if not correct:
    print("\\n  ERROR: Correctness verification failed!")
    # Continue anyway to see performance difference

# ============================================================================
# PHASE 2: ISOLATED WARMUP - BASELINE ONLY
# ============================================================================
print("\\n" + "=" * 80)
print("PHASE 2: BASELINE WARMUP (10000 iterations)")
print("=" * 80)

def run_baseline():
    biased_grouped_topk(gating, bias.to(gating.dtype), b_topk_w, b_topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    fast_append[(1,)](b_topk_ids, b_topk_w, b_ids_out, b_w_out, NUM_EXPERTS, 1.0, K, S, M)
    b_num_valid.zero_()
    moe_sorting_fwd(b_ids_out, b_w_out, b_sorted_ids, b_sorted_w, b_sorted_exp, b_num_valid, b_moe_buf, TOTAL_EXPERTS, UNIT_SIZE)

for i in range(10000):
    run_baseline()
    if (i+1) % 2500 == 0:
        torch.cuda.synchronize()
        print(f"  Progress: {i+1}/10000")
torch.cuda.synchronize()

# ============================================================================
# PHASE 3: BASELINE BENCHMARK
# ============================================================================
print("\\n" + "=" * 80)
print("PHASE 3: BASELINE BENCHMARK (3000 iterations)")
print("=" * 80)

baseline_times = []
for _ in range(3000):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    run_baseline()
    end.record()
    torch.cuda.synchronize()
    baseline_times.append(start.elapsed_time(end) * 1000)

baseline_times = np.array(baseline_times)
baseline_mean = np.mean(baseline_times)
baseline_std = np.std(baseline_times)
print(f"  Baseline: {baseline_mean:.2f} ± {baseline_std:.2f} us")

# ============================================================================
# PHASE 4: ISOLATED WARMUP - OPTIMIZED ONLY
# ============================================================================
print("\\n" + "=" * 80)
print("PHASE 4: OPTIMIZED WARMUP (10000 iterations)")
print("=" * 80)

def run_optimized():
    biased_grouped_topk(gating, bias.to(gating.dtype), o_topk_w, o_topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    o_ids_out[:, :K] = o_topk_ids
    o_w_out[:, :K] = o_topk_w
    o_num_valid.zero_()
    moe_sorting_fwd(o_ids_out, o_w_out, o_sorted_ids, o_sorted_w, o_sorted_exp, o_num_valid, o_moe_buf, TOTAL_EXPERTS, UNIT_SIZE)

for i in range(10000):
    run_optimized()
    if (i+1) % 2500 == 0:
        torch.cuda.synchronize()
        print(f"  Progress: {i+1}/10000")
torch.cuda.synchronize()

# ============================================================================
# PHASE 5: OPTIMIZED BENCHMARK
# ============================================================================
print("\\n" + "=" * 80)
print("PHASE 5: OPTIMIZED BENCHMARK (3000 iterations)")
print("=" * 80)

opt_times = []
for _ in range(3000):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    run_optimized()
    end.record()
    torch.cuda.synchronize()
    opt_times.append(start.elapsed_time(end) * 1000)

opt_times = np.array(opt_times)
opt_mean = np.mean(opt_times)
opt_std = np.std(opt_times)
print(f"  Optimized: {opt_mean:.2f} ± {opt_std:.2f} us")

# ============================================================================
# PHASE 6: STATISTICAL ANALYSIS
# ============================================================================
print("\\n" + "=" * 80)
print("PHASE 6: STATISTICAL ANALYSIS")
print("=" * 80)

speedup = baseline_mean / opt_mean
improvement_pct = (baseline_mean - opt_mean) / baseline_mean * 100

# T-test for significance
t_stat, p_value = stats.ttest_ind(baseline_times, opt_times)
cohens_d = (baseline_mean - opt_mean) / np.sqrt((baseline_std**2 + opt_std**2) / 2)
significant = p_value < 0.001 and cohens_d > 0.2

print(f"  Speedup: {speedup:.2f}x")
print(f"  Improvement: {improvement_pct:.1f}%")
print(f"  T-statistic: {t_stat:.2f}")
print(f"  P-value: {p_value:.2e}")
print(f"  Cohen's d: {cohens_d:.2f}")
print(f"  Statistically Significant: {significant}")

# ============================================================================
# FINAL RESULTS
# ============================================================================
print("\\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          TOPK & SORT MODULE OPTIMIZATION RESULTS (BS={M})                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  BASELINE (HIP + Triton + CK):                                               ║
║    Latency:  {baseline_mean:>7.2f} ± {baseline_std:.2f} us                                         ║
║    P50:      {np.percentile(baseline_times, 50):>7.2f} us                                                 ║
║    P95:      {np.percentile(baseline_times, 95):>7.2f} us                                                 ║
║                                                                              ║
║  OPTIMIZED (HIP + Copy + CK):                                                ║
║    Latency:  {opt_mean:>7.2f} ± {opt_std:.2f} us                                         ║
║    P50:      {np.percentile(opt_times, 50):>7.2f} us                                                 ║
║    P95:      {np.percentile(opt_times, 95):>7.2f} us                                                 ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SPEEDUP:           {speedup:>5.2f}x                                                  ║
║  IMPROVEMENT:       {improvement_pct:>5.1f}%                                                ║
║  CORRECT:           {'YES' if correct else 'NO ':3}                                                   ║
║  SIGNIFICANT:       {'YES' if significant else 'NO ':3} (p={p_value:.2e})                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

ANALYSIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The optimization eliminates the Triton kernel launch overhead by:

1. PRE-FILLING shared expert columns in output buffer (done once at init)
2. REPLACING Triton kernel with PyTorch tensor assignment (lower overhead)

For small batch sizes (M=4), the Triton kernel launch overhead (~30-35 us)
dominates the actual data movement (~0.5 us worth of work).

The optimization reduces latency by ~{baseline_mean - opt_mean:.1f} us ({improvement_pct:.1f}%).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# Save results
results = {
    "config": {"M": M, "K": K, "S": S, "num_experts": NUM_EXPERTS},
    "baseline": {
        "mean_us": float(baseline_mean),
        "std_us": float(baseline_std),
        "p50_us": float(np.percentile(baseline_times, 50)),
        "p95_us": float(np.percentile(baseline_times, 95)),
        "p99_us": float(np.percentile(baseline_times, 99)),
    },
    "optimized": {
        "mean_us": float(opt_mean),
        "std_us": float(opt_std),
        "p50_us": float(np.percentile(opt_times, 50)),
        "p95_us": float(np.percentile(opt_times, 95)),
        "p99_us": float(np.percentile(opt_times, 99)),
    },
    "speedup": float(speedup),
    "improvement_pct": float(improvement_pct),
    "p_value": float(p_value),
    "cohens_d": float(cohens_d),
    "correct": bool(correct),
    "significant": bool(significant),
}

with open("/workspace/final_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Results saved to /workspace/final_results.json")
'''


def main():
    print("=" * 80)
    print("FINAL TOPK & SORT OPTIMIZATION (PROFILING-GUIDED)")
    print("=" * 80)
    
    # Write and run script
    script_path = WORKSPACE / "final_benchmark.py"
    script_path.write_text(FINAL_SCRIPT)
    
    cmd = [
        "docker", "run", "--rm",
        "--device=/dev/kfd", "--device=/dev/dri", "--ipc=host",
        "-e", "HIP_VISIBLE_DEVICES=3",
        "-v", f"{WORKSPACE}:/workspace",
        "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x",
        "python3", "/workspace/final_benchmark.py"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    print(result.stdout)
    
    if result.stderr:
        stderr_lines = [l for l in result.stderr.split('\n') if not l.startswith('[aiter]') and l.strip()]
        if stderr_lines:
            print("\nNotes:", stderr_lines[:5])
    
    # Load results
    results_path = WORKSPACE / "final_results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        return results
    
    return None


if __name__ == "__main__":
    results = main()

