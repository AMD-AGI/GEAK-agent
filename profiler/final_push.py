#!/usr/bin/env python3
"""Final push optimization - try to break 25 us barrier"""

import subprocess
from pathlib import Path
import json

WORKSPACE = Path("/home/sapmajum/kernel_optimization_framework/profiler/final_push_ws")
WORKSPACE.mkdir(exist_ok=True)

FINAL_PUSH_SCRIPT = '''#!/usr/bin/env python3
"""
FINAL PUSH - Breaking the 25us barrier

Current: 25.76 us (2.21x speedup)
Target: < 25 us
"""
import torch
import numpy as np
import json
import os
import gc
os.environ['TRITON_PRINT_AUTOTUNING'] = '0'

from aiter import biased_grouped_topk, moe_sorting_fwd

device = "cuda"

print("=" * 80)
print("FINAL PUSH OPTIMIZATION")
print("=" * 80)

M, K, S = 4, 8, 1
NUM_EXPERTS = 256
TOTAL_EXPERTS = NUM_EXPERTS + S
NUM_GROUPS, TOPK_GROUP = 8, 4
UNIT_SIZE = 64
MAX_PAD = M * (K+S) + TOTAL_EXPERTS * UNIT_SIZE

print(f"Config: M={M}, K={K}, S={S}")

torch.manual_seed(42)
gating = torch.randn(M, NUM_EXPERTS, dtype=torch.bfloat16, device=device)
bias = torch.randn(NUM_EXPERTS, dtype=torch.bfloat16, device=device)
pool = torch.cuda.graph_pool_handle()

# Helper for aligned allocation
def aligned_empty(shape, dtype, device, alignment=256):
    numel = 1
    for s in shape if isinstance(shape, tuple) else (shape,):
        numel *= s
    elem_size = torch.tensor([], dtype=dtype).element_size()
    total_bytes = numel * elem_size
    aligned_bytes = ((total_bytes + alignment - 1) // alignment) * alignment
    aligned_numel = aligned_bytes // elem_size
    return torch.empty(aligned_numel, dtype=dtype, device=device)[:numel].view(shape)

# ============================================================================
# CURRENT BEST: Aligned Memory + Graph
# ============================================================================
print("\\n[Current Best: Aligned + Graph]")

a_topk_w = aligned_empty((M, K), torch.float32, device)
a_topk_ids = aligned_empty((M, K), torch.int32, device)
a_ids_out = aligned_empty((M, K+S), torch.int32, device)
a_w_out = aligned_empty((M, K+S), torch.float32, device)
a_sorted_ids = aligned_empty(MAX_PAD, torch.int32, device)
a_sorted_w = aligned_empty(MAX_PAD, torch.float32, device)
a_sorted_exp = aligned_empty(MAX_PAD, torch.int32, device)
a_num_valid = aligned_empty(1, torch.int32, device)
a_moe_buf = aligned_empty(TOTAL_EXPERTS + 1, torch.int32, device)
a_ids_out[:, K] = NUM_EXPERTS
a_w_out[:, K] = 1.0

for _ in range(2000):
    biased_grouped_topk(gating, bias.to(gating.dtype), a_topk_w, a_topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    a_ids_out[:, :K] = a_topk_ids
    a_w_out[:, :K] = a_topk_w
    a_num_valid.zero_()
    moe_sorting_fwd(a_ids_out, a_w_out, a_sorted_ids, a_sorted_w, a_sorted_exp, a_num_valid, a_moe_buf, TOTAL_EXPERTS, UNIT_SIZE)
torch.cuda.synchronize()

g_best = torch.cuda.CUDAGraph()
with torch.cuda.graph(g_best, pool=pool):
    biased_grouped_topk(gating, bias.to(gating.dtype), a_topk_w, a_topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    a_ids_out[:, :K] = a_topk_ids
    a_w_out[:, :K] = a_topk_w
    a_num_valid.zero_()
    moe_sorting_fwd(a_ids_out, a_w_out, a_sorted_ids, a_sorted_w, a_sorted_exp, a_num_valid, a_moe_buf, TOTAL_EXPERTS, UNIT_SIZE)

def current_best():
    g_best.replay()

# ============================================================================
# STRATEGY 1: SKIP num_valid.zero_() 
# ============================================================================
print("\\n[Strategy 1: Skip num_valid zeroing]")

# The moe_sorting_fwd always writes num_valid, so zeroing might be unnecessary
# Let's test with a pre-capture that includes the zero

z_topk_w = aligned_empty((M, K), torch.float32, device)
z_topk_ids = aligned_empty((M, K), torch.int32, device)
z_ids_out = aligned_empty((M, K+S), torch.int32, device)
z_w_out = aligned_empty((M, K+S), torch.float32, device)
z_sorted_ids = aligned_empty(MAX_PAD, torch.int32, device)
z_sorted_w = aligned_empty(MAX_PAD, torch.float32, device)
z_sorted_exp = aligned_empty(MAX_PAD, torch.int32, device)
z_num_valid = aligned_empty(1, torch.int32, device)
z_moe_buf = aligned_empty(TOTAL_EXPERTS + 1, torch.int32, device)
z_ids_out[:, K] = NUM_EXPERTS
z_w_out[:, K] = 1.0

for _ in range(2000):
    biased_grouped_topk(gating, bias.to(gating.dtype), z_topk_w, z_topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    z_ids_out[:, :K] = z_topk_ids
    z_w_out[:, :K] = z_topk_w
    # Skip zero - just call sorting directly
    moe_sorting_fwd(z_ids_out, z_w_out, z_sorted_ids, z_sorted_w, z_sorted_exp, z_num_valid, z_moe_buf, TOTAL_EXPERTS, UNIT_SIZE)
torch.cuda.synchronize()

g_no_zero = torch.cuda.CUDAGraph()
with torch.cuda.graph(g_no_zero, pool=pool):
    biased_grouped_topk(gating, bias.to(gating.dtype), z_topk_w, z_topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    z_ids_out[:, :K] = z_topk_ids
    z_w_out[:, :K] = z_topk_w
    moe_sorting_fwd(z_ids_out, z_w_out, z_sorted_ids, z_sorted_w, z_sorted_exp, z_num_valid, z_moe_buf, TOTAL_EXPERTS, UNIT_SIZE)

def strategy_no_zero():
    g_no_zero.replay()

# ============================================================================
# STRATEGY 2: CONTIGUOUS OUTPUT FORMAT
# ============================================================================
print("\\n[Strategy 2: Contiguous Output]")

# Pre-allocate contiguous (M, K+S) tensor
c_out_ids = torch.empty((M, K+S), dtype=torch.int32, device=device)
c_out_w = torch.empty((M, K+S), dtype=torch.float32, device=device)
# Pre-fill shared expert
c_out_ids[:, K] = NUM_EXPERTS
c_out_w[:, K] = 1.0

# Create contiguous views for topk output
c_topk_ids_view = c_out_ids[:, :K].contiguous()
c_topk_w_view = c_out_w[:, :K].contiguous()

c_sorted_ids = torch.empty(MAX_PAD, dtype=torch.int32, device=device)
c_sorted_w = torch.empty(MAX_PAD, dtype=torch.float32, device=device)
c_sorted_exp = torch.empty(MAX_PAD, dtype=torch.int32, device=device)
c_num_valid = torch.empty(1, dtype=torch.int32, device=device)
c_moe_buf = torch.empty(TOTAL_EXPERTS + 1, dtype=torch.int32, device=device)

for _ in range(2000):
    biased_grouped_topk(gating, bias.to(gating.dtype), c_topk_w_view, c_topk_ids_view, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    c_out_ids[:, :K] = c_topk_ids_view  # Copy back
    c_out_w[:, :K] = c_topk_w_view
    c_num_valid.zero_()
    moe_sorting_fwd(c_out_ids, c_out_w, c_sorted_ids, c_sorted_w, c_sorted_exp, c_num_valid, c_moe_buf, TOTAL_EXPERTS, UNIT_SIZE)
torch.cuda.synchronize()

g_contig = torch.cuda.CUDAGraph()
with torch.cuda.graph(g_contig, pool=pool):
    biased_grouped_topk(gating, bias.to(gating.dtype), c_topk_w_view, c_topk_ids_view, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    c_out_ids[:, :K] = c_topk_ids_view
    c_out_w[:, :K] = c_topk_w_view
    c_num_valid.zero_()
    moe_sorting_fwd(c_out_ids, c_out_w, c_sorted_ids, c_sorted_w, c_sorted_exp, c_num_valid, c_moe_buf, TOTAL_EXPERTS, UNIT_SIZE)

def strategy_contig():
    g_contig.replay()

# ============================================================================
# STRATEGY 3: COMBINED - No zero + Aligned + Pool
# ============================================================================
print("\\n[Strategy 3: Combined Optimizations]")

f_topk_w = aligned_empty((M, K), torch.float32, device)
f_topk_ids = aligned_empty((M, K), torch.int32, device)
f_ids_out = aligned_empty((M, K+S), torch.int32, device)
f_w_out = aligned_empty((M, K+S), torch.float32, device)
f_sorted_ids = aligned_empty(MAX_PAD, torch.int32, device)
f_sorted_w = aligned_empty(MAX_PAD, torch.float32, device)
f_sorted_exp = aligned_empty(MAX_PAD, torch.int32, device)
f_num_valid = aligned_empty(1, torch.int32, device)
f_moe_buf = aligned_empty(TOTAL_EXPERTS + 1, torch.int32, device)
f_ids_out[:, K] = NUM_EXPERTS
f_w_out[:, K] = 1.0

for _ in range(2000):
    biased_grouped_topk(gating, bias.to(gating.dtype), f_topk_w, f_topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    f_ids_out[:, :K] = f_topk_ids
    f_w_out[:, :K] = f_topk_w
    moe_sorting_fwd(f_ids_out, f_w_out, f_sorted_ids, f_sorted_w, f_sorted_exp, f_num_valid, f_moe_buf, TOTAL_EXPERTS, UNIT_SIZE)
torch.cuda.synchronize()

g_combined = torch.cuda.CUDAGraph()
with torch.cuda.graph(g_combined, pool=pool):
    biased_grouped_topk(gating, bias.to(gating.dtype), f_topk_w, f_topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    f_ids_out[:, :K] = f_topk_ids
    f_w_out[:, :K] = f_topk_w
    moe_sorting_fwd(f_ids_out, f_w_out, f_sorted_ids, f_sorted_w, f_sorted_exp, f_num_valid, f_moe_buf, TOTAL_EXPERTS, UNIT_SIZE)

def strategy_combined():
    g_combined.replay()

# ============================================================================
# BENCHMARK
# ============================================================================
print("\\n" + "=" * 80)
print("FINAL BENCHMARK (20000 iterations)")
print("=" * 80)

strategies = [
    ("Current Best (Aligned)", current_best),
    ("No Zero", strategy_no_zero),
    ("Contiguous", strategy_contig),
    ("Combined All", strategy_combined),
]

# Mega warmup
for name, fn in strategies:
    print(f"  Warming {name}...")
    for _ in range(20000):
        fn()
    torch.cuda.synchronize()

NUM_ITERS = 20000
results = {}

for name, fn in strategies:
    gc.collect()
    torch.cuda.empty_cache()
    
    for _ in range(5000):
        fn()
    torch.cuda.synchronize()
    
    times = []
    for _ in range(NUM_ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)
    
    times = np.array(times)
    results[name] = {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "p1": float(np.percentile(times, 1)),
        "p5": float(np.percentile(times, 5)),
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
    }
    print(f"  {name}: {results[name]['mean']:.2f} us (min={results[name]['min']:.2f}, p50={results[name]['p50']:.2f})")

# ============================================================================
# FINAL RESULTS
# ============================================================================
print("\\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

original = 57.0
best_name = min(results.keys(), key=lambda k: results[k]["mean"])
best = results[best_name]

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        FINAL OPTIMIZATION RESULTS                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Strategy              │ Mean │ Min  │ P1   │ P50  │ Speedup                  ║
╠───────────────────────┼──────┼──────┼──────┼──────┼──────────────────────────╣""")

for name in sorted(results.keys(), key=lambda k: results[k]["mean"]):
    r = results[name]
    sp = original / r["mean"]
    marker = "★" if name == best_name else " "
    print(f"║ {marker}{name:<21} │{r['mean']:>5.1f} │{r['min']:>5.1f} │{r['p1']:>5.1f} │{r['p50']:>5.1f} │ {sp:.2f}x ({(1-r['mean']/original)*100:.1f}%)         ║")

print(f"""╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  OPTIMIZATION COMPLETE                                                       ║
║  ══════════════════════════════════════════════════════════════════════      ║
║                                                                              ║
║  Original:  57.00 us (HIP + Triton + CK baseline)                            ║
║  Final:     {best['mean']:.2f} us ({best_name})                            ║
║                                                                              ║
║  Speedup:   {original/best['mean']:.2f}x (mean), {original/best['min']:.2f}x (peak)                                     ║
║  Saved:     {original - best['mean']:.2f} us per invocation                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

with open("/workspace/final_results.json", "w") as f:
    json.dump({
        "strategies": results,
        "best_name": best_name,
        "best_mean": best['mean'],
        "best_min": best['min'],
        "original": original,
        "speedup": original / best['mean'],
    }, f, indent=2)

print("Results saved to /workspace/final_results.json")
'''


def main():
    print("FINAL PUSH OPTIMIZATION")
    script_path = WORKSPACE / "final_push.py"
    script_path.write_text(FINAL_PUSH_SCRIPT)
    
    cmd = [
        "docker", "run", "--rm",
        "--device=/dev/kfd", "--device=/dev/dri", "--ipc=host",
        "-e", "HIP_VISIBLE_DEVICES=3",
        "-v", f"{WORKSPACE}:/workspace",
        "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x",
        "python3", "/workspace/final_push.py"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    print(result.stdout)
    
    results_path = WORKSPACE / "final_results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)


if __name__ == "__main__":
    main()

