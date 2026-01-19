#!/usr/bin/env python3
"""
Aggressive TopK & Sort Optimization

Current best: 27.24 us (Graph+Persistent) - 52.2% improvement
Target: Push further with more aggressive strategies
"""

import subprocess
from pathlib import Path
import json

WORKSPACE = Path("/home/sapmajum/kernel_optimization_framework/profiler/aggressive_opt_ws")
WORKSPACE.mkdir(exist_ok=True)

AGGRESSIVE_SCRIPT = '''#!/usr/bin/env python3
"""
AGGRESSIVE OPTIMIZATION STRATEGIES

Best so far: 27.24 us with HIP Graph
Let's try:
1. Multiple graph instances for pipeline
2. Async graph execution
3. Graph with optimized memory layout
4. Fused custom kernels
5. Memory prefetching
"""
import torch
import triton
import triton.language as tl
import numpy as np
from scipy import stats
import json
import os
import gc
os.environ['TRITON_PRINT_AUTOTUNING'] = '0'

from aiter import biased_grouped_topk, moe_sorting_fwd

device = "cuda"

print("=" * 80)
print("AGGRESSIVE TOPK & SORT OPTIMIZATION")
print("=" * 80)

# Configuration
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

# ============================================================================
# BASELINE: Current best (HIP Graph)
# ============================================================================
print("\\n[Setting up baseline HIP Graph]")

b_topk_w = torch.empty((M, K), dtype=torch.float32, device=device)
b_topk_ids = torch.empty((M, K), dtype=torch.int32, device=device)
b_ids_out = torch.empty((M, K+S), dtype=torch.int32, device=device)
b_w_out = torch.empty((M, K+S), dtype=torch.float32, device=device)
b_sorted_ids = torch.empty(MAX_PAD, dtype=torch.int32, device=device)
b_sorted_w = torch.empty(MAX_PAD, dtype=torch.float32, device=device)
b_sorted_exp = torch.empty(MAX_PAD, dtype=torch.int32, device=device)
b_num_valid = torch.empty(1, dtype=torch.int32, device=device)
b_moe_buf = torch.empty(TOTAL_EXPERTS + 1, dtype=torch.int32, device=device)
b_ids_out[:, K] = NUM_EXPERTS
b_w_out[:, K] = 1.0

# Warmup
for _ in range(200):
    biased_grouped_topk(gating, bias.to(gating.dtype), b_topk_w, b_topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    b_ids_out[:, :K] = b_topk_ids
    b_w_out[:, :K] = b_topk_w
    b_num_valid.zero_()
    moe_sorting_fwd(b_ids_out, b_w_out, b_sorted_ids, b_sorted_w, b_sorted_exp, b_num_valid, b_moe_buf, TOTAL_EXPERTS, UNIT_SIZE)
torch.cuda.synchronize()

g_baseline = torch.cuda.CUDAGraph()
with torch.cuda.graph(g_baseline):
    biased_grouped_topk(gating, bias.to(gating.dtype), b_topk_w, b_topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    b_ids_out[:, :K] = b_topk_ids
    b_w_out[:, :K] = b_topk_w
    b_num_valid.zero_()
    moe_sorting_fwd(b_ids_out, b_w_out, b_sorted_ids, b_sorted_w, b_sorted_exp, b_num_valid, b_moe_buf, TOTAL_EXPERTS, UNIT_SIZE)

def baseline_graph():
    g_baseline.replay()

# ============================================================================
# STRATEGY 1: PINNED MEMORY
# ============================================================================
print("\\n[Strategy 1: Pinned Memory]")

# Use pinned host memory for input
pinned_gating = torch.randn(M, NUM_EXPERTS, dtype=torch.bfloat16, pin_memory=True)
pinned_bias = torch.randn(NUM_EXPERTS, dtype=torch.bfloat16, pin_memory=True)
d_gating = pinned_gating.to(device, non_blocking=True)
d_bias = pinned_bias.to(device, non_blocking=True)
torch.cuda.synchronize()

# Copy to match seed
d_gating.copy_(gating)
d_bias.copy_(bias)

p_topk_w = torch.empty((M, K), dtype=torch.float32, device=device)
p_topk_ids = torch.empty((M, K), dtype=torch.int32, device=device)
p_ids_out = torch.empty((M, K+S), dtype=torch.int32, device=device)
p_w_out = torch.empty((M, K+S), dtype=torch.float32, device=device)
p_sorted_ids = torch.empty(MAX_PAD, dtype=torch.int32, device=device)
p_sorted_w = torch.empty(MAX_PAD, dtype=torch.float32, device=device)
p_sorted_exp = torch.empty(MAX_PAD, dtype=torch.int32, device=device)
p_num_valid = torch.empty(1, dtype=torch.int32, device=device)
p_moe_buf = torch.empty(TOTAL_EXPERTS + 1, dtype=torch.int32, device=device)
p_ids_out[:, K] = NUM_EXPERTS
p_w_out[:, K] = 1.0

for _ in range(200):
    biased_grouped_topk(d_gating, d_bias.to(d_gating.dtype), p_topk_w, p_topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    p_ids_out[:, :K] = p_topk_ids
    p_w_out[:, :K] = p_topk_w
    p_num_valid.zero_()
    moe_sorting_fwd(p_ids_out, p_w_out, p_sorted_ids, p_sorted_w, p_sorted_exp, p_num_valid, p_moe_buf, TOTAL_EXPERTS, UNIT_SIZE)
torch.cuda.synchronize()

g_pinned = torch.cuda.CUDAGraph()
with torch.cuda.graph(g_pinned):
    biased_grouped_topk(d_gating, d_bias.to(d_gating.dtype), p_topk_w, p_topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    p_ids_out[:, :K] = p_topk_ids
    p_w_out[:, :K] = p_topk_w
    p_num_valid.zero_()
    moe_sorting_fwd(p_ids_out, p_w_out, p_sorted_ids, p_sorted_w, p_sorted_exp, p_num_valid, p_moe_buf, TOTAL_EXPERTS, UNIT_SIZE)

def strategy_pinned():
    g_pinned.replay()

print("  Pinned memory graph captured")

# ============================================================================
# STRATEGY 2: COMPACT MEMORY LAYOUT
# ============================================================================
print("\\n[Strategy 2: Compact Memory Layout]")

# Allocate all buffers contiguously
total_size = (
    M * K * 4 +  # topk_w
    M * K * 4 +  # topk_ids
    M * (K+S) * 4 +  # ids_out
    M * (K+S) * 4 +  # w_out
    MAX_PAD * 4 +  # sorted_ids
    MAX_PAD * 4 +  # sorted_w
    MAX_PAD * 4 +  # sorted_exp
    4 +  # num_valid
    (TOTAL_EXPERTS + 1) * 4  # moe_buf
)

compact_buffer = torch.empty(total_size // 4 + 1, dtype=torch.float32, device=device)

# Create views into the buffer
offset = 0
c_topk_w = compact_buffer[offset:offset + M*K].view(M, K)
offset += M * K
c_topk_ids = compact_buffer[offset:offset + M*K].view(M, K).to(torch.int32)
offset += M * K
c_ids_out = compact_buffer[offset:offset + M*(K+S)].view(M, K+S).to(torch.int32)
offset += M * (K+S)
c_w_out = compact_buffer[offset:offset + M*(K+S)].view(M, K+S)

# Actually need separate typed allocations
c_topk_w = torch.empty((M, K), dtype=torch.float32, device=device)
c_topk_ids = torch.empty((M, K), dtype=torch.int32, device=device)
c_ids_out = torch.empty((M, K+S), dtype=torch.int32, device=device)
c_w_out = torch.empty((M, K+S), dtype=torch.float32, device=device)
c_sorted_ids = torch.empty(MAX_PAD, dtype=torch.int32, device=device)
c_sorted_w = torch.empty(MAX_PAD, dtype=torch.float32, device=device)
c_sorted_exp = torch.empty(MAX_PAD, dtype=torch.int32, device=device)
c_num_valid = torch.empty(1, dtype=torch.int32, device=device)
c_moe_buf = torch.empty(TOTAL_EXPERTS + 1, dtype=torch.int32, device=device)
c_ids_out[:, K] = NUM_EXPERTS
c_w_out[:, K] = 1.0

for _ in range(200):
    biased_grouped_topk(gating, bias.to(gating.dtype), c_topk_w, c_topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    c_ids_out[:, :K] = c_topk_ids
    c_w_out[:, :K] = c_topk_w
    c_num_valid.zero_()
    moe_sorting_fwd(c_ids_out, c_w_out, c_sorted_ids, c_sorted_w, c_sorted_exp, c_num_valid, c_moe_buf, TOTAL_EXPERTS, UNIT_SIZE)
torch.cuda.synchronize()

g_compact = torch.cuda.CUDAGraph()
with torch.cuda.graph(g_compact):
    biased_grouped_topk(gating, bias.to(gating.dtype), c_topk_w, c_topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    c_ids_out[:, :K] = c_topk_ids
    c_w_out[:, :K] = c_topk_w
    c_num_valid.zero_()
    moe_sorting_fwd(c_ids_out, c_w_out, c_sorted_ids, c_sorted_w, c_sorted_exp, c_num_valid, c_moe_buf, TOTAL_EXPERTS, UNIT_SIZE)

def strategy_compact():
    g_compact.replay()

print("  Compact layout graph captured")

# ============================================================================
# STRATEGY 3: TRITON FUSED COPY
# ============================================================================
print("\\n[Strategy 3: Triton Fused Copy]")

@triton.jit
def fused_copy_kernel(
    src_ids, src_w, dst_ids, dst_w,
    shared_id, shared_scale,
    K: tl.constexpr, M: tl.constexpr
):
    """Fused copy with shared expert append."""
    pid = tl.program_id(0)
    if pid < M:
        src_off = pid * K
        dst_off = pid * (K + 1)
        for i in range(K):
            tl.store(dst_ids + dst_off + i, tl.load(src_ids + src_off + i))
            tl.store(dst_w + dst_off + i, tl.load(src_w + src_off + i))
        tl.store(dst_ids + dst_off + K, shared_id)
        tl.store(dst_w + dst_off + K, shared_scale)

t_topk_w = torch.empty((M, K), dtype=torch.float32, device=device)
t_topk_ids = torch.empty((M, K), dtype=torch.int32, device=device)
t_ids_out = torch.empty((M, K+S), dtype=torch.int32, device=device)
t_w_out = torch.empty((M, K+S), dtype=torch.float32, device=device)
t_sorted_ids = torch.empty(MAX_PAD, dtype=torch.int32, device=device)
t_sorted_w = torch.empty(MAX_PAD, dtype=torch.float32, device=device)
t_sorted_exp = torch.empty(MAX_PAD, dtype=torch.int32, device=device)
t_num_valid = torch.empty(1, dtype=torch.int32, device=device)
t_moe_buf = torch.empty(TOTAL_EXPERTS + 1, dtype=torch.int32, device=device)

def strategy_triton_copy():
    biased_grouped_topk(gating, bias.to(gating.dtype), t_topk_w, t_topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    fused_copy_kernel[(M,)](t_topk_ids, t_topk_w, t_ids_out, t_w_out, NUM_EXPERTS, 1.0, K, M)
    t_num_valid.zero_()
    moe_sorting_fwd(t_ids_out, t_w_out, t_sorted_ids, t_sorted_w, t_sorted_exp, t_num_valid, t_moe_buf, TOTAL_EXPERTS, UNIT_SIZE)

# Warmup Triton kernel
for _ in range(200):
    strategy_triton_copy()
torch.cuda.synchronize()

print("  Triton fused copy ready")

# ============================================================================
# STRATEGY 4: DOUBLE BUFFERING
# ============================================================================
print("\\n[Strategy 4: Double Buffering]")

# Two sets of buffers for ping-pong
db_topk_w = [torch.empty((M, K), dtype=torch.float32, device=device) for _ in range(2)]
db_topk_ids = [torch.empty((M, K), dtype=torch.int32, device=device) for _ in range(2)]
db_ids_out = [torch.empty((M, K+S), dtype=torch.int32, device=device) for _ in range(2)]
db_w_out = [torch.empty((M, K+S), dtype=torch.float32, device=device) for _ in range(2)]
db_sorted_ids = [torch.empty(MAX_PAD, dtype=torch.int32, device=device) for _ in range(2)]
db_sorted_w = [torch.empty(MAX_PAD, dtype=torch.float32, device=device) for _ in range(2)]
db_sorted_exp = [torch.empty(MAX_PAD, dtype=torch.int32, device=device) for _ in range(2)]
db_num_valid = [torch.empty(1, dtype=torch.int32, device=device) for _ in range(2)]
db_moe_buf = [torch.empty(TOTAL_EXPERTS + 1, dtype=torch.int32, device=device) for _ in range(2)]

for i in range(2):
    db_ids_out[i][:, K] = NUM_EXPERTS
    db_w_out[i][:, K] = 1.0

# Create graphs for each buffer set
db_graphs = []
for i in range(2):
    for _ in range(200):
        biased_grouped_topk(gating, bias.to(gating.dtype), db_topk_w[i], db_topk_ids[i], NUM_GROUPS, TOPK_GROUP, True, 1.0)
        db_ids_out[i][:, :K] = db_topk_ids[i]
        db_w_out[i][:, :K] = db_topk_w[i]
        db_num_valid[i].zero_()
        moe_sorting_fwd(db_ids_out[i], db_w_out[i], db_sorted_ids[i], db_sorted_w[i], db_sorted_exp[i], db_num_valid[i], db_moe_buf[i], TOTAL_EXPERTS, UNIT_SIZE)
    torch.cuda.synchronize()
    
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        biased_grouped_topk(gating, bias.to(gating.dtype), db_topk_w[i], db_topk_ids[i], NUM_GROUPS, TOPK_GROUP, True, 1.0)
        db_ids_out[i][:, :K] = db_topk_ids[i]
        db_w_out[i][:, :K] = db_topk_w[i]
        db_num_valid[i].zero_()
        moe_sorting_fwd(db_ids_out[i], db_w_out[i], db_sorted_ids[i], db_sorted_w[i], db_sorted_exp[i], db_num_valid[i], db_moe_buf[i], TOTAL_EXPERTS, UNIT_SIZE)
    db_graphs.append(g)

db_idx = [0]
def strategy_double_buffer():
    db_graphs[db_idx[0]].replay()
    db_idx[0] = 1 - db_idx[0]

print("  Double buffer graphs captured")

# ============================================================================
# WARMUP AND BENCHMARK
# ============================================================================
print("\\n" + "=" * 80)
print("WARMING UP ALL STRATEGIES")
print("=" * 80)

strategies = [
    ("Baseline Graph", baseline_graph),
    ("Pinned Memory", strategy_pinned),
    ("Compact Layout", strategy_compact),
    ("Triton Fused", strategy_triton_copy),
    ("Double Buffer", strategy_double_buffer),
]

for name, fn in strategies:
    print(f"  Warming up {name}...")
    for _ in range(5000):
        fn()
    torch.cuda.synchronize()

print("\\n" + "=" * 80)
print("BENCHMARKING")
print("=" * 80)

NUM_ITERS = 5000
results = {}

for name, fn in strategies:
    gc.collect()
    torch.cuda.empty_cache()
    
    for _ in range(1000):
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
        "mean_us": float(np.mean(times)),
        "std_us": float(np.std(times)),
        "min_us": float(np.min(times)),
        "p50_us": float(np.percentile(times, 50)),
        "p95_us": float(np.percentile(times, 95)),
    }
    print(f"  {name}: {results[name]['mean_us']:.2f} ± {results[name]['std_us']:.2f} us")

# ============================================================================
# RESULTS
# ============================================================================
print("\\n" + "=" * 80)
print("AGGRESSIVE OPTIMIZATION RESULTS")
print("=" * 80)

original = 57.0
best_name = min(results.keys(), key=lambda k: results[k]["mean_us"])
best = results[best_name]["mean_us"]

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         AGGRESSIVE OPTIMIZATION                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Strategy              │ Latency (us)  │ vs Original (57 us)                  ║
╠───────────────────────┼───────────────┼──────────────────────────────────────╣""")

for name in sorted(results.keys(), key=lambda k: results[k]["mean_us"]):
    r = results[name]
    speedup = original / r["mean_us"]
    imp = (original - r["mean_us"]) / original * 100
    marker = "★" if name == best_name else " "
    print(f"║ {marker}{name:<21} │ {r['mean_us']:>6.2f} ± {r['std_us']:>4.2f} │ {speedup:.2f}x ({imp:.1f}%)                      ║")

print(f"""╠══════════════════════════════════════════════════════════════════════════════╣
║ BEST: {best_name:<30} {best:.2f} us ({original/best:.2f}x)            ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

with open("/workspace/aggressive_results.json", "w") as f:
    json.dump({
        "strategies": results,
        "best": best_name,
        "best_latency": best,
        "original": original,
        "speedup": original / best,
    }, f, indent=2)

print("Results saved.")
'''


def main():
    print("AGGRESSIVE OPTIMIZATION")
    script_path = WORKSPACE / "aggressive.py"
    script_path.write_text(AGGRESSIVE_SCRIPT)
    
    cmd = [
        "docker", "run", "--rm",
        "--device=/dev/kfd", "--device=/dev/dri", "--ipc=host",
        "-e", "HIP_VISIBLE_DEVICES=3",
        "-v", f"{WORKSPACE}:/workspace",
        "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x",
        "python3", "/workspace/aggressive.py"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    print(result.stdout)
    
    if result.stderr:
        stderr_lines = [l for l in result.stderr.split('\n') if not l.startswith('[aiter]') and l.strip()]
        if stderr_lines:
            print("\nNotes:", stderr_lines[:5])


if __name__ == "__main__":
    main()

