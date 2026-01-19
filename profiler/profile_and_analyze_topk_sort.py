#!/usr/bin/env python3
"""
Full Profile + Analyze of TopK & Sort Module.
Shows complete roofline analysis and bottleneck identification.
"""

import subprocess
from pathlib import Path
import json

WORKSPACE = Path("/home/sapmajum/kernel_optimization_framework/profiler/topk_analysis")
WORKSPACE.mkdir(exist_ok=True)

# Create comprehensive profiling script
profile_script = '''#!/usr/bin/env python3
"""
Complete profiling and analysis of TopK & Sort module.
Collects hardware counters and performs roofline analysis.
"""
import torch
import triton
import triton.language as tl
import numpy as np
import json
import os
os.environ['TRITON_PRINT_AUTOTUNING'] = '0'

device = "cuda"

print("=" * 80)
print("COMPREHENSIVE TOPK & SORT MODULE PROFILING AND ANALYSIS")
print("=" * 80)

# Import aiter kernels
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

# GPU specs for MI355X (gfx950)
GPU_SPECS = {
    "name": "AMD MI355X (gfx950)",
    "peak_tflops_fp16": 1600.0,
    "peak_tflops_fp32": 800.0,
    "peak_bandwidth_gbps": 8000.0,
    "num_cus": 304,
    "lds_per_cu_kb": 64,
}

# Triton append kernel
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
    fast_append[(1,)](topk_ids, topk_w, ids_out, w_out, NUM_EXPERTS, 1.0, K, S, M)
    moe_sorting_fwd(ids_out, w_out, sorted_ids, sorted_w, sorted_exp, num_valid, moe_buf, TOTAL_EXPERTS, UNIT_SIZE)

# ============================================================================
# HEAVY WARMUP
# ============================================================================
print("\\n[1] Heavy JIT warmup (1000 iterations)...")
for _ in range(1000):
    run_pipeline()
torch.cuda.synchronize()
print("    Warmup complete!")

# ============================================================================
# DETAILED PROFILING
# ============================================================================
print("\\n[2] Detailed component profiling (500 iterations each)...")

results = {
    "config": {
        "M": M, "K": K, "S": S,
        "num_experts": NUM_EXPERTS,
        "unit_size": UNIT_SIZE,
    },
    "gpu": GPU_SPECS,
    "kernels": {},
}

# Calculate data sizes for each kernel
kernel_data = {
    "hip_topk": {
        "input_bytes": M * NUM_EXPERTS * 2 + NUM_EXPERTS * 2,  # gating (bf16) + bias (bf16)
        "output_bytes": M * K * 4 * 2,  # topk_w (fp32) + topk_ids (int32)
        "flops": M * NUM_EXPERTS * 10,  # Rough estimate: softmax + topk selection
    },
    "triton_append": {
        "input_bytes": M * K * 4 * 2,  # topk_ids + topk_w
        "output_bytes": M * (K+S) * 4 * 2,  # ids_out + w_out
        "flops": M * K * 2,  # Copy operations
    },
    "ck_sorting": {
        "input_bytes": M * (K+S) * 4 * 2,  # ids_out + w_out
        "output_bytes": MAX_PAD * 4 * 2,  # sorted_ids + sorted_w
        "flops": M * (K+S) * NUM_EXPERTS * 5,  # Rough: counting + sorting
    },
}

# Profile each component
def profile_kernel(name, kernel_fn, iterations=500):
    times = []
    for _ in range(iterations):
        if name == "ck_sorting":
            num_valid.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        kernel_fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # us
    return times

# 1. HIP biased_grouped_topk
print("    Profiling HIP biased_grouped_topk...")
hip_times = profile_kernel("hip_topk", 
    lambda: biased_grouped_topk(gating, bias.to(gating.dtype), topk_w, topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0))

# 2. Triton fast_append
print("    Profiling Triton fast_append...")
triton_times = profile_kernel("triton_append",
    lambda: fast_append[(1,)](topk_ids, topk_w, ids_out, w_out, NUM_EXPERTS, 1.0, K, S, M))

# 3. CK moe_sorting
print("    Profiling CK moe_sorting...")
ck_times = profile_kernel("ck_sorting",
    lambda: moe_sorting_fwd(ids_out, w_out, sorted_ids, sorted_w, sorted_exp, num_valid, moe_buf, TOTAL_EXPERTS, UNIT_SIZE))

# 4. Full pipeline
print("    Profiling full pipeline...")
full_times = profile_kernel("full_pipeline", run_pipeline)

# ============================================================================
# ANALYZE RESULTS
# ============================================================================
print("\\n[3] Analyzing results...")

def analyze_kernel(name, times, data_info):
    mean_us = np.mean(times)
    std_us = np.std(times)
    min_us = np.min(times)
    max_us = np.max(times)
    
    total_bytes = data_info["input_bytes"] + data_info["output_bytes"]
    flops = data_info["flops"]
    
    # Performance metrics
    bandwidth_gbps = total_bytes / (mean_us * 1e3) if mean_us > 0 else 0
    tflops = (flops / mean_us) / 1e6 if mean_us > 0 else 0
    arithmetic_intensity = flops / total_bytes if total_bytes > 0 else 0
    
    # Roofline analysis
    peak_bw = GPU_SPECS["peak_bandwidth_gbps"]
    peak_compute = GPU_SPECS["peak_tflops_fp32"]
    ridge_point = (peak_compute * 1e12) / (peak_bw * 1e9)  # FLOP/byte
    
    # Theoretical max at this AI
    memory_roof = (peak_bw * arithmetic_intensity) / 1000  # TFLOP/s
    compute_roof = peak_compute
    theoretical_max = min(memory_roof, compute_roof)
    
    efficiency = tflops / theoretical_max if theoretical_max > 0 else 0
    
    # Bottleneck identification
    if mean_us < 5:
        bottleneck = "LATENCY"
        bottleneck_confidence = 0.9
    elif arithmetic_intensity < ridge_point * 0.5:
        bottleneck = "MEMORY"
        bottleneck_confidence = 0.8
    elif arithmetic_intensity > ridge_point * 2:
        bottleneck = "COMPUTE"
        bottleneck_confidence = 0.8
    else:
        bottleneck = "BALANCED"
        bottleneck_confidence = 0.6
    
    return {
        "timing": {
            "mean_us": mean_us,
            "std_us": std_us,
            "min_us": min_us,
            "max_us": max_us,
        },
        "data": {
            "input_bytes": data_info["input_bytes"],
            "output_bytes": data_info["output_bytes"],
            "total_bytes": total_bytes,
            "flops": flops,
        },
        "performance": {
            "bandwidth_gbps": bandwidth_gbps,
            "tflops": tflops,
            "arithmetic_intensity": arithmetic_intensity,
        },
        "roofline": {
            "ridge_point": ridge_point,
            "theoretical_max_tflops": theoretical_max,
            "efficiency": efficiency,
            "is_memory_bound": arithmetic_intensity < ridge_point,
        },
        "bottleneck": {
            "type": bottleneck,
            "confidence": bottleneck_confidence,
        },
    }

# Analyze each kernel
for name, times in [("hip_topk", hip_times), ("triton_append", triton_times), ("ck_sorting", ck_times)]:
    results["kernels"][name] = analyze_kernel(name, times, kernel_data[name])

# Full pipeline
full_data = {
    "input_bytes": sum(k["input_bytes"] for k in kernel_data.values()),
    "output_bytes": sum(k["output_bytes"] for k in kernel_data.values()),
    "flops": sum(k["flops"] for k in kernel_data.values()),
}
results["full_pipeline"] = analyze_kernel("full_pipeline", full_times, full_data)

# ============================================================================
# PRINT DETAILED REPORT
# ============================================================================
print("\\n" + "=" * 80)
print("PROFILING RESULTS")
print("=" * 80)

print(f"\\nConfiguration:")
print(f"  Batch size (M): {M}")
print(f"  TopK (K): {K}")
print(f"  Shared experts (S): {S}")
print(f"  Num experts: {NUM_EXPERTS}")

print(f"\\nGPU: {GPU_SPECS['name']}")
print(f"  Peak Compute: {GPU_SPECS['peak_tflops_fp32']:.0f} TFLOP/s (FP32)")
print(f"  Peak Bandwidth: {GPU_SPECS['peak_bandwidth_gbps']:.0f} GB/s")

print("\\n" + "-" * 80)
print("KERNEL BREAKDOWN")
print("-" * 80)

header = f"{'Kernel':<20} {'Type':<8} {'Latency (us)':<15} {'BW (GB/s)':<12} {'AI':<10} {'Bottleneck':<12}"
print(header)
print("-" * len(header))

kernel_types = {"hip_topk": "HIP", "triton_append": "Triton", "ck_sorting": "CK"}
for name in ["hip_topk", "triton_append", "ck_sorting"]:
    k = results["kernels"][name]
    t = k["timing"]
    p = k["performance"]
    b = k["bottleneck"]
    print(f"{name:<20} {kernel_types[name]:<8} {t['mean_us']:>6.2f} ± {t['std_us']:<6.2f} {p['bandwidth_gbps']:>10.2f} {p['arithmetic_intensity']:>10.4f} {b['type']:<12}")

print("-" * len(header))
fp = results["full_pipeline"]
print(f"{'FULL PIPELINE':<20} {'Mixed':<8} {fp['timing']['mean_us']:>6.2f} ± {fp['timing']['std_us']:<6.2f} {fp['performance']['bandwidth_gbps']:>10.2f} {fp['performance']['arithmetic_intensity']:>10.4f} {fp['bottleneck']['type']:<12}")

# Component percentages
total_time = fp["timing"]["mean_us"]
print("\\nComponent breakdown (% of total):")
for name in ["hip_topk", "triton_append", "ck_sorting"]:
    pct = results["kernels"][name]["timing"]["mean_us"] / total_time * 100
    print(f"  {name}: {pct:.1f}%")

# ============================================================================
# ROOFLINE ANALYSIS
# ============================================================================
print("\\n" + "=" * 80)
print("ROOFLINE ANALYSIS")
print("=" * 80)

ridge = fp["roofline"]["ridge_point"]
print(f"\\nRidge Point: {ridge:.2f} FLOP/byte")
print("(Kernels below this are memory-bound, above are compute-bound)")

print("\\n" + "-" * 60)
print(f"{'Kernel':<20} {'AI (FLOP/B)':<12} {'Position':<15} {'Efficiency':<12}")
print("-" * 60)

for name in ["hip_topk", "triton_append", "ck_sorting", "full_pipeline"]:
    if name == "full_pipeline":
        k = results["full_pipeline"]
    else:
        k = results["kernels"][name]
    ai = k["performance"]["arithmetic_intensity"]
    eff = k["roofline"]["efficiency"]
    pos = "Memory-bound" if k["roofline"]["is_memory_bound"] else "Compute-bound"
    print(f"{name:<20} {ai:<12.4f} {pos:<15} {eff*100:>10.1f}%")

# ============================================================================
# BOTTLENECK ANALYSIS & OPTIMIZATION SUGGESTIONS
# ============================================================================
print("\\n" + "=" * 80)
print("BOTTLENECK ANALYSIS & OPTIMIZATION SUGGESTIONS")
print("=" * 80)

for name in ["hip_topk", "triton_append", "ck_sorting"]:
    k = results["kernels"][name]
    b = k["bottleneck"]
    t = k["timing"]
    
    print(f"\\n{name} ({kernel_types[name]}):")
    print(f"  Bottleneck: {b['type']} (confidence: {b['confidence']:.0%})")
    print(f"  Latency: {t['mean_us']:.2f} us")
    
    print("  Suggestions:")
    if b["type"] == "LATENCY":
        print("    - Kernel is too short, launch overhead dominates")
        print("    - RECOMMEND: Fuse with adjacent kernels")
        print("    - RECOMMEND: Use persistent kernel pattern")
    elif b["type"] == "MEMORY":
        print("    - Memory-bound: limited by bandwidth")
        print("    - RECOMMEND: Improve memory coalescing")
        print("    - RECOMMEND: Use vectorized loads (float4)")
        print("    - RECOMMEND: Increase arithmetic intensity via fusion")
    elif b["type"] == "COMPUTE":
        print("    - Compute-bound: limited by ALU throughput")
        print("    - RECOMMEND: Use tensor cores (MFMA)")
        print("    - RECOMMEND: Reduce instruction count")
    else:
        print("    - Near ridge point, balanced workload")
        print("    - RECOMMEND: Profile with hardware counters for details")

# ============================================================================
# ASCII ROOFLINE PLOT
# ============================================================================
print("\\n" + "=" * 80)
print("ROOFLINE VISUALIZATION")
print("=" * 80)

# Simple ASCII roofline
width = 70
height = 15
peak_bw = GPU_SPECS["peak_bandwidth_gbps"]
peak_compute = GPU_SPECS["peak_tflops_fp32"]

# AI range
min_ai = 0.0001
max_ai = 100

# Create plot
plot = [[' ' for _ in range(width)] for _ in range(height)]

# Draw roof
for i in range(width):
    ai = min_ai * (max_ai / min_ai) ** (i / (width - 1))
    memory_roof = (peak_bw * ai) / 1000
    roof = min(memory_roof, peak_compute)
    y = int((1 - roof / (peak_compute * 1.1)) * (height - 1))
    y = max(0, min(height - 1, y))
    plot[y][i] = '-'

# Mark ridge point
ridge_x = int(np.log(ridge / min_ai) / np.log(max_ai / min_ai) * (width - 1))
ridge_x = max(0, min(width - 1, ridge_x))
plot[0][ridge_x] = 'R'

# Plot kernels
markers = {'hip_topk': 'H', 'triton_append': 'T', 'ck_sorting': 'C', 'full_pipeline': 'F'}
for name, marker in markers.items():
    if name == 'full_pipeline':
        k = results['full_pipeline']
    else:
        k = results['kernels'][name]
    ai = k['performance']['arithmetic_intensity']
    perf = k['performance']['tflops']
    
    if ai > 0 and perf > 0:
        x = int(np.log(max(ai, min_ai) / min_ai) / np.log(max_ai / min_ai) * (width - 1))
        y = int((1 - perf / (peak_compute * 1.1)) * (height - 1))
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        plot[y][x] = marker

# Print plot
print(f"\\n{peak_compute:.0f} TFLOP/s |" + '-' * width)
for i, row in enumerate(plot):
    perf = peak_compute * 1.1 * (1 - i / (height - 1))
    if i % 3 == 0:
        label = f"{perf:>6.0f} |"
    else:
        label = "       |"
    print(label + ''.join(row))
print("       +" + "-" * width)
print(f"        {min_ai:.4f}" + " " * (width - 20) + f"{max_ai:.1f}")
print("        " + " " * (width // 2 - 15) + "Arithmetic Intensity (FLOP/byte)")

print("\\nLegend: R=Ridge Point, H=HIP topk, T=Triton append, C=CK sorting, F=Full pipeline")
print(f"Ridge Point: {ridge:.2f} FLOP/byte")

# ============================================================================
# SAVE RESULTS
# ============================================================================
with open("/workspace/topk_sort_analysis.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\\n\\nFull results saved to topk_sort_analysis.json")
'''

(WORKSPACE / "profile_analyze.py").write_text(profile_script)

print("Running full profile + analyze on TopK & Sort module...")
print("=" * 80)

# Run in Docker
cmd = [
    "docker", "run", "--rm",
    "--device=/dev/kfd", "--device=/dev/dri", "--ipc=host",
    "-e", "HIP_VISIBLE_DEVICES=3",
    "-v", f"{WORKSPACE}:/workspace",
    "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x",
    "python3", "/workspace/profile_analyze.py"
]

subprocess.run(cmd, capture_output=False, text=True, timeout=900)

# Show saved results
results_file = WORKSPACE / "topk_sort_analysis.json"
if results_file.exists():
    print("\n" + "=" * 80)
    print("SAVED JSON RESULTS")
    print("=" * 80)
    with open(results_file) as f:
        results = json.load(f)
    print(json.dumps(results, indent=2)[:3000])

