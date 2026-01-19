#!/usr/bin/env python3
"""
Comprehensive TopK & Sort Module Profiling

This script performs full profiling including:
- Detailed timing metrics (mean, std, percentiles)
- Memory bandwidth analysis
- Roofline model analysis
- Speed of Light (SOL) analysis
- Bottleneck identification
- Optimization recommendations
"""

import subprocess
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from profiler.generic_profiler import GenericProfiler, BottleneckType
from profiler.roofline import RooflineModel
from profiler.bottleneck_analyzer import BottleneckAnalyzer

WORKSPACE = Path("/home/sapmajum/kernel_optimization_framework/profiler/comprehensive_ws_v2")
WORKSPACE.mkdir(exist_ok=True)

# MI355X (gfx950) specifications for SOL analysis
GPU_SPECS = {
    "name": "AMD MI355X (gfx950)",
    "peak_tflops_fp16": 1600.0,
    "peak_tflops_fp32": 800.0,
    "peak_bandwidth_gbps": 8000.0,
    "num_cus": 304,
    "simd_width": 64,
    "waves_per_cu": 32,
    "max_waves": 304 * 32,
    "lds_per_cu_kb": 64,
    "l2_cache_mb": 256,
    "clock_mhz": 2500,
    "l1_bandwidth_gbps": 47000,  # Per-CU L1 bandwidth * CUs
    "lds_bandwidth_gbps": 98000,  # LDS bandwidth
}

COMPREHENSIVE_PROFILE_SCRIPT = '''#!/usr/bin/env python3
"""
Comprehensive profiling script for TopK & Sort module.
Collects detailed metrics for roofline and SOL analysis.
"""
import torch
import triton
import triton.language as tl
import numpy as np
import json
import os
import gc
os.environ['TRITON_PRINT_AUTOTUNING'] = '0'

device = "cuda"

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
print(f"  Unit size: {UNIT_SIZE}")
print(f"  Max pad: {MAX_PAD}")

# Define kernels
@triton.jit
def fast_append_kernel(
    ids_in, w_in, ids_out, w_out,
    num_experts, scale,
    K: tl.constexpr, S: tl.constexpr, M: tl.constexpr
):
    """Optimized single-block append kernel."""
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

# Pre-compile Triton kernel
fast_append_kernel[(1,)](topk_ids, topk_w, ids_out, w_out, NUM_EXPERTS, 1.0, K, S, M)
torch.cuda.synchronize()

def run_hip_topk():
    biased_grouped_topk(gating, bias.to(gating.dtype), topk_w, topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)

def run_triton_append():
    fast_append_kernel[(1,)](topk_ids, topk_w, ids_out, w_out, NUM_EXPERTS, 1.0, K, S, M)

def run_ck_sorting():
    num_valid.zero_()
    moe_sorting_fwd(ids_out, w_out, sorted_ids, sorted_w, sorted_exp, num_valid, moe_buf, TOTAL_EXPERTS, UNIT_SIZE)

def run_full_pipeline():
    run_hip_topk()
    run_triton_append()
    run_ck_sorting()

# Heavy warmup
print("\\n" + "=" * 80)
print("PHASE 1: HEAVY JIT WARMUP (3000 iterations)")
print("=" * 80)
for i in range(3000):
    run_full_pipeline()
    if (i+1) % 500 == 0:
        torch.cuda.synchronize()
        gc.collect()
        print(f"  Progress: {i+1}/3000")
torch.cuda.synchronize()
print("Warmup complete.")

# Detailed profiling parameters
NUM_WARMUP = 500  # Per-component warmup
NUM_ITERS = 2000  # More iterations for better statistics

results = {
    "config": {
        "M": M, "K": K, "S": S,
        "num_experts": NUM_EXPERTS,
        "total_experts": TOTAL_EXPERTS,
        "unit_size": UNIT_SIZE,
        "max_pad": MAX_PAD,
        "num_iterations": NUM_ITERS,
    },
    "components": {},
    "memory_analysis": {},
    "compute_analysis": {},
}

print("\\n" + "=" * 80)
print("PHASE 2: DETAILED COMPONENT PROFILING")
print("=" * 80)

def profile_kernel(name, kernel_fn, input_bytes, output_bytes, estimated_flops, kernel_type):
    """Profile a single kernel with detailed metrics."""
    print(f"\\n[Profiling {name}]")
    
    # Per-component warmup
    for _ in range(NUM_WARMUP):
        kernel_fn()
    torch.cuda.synchronize()
    
    # Collect timing samples
    times = []
    for _ in range(NUM_ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        kernel_fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # us
    
    times = np.array(times)
    total_bytes = input_bytes + output_bytes
    
    # Calculate metrics
    mean_us = float(np.mean(times))
    std_us = float(np.std(times))
    
    # Bandwidth (GB/s)
    achieved_bw = total_bytes / (mean_us * 1e3) if mean_us > 0 else 0
    
    # Throughput (TFLOP/s) - if compute-heavy
    achieved_tflops = (estimated_flops / mean_us) / 1e6 if mean_us > 0 and estimated_flops > 0 else 0
    
    # Arithmetic intensity (FLOP/byte)
    ai = estimated_flops / total_bytes if total_bytes > 0 else 0
    
    data = {
        "type": kernel_type,
        "timing": {
            "mean_us": mean_us,
            "std_us": std_us,
            "min_us": float(np.min(times)),
            "max_us": float(np.max(times)),
            "p25_us": float(np.percentile(times, 25)),
            "p50_us": float(np.percentile(times, 50)),
            "p75_us": float(np.percentile(times, 75)),
            "p90_us": float(np.percentile(times, 90)),
            "p95_us": float(np.percentile(times, 95)),
            "p99_us": float(np.percentile(times, 99)),
            "variance": float(np.var(times)),
            "cv": float(std_us / mean_us * 100) if mean_us > 0 else 0,  # Coefficient of variation %
        },
        "memory": {
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "total_bytes": total_bytes,
            "achieved_bandwidth_gbps": achieved_bw,
        },
        "compute": {
            "estimated_flops": estimated_flops,
            "achieved_tflops": achieved_tflops,
            "arithmetic_intensity": ai,
        },
        "raw_times": times.tolist()[-100:],  # Last 100 samples for analysis
    }
    
    print(f"  Latency: {mean_us:.2f} ± {std_us:.2f} us (CV: {data['timing']['cv']:.1f}%)")
    print(f"  P50/P95/P99: {data['timing']['p50_us']:.2f} / {data['timing']['p95_us']:.2f} / {data['timing']['p99_us']:.2f} us")
    print(f"  Bandwidth: {achieved_bw:.2f} GB/s")
    print(f"  AI: {ai:.4f} FLOP/byte")
    
    return data

# 1. HIP biased_grouped_topk
# Input: gating (M*NUM_EXPERTS*2) + bias (NUM_EXPERTS*2)
# Output: topk_w (M*K*4) + topk_ids (M*K*4)
# FLOPs: softmax (~3*M*NUM_EXPERTS) + topk selection (~M*NUM_EXPERTS*log2(K))
hip_input = M * NUM_EXPERTS * 2 + NUM_EXPERTS * 2
hip_output = M * K * 4 + M * K * 4
hip_flops = 3 * M * NUM_EXPERTS + M * NUM_EXPERTS * np.log2(K) * 2  # Approximate
results["components"]["hip_topk"] = profile_kernel(
    "HIP biased_grouped_topk", run_hip_topk, hip_input, hip_output, int(hip_flops), "HIP"
)

# 2. Triton fast_append
# Input: topk_ids (M*K*4) + topk_w (M*K*4)
# Output: ids_out (M*(K+S)*4) + w_out (M*(K+S)*4)
# FLOPs: negligible (pure memory operation)
triton_input = M * K * 4 + M * K * 4
triton_output = M * (K + S) * 4 + M * (K + S) * 4
triton_flops = 0
results["components"]["triton_append"] = profile_kernel(
    "Triton fast_append", run_triton_append, triton_input, triton_output, triton_flops, "Triton"
)

# 3. CK moe_sorting (2-phase)
# Input: ids_out (M*(K+S)*4) + w_out (M*(K+S)*4)
# Output: sorted_ids (MAX_PAD*4) + sorted_w (MAX_PAD*4) + sorted_exp (MAX_PAD*4) + moe_buf
# FLOPs: sorting ~O(N*log(N)) comparisons
ck_input = M * (K + S) * 4 * 2
ck_output = MAX_PAD * 4 * 3 + (TOTAL_EXPERTS + 1) * 4
ck_n = M * (K + S)
ck_flops = int(ck_n * np.log2(max(ck_n, 2)) * 2)  # Approximate sorting FLOPs
results["components"]["ck_sorting"] = profile_kernel(
    "CK moe_sorting", run_ck_sorting, ck_input, ck_output, ck_flops, "CK"
)

# 4. Full pipeline
print("\\n[Profiling Full Pipeline]")
for _ in range(NUM_WARMUP):
    num_valid.zero_()
    run_full_pipeline()
torch.cuda.synchronize()

full_times = []
for _ in range(NUM_ITERS):
    num_valid.zero_()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    run_full_pipeline()
    end.record()
    torch.cuda.synchronize()
    full_times.append(start.elapsed_time(end) * 1000)

full_times = np.array(full_times)
results["full_pipeline"] = {
    "timing": {
        "mean_us": float(np.mean(full_times)),
        "std_us": float(np.std(full_times)),
        "min_us": float(np.min(full_times)),
        "max_us": float(np.max(full_times)),
        "p50_us": float(np.percentile(full_times, 50)),
        "p95_us": float(np.percentile(full_times, 95)),
        "p99_us": float(np.percentile(full_times, 99)),
        "cv": float(np.std(full_times) / np.mean(full_times) * 100),
    },
    "total_bytes": hip_input + hip_output + triton_input + triton_output + ck_input + ck_output,
    "total_flops": int(hip_flops + triton_flops + ck_flops),
}

print(f"  Latency: {results['full_pipeline']['timing']['mean_us']:.2f} ± {results['full_pipeline']['timing']['std_us']:.2f} us")

# Summary
print("\\n" + "=" * 80)
print("PROFILING SUMMARY")
print("=" * 80)

total = results["full_pipeline"]["timing"]["mean_us"]
print(f"\\n{'Component':<25} {'Type':<8} {'Latency (us)':<18} {'% Total':<10} {'BW (GB/s)':<12} {'AI':<10}")
print("-" * 90)
for name, data in results["components"].items():
    t = data["timing"]["mean_us"]
    pct = t / total * 100
    bw = data["memory"]["achieved_bandwidth_gbps"]
    ai = data["compute"]["arithmetic_intensity"]
    print(f"{name:<25} {data['type']:<8} {t:>6.2f} ± {data['timing']['std_us']:>5.2f}   {pct:>6.1f}%     {bw:>8.2f}     {ai:>8.4f}")

print("-" * 90)
print(f"{'Full Pipeline':<25} {'mixed':<8} {total:>6.2f} ± {results['full_pipeline']['timing']['std_us']:>5.2f}   {'100.0%':>8}")

# Save results
with open("/workspace/comprehensive_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\\n" + "=" * 80)
print("Results saved to /workspace/comprehensive_results.json")
print("=" * 80)
'''


def run_comprehensive_profiling():
    """Run comprehensive profiling and analysis."""
    print("=" * 80)
    print("COMPREHENSIVE TOPK & SORT MODULE ANALYSIS")
    print("=" * 80)
    print("\nRunning comprehensive profiling with roofline and SOL analysis...")
    print("(This includes heavy warmup and 2000 iterations per component)\n")
    
    # Write and run script (avoid name conflict with Python's profile module)
    script_path = WORKSPACE / "run_profiling.py"
    script_path.write_text(COMPREHENSIVE_PROFILE_SCRIPT)
    
    cmd = [
        "docker", "run", "--rm",
        "--device=/dev/kfd", "--device=/dev/dri", "--ipc=host",
        "-e", "HIP_VISIBLE_DEVICES=3",
        "-v", f"{WORKSPACE}:/workspace",
        "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x",
        "python3", "/workspace/run_profiling.py"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    print(result.stdout)
    
    if result.stderr:
        stderr_lines = [l for l in result.stderr.split('\n') if not l.startswith('[aiter]') and l.strip()]
        if stderr_lines:
            print("\nNotes:", stderr_lines[:5])
    
    # Load results
    results_path = WORKSPACE / "comprehensive_results.json"
    if not results_path.exists():
        print("ERROR: No results produced!")
        return None
    
    with open(results_path) as f:
        results = json.load(f)
    
    return results


def analyze_roofline(results):
    """Perform roofline analysis on profiling results."""
    print("\n" + "=" * 80)
    print("ROOFLINE ANALYSIS")
    print("=" * 80)
    
    model = RooflineModel(gpu_arch="gfx950")
    
    for name, data in results["components"].items():
        if data["compute"]["estimated_flops"] > 0:
            try:
                model.add_kernel(
                    name=name,
                    flops=data["compute"]["estimated_flops"],
                    memory_bytes=data["memory"]["total_bytes"],
                    duration_us=data["timing"]["mean_us"],
                )
            except ValueError:
                # Handle zero/negative values
                model.add_point(
                    name=name,
                    arithmetic_intensity=data["compute"]["arithmetic_intensity"],
                    performance=0.0,
                )
        else:
            # Pure memory kernel
            model.add_point(
                name=name,
                arithmetic_intensity=0.0,
                performance=0.0,
            )
    
    # Print roofline summary
    print(f"\nGPU: {model.spec.name}")
    print(f"Peak Compute: {model.spec.peak_compute_tflops:.0f} TFLOP/s (FP16)")
    print(f"Peak Bandwidth: {model.spec.peak_bandwidth_gbps:.0f} GB/s")
    print(f"Ridge Point: {model.spec.ridge_point:.2f} FLOP/byte")
    
    print(f"\n{'Kernel':<25} {'AI (F/B)':<12} {'Perf (TF/s)':<14} {'Efficiency':<12} {'Bound':<12}")
    print("-" * 75)
    
    for point in model.get_points():
        bound = "Memory" if point.is_memory_bound else "Compute"
        print(f"{point.name:<25} {point.arithmetic_intensity:<12.4f} {point.performance:<14.4f} {point.efficiency:<12.1%} {bound:<12}")
    
    return model


def analyze_speed_of_light(results):
    """Perform Speed of Light (SOL) analysis."""
    print("\n" + "=" * 80)
    print("SPEED OF LIGHT (SOL) ANALYSIS")
    print("=" * 80)
    
    print(f"\nTarget GPU: {GPU_SPECS['name']}")
    print(f"Peak Memory BW: {GPU_SPECS['peak_bandwidth_gbps']:.0f} GB/s")
    print(f"Peak FP16 Compute: {GPU_SPECS['peak_tflops_fp16']:.0f} TFLOP/s")
    
    print(f"\n{'Kernel':<25} {'BW SOL %':<12} {'Compute SOL %':<15} {'Limiting Factor':<20}")
    print("-" * 75)
    
    sol_results = {}
    
    for name, data in results["components"].items():
        # Bandwidth SOL
        achieved_bw = data["memory"]["achieved_bandwidth_gbps"]
        bw_sol = achieved_bw / GPU_SPECS["peak_bandwidth_gbps"] * 100
        
        # Compute SOL  
        achieved_tflops = data["compute"]["achieved_tflops"]
        compute_sol = achieved_tflops / GPU_SPECS["peak_tflops_fp16"] * 100 if achieved_tflops > 0 else 0
        
        # Determine limiting factor
        if bw_sol < 1 and compute_sol < 1:
            if data["timing"]["mean_us"] < 10:
                limiting = "KERNEL LAUNCH OVERHEAD"
            else:
                limiting = "UNKNOWN (low utilization)"
        elif bw_sol > compute_sol:
            limiting = "Compute-limited"
        else:
            limiting = "Memory-limited"
        
        sol_results[name] = {
            "bw_sol": bw_sol,
            "compute_sol": compute_sol,
            "limiting_factor": limiting,
        }
        
        print(f"{name:<25} {bw_sol:<12.3f} {compute_sol:<15.3f} {limiting:<20}")
    
    return sol_results


def identify_bottlenecks(results, sol_results):
    """Identify bottlenecks and generate optimization recommendations."""
    print("\n" + "=" * 80)
    print("BOTTLENECK IDENTIFICATION")
    print("=" * 80)
    
    total_latency = results["full_pipeline"]["timing"]["mean_us"]
    
    bottlenecks = []
    for name, data in results["components"].items():
        latency = data["timing"]["mean_us"]
        pct = latency / total_latency * 100
        sol = sol_results.get(name, {})
        
        bottleneck_info = {
            "name": name,
            "latency_us": latency,
            "pct_of_total": pct,
            "bw_sol": sol.get("bw_sol", 0),
            "compute_sol": sol.get("compute_sol", 0),
            "limiting_factor": sol.get("limiting_factor", "unknown"),
            "ai": data["compute"]["arithmetic_intensity"],
        }
        bottlenecks.append(bottleneck_info)
    
    # Sort by percentage of total
    bottlenecks.sort(key=lambda x: x["pct_of_total"], reverse=True)
    
    print(f"\nRanked by impact on total latency ({total_latency:.2f} us):\n")
    
    for i, b in enumerate(bottlenecks, 1):
        print(f"{i}. {b['name']} ({b['latency_us']:.2f} us, {b['pct_of_total']:.1f}% of total)")
        print(f"   Limiting Factor: {b['limiting_factor']}")
        print(f"   BW SOL: {b['bw_sol']:.3f}%, Compute SOL: {b['compute_sol']:.3f}%")
        print(f"   Arithmetic Intensity: {b['ai']:.4f} FLOP/byte")
        print()
    
    return bottlenecks


def generate_optimization_recommendations(bottlenecks, results):
    """Generate detailed optimization recommendations."""
    print("\n" + "=" * 80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = []
    
    for b in bottlenecks:
        name = b["name"]
        limiting = b["limiting_factor"]
        bw_sol = b["bw_sol"]
        pct = b["pct_of_total"]
        
        print(f"\n{'='*60}")
        print(f"KERNEL: {name} ({pct:.1f}% of total latency)")
        print(f"{'='*60}")
        
        recs = []
        
        if "LAUNCH OVERHEAD" in limiting:
            print("\n  [DIAGNOSIS] Kernel launch overhead dominates execution time")
            print("  This kernel has very little work but pays fixed launch cost.")
            recs.append(("[CRITICAL] FUSE WITH ADJACENT KERNELS", 
                        "Combine with preceding or following kernels to amortize launch overhead"))
            recs.append(("[HIGH] USE PERSISTENT KERNEL", 
                        "Keep thread blocks alive across multiple invocations"))
            recs.append(("[MED] BATCH OPERATIONS", 
                        "Process multiple batches in single kernel launch"))
            
        elif "Memory-limited" in limiting or bw_sol < 1:
            print(f"\n  [DIAGNOSIS] Memory-bound with only {bw_sol:.3f}% bandwidth utilization")
            print("  Significant room for improvement in memory access patterns.")
            
            if bw_sol < 0.1:
                recs.append(("[CRITICAL] INVESTIGATE LOW BW UTILIZATION", 
                            "BW utilization < 0.1% suggests severe memory inefficiency"))
                recs.append(("[HIGH] CHECK MEMORY COALESCING", 
                            "Ensure threads access contiguous memory addresses"))
            
            recs.append(("[HIGH] VECTORIZE MEMORY ACCESS", 
                        "Use float4/int4 for 4x fewer memory transactions"))
            recs.append(("[HIGH] IMPROVE DATA LOCALITY", 
                        "Use shared memory (LDS) for frequently accessed data"))
            recs.append(("[MED] PREFETCH DATA", 
                        "Hide memory latency with software prefetching"))
            
        elif "Compute-limited" in limiting:
            print(f"\n  [DIAGNOSIS] Compute-bound with {b['compute_sol']:.3f}% compute utilization")
            recs.append(("[HIGH] USE TENSOR CORES (MFMA)", 
                        "Matrix operations should use MFMA instructions"))
            recs.append(("[MED] REDUCE INSTRUCTION COUNT", 
                        "Eliminate redundant computations"))
            recs.append(("[MED] INCREASE ILP", 
                        "Unroll loops to expose instruction-level parallelism"))
            
        else:
            print(f"\n  [DIAGNOSIS] Unknown bottleneck - requires deeper profiling")
            recs.append(("[HIGH] USE HARDWARE COUNTERS", 
                        "Profile with rocprofv3 to get detailed performance counters"))
        
        # Kernel-specific recommendations
        if "triton" in name.lower():
            recs.append(("[INFO] TRITON-SPECIFIC", 
                        "Consider reducing constexpr parameters or tuning block sizes"))
        elif "hip" in name.lower():
            recs.append(("[INFO] HIP-SPECIFIC", 
                        "Check for warp divergence and register pressure"))
        elif "ck" in name.lower():
            recs.append(("[INFO] CK-SPECIFIC", 
                        "This is AMD-optimized code; limited optimization potential"))
        
        print("\n  RECOMMENDATIONS:")
        for priority, rec in recs:
            print(f"    {priority}: {rec}")
            recommendations.append({"kernel": name, "priority": priority, "recommendation": rec})
    
    return recommendations


def generate_optimization_plan(bottlenecks, recommendations):
    """Generate a concrete optimization plan based on analysis."""
    print("\n" + "=" * 80)
    print("CONCRETE OPTIMIZATION PLAN")
    print("=" * 80)
    
    # Sort bottlenecks by impact
    top_bottleneck = bottlenecks[0]
    
    print(f"\nPrimary Target: {top_bottleneck['name']} ({top_bottleneck['pct_of_total']:.1f}% of latency)")
    print(f"Limiting Factor: {top_bottleneck['limiting_factor']}")
    
    print("\n" + "-" * 60)
    print("RECOMMENDED OPTIMIZATION STRATEGY")
    print("-" * 60)
    
    if "triton_append" in top_bottleneck["name"] and "LAUNCH OVERHEAD" in top_bottleneck["limiting_factor"]:
        print("""
STRATEGY 1: FUSE TRITON APPEND WITH HIP TOPK
----------------------------------------------
The Triton append kernel has minimal work but high launch overhead.

Approach:
1. Modify the HIP biased_grouped_topk kernel to output directly in
   the format required by CK sorting (with shared expert appended)
2. This eliminates the Triton kernel entirely
3. Expected improvement: ~60% latency reduction for the pipeline

Implementation:
- Modify aiter/csrc/kernels/topk_softmax_kernels_group.cu
- Add shared expert ID and weight at position K after each row
- Output to ids_out, w_out directly instead of topk_ids, topk_w

STRATEGY 2: FUSE CK SORTING WITH APPEND
----------------------------------------------
Alternative: Integrate append logic into first phase of CK sorting.

Approach:
1. Modify MoeSortingMultiPhaseKernel_P0_v2 to handle input without
   shared expert appended
2. Append shared expert as part of P0 initialization
3. Expected improvement: ~50% latency reduction

STRATEGY 3: USE HIP GRAPHS
----------------------------------------------
If modification of existing kernels is not feasible:

Approach:
1. Capture the entire pipeline as a HIP Graph
2. Replay graph instead of launching individual kernels
3. Reduces launch overhead by ~80%
4. Expected improvement: ~40% latency reduction

RECOMMENDED ORDER:
1. First try Strategy 1 (highest impact, single kernel modification)
2. If not feasible, try Strategy 3 (least invasive)
3. Strategy 2 as last resort (modifying CK code is complex)
""")
    else:
        print("""
GENERAL OPTIMIZATION APPROACH:
1. Profile with hardware counters (rocprofv3 with PMC counters)
2. Identify specific bottleneck (cache misses, occupancy, etc.)
3. Apply targeted optimization based on findings
4. Measure and iterate
""")
    
    return top_bottleneck


def main():
    # Run profiling
    results = run_comprehensive_profiling()
    if results is None:
        return
    
    # Roofline analysis
    roofline_model = analyze_roofline(results)
    
    # Speed of Light analysis
    sol_results = analyze_speed_of_light(results)
    
    # Bottleneck identification
    bottlenecks = identify_bottlenecks(results, sol_results)
    
    # Generate recommendations
    recommendations = generate_optimization_recommendations(bottlenecks, results)
    
    # Generate optimization plan
    optimization_target = generate_optimization_plan(bottlenecks, recommendations)
    
    # Save full analysis
    analysis_output = {
        "profiling_results": results,
        "sol_analysis": sol_results,
        "bottlenecks": bottlenecks,
        "recommendations": [{"kernel": r["kernel"], "priority": r["priority"], 
                           "recommendation": r["recommendation"]} for r in recommendations],
        "primary_target": optimization_target,
    }
    
    output_path = WORKSPACE / "full_analysis.json"
    with open(output_path, "w") as f:
        json.dump(analysis_output, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print(f"Full analysis saved to: {output_path}")
    print(f"{'='*80}")
    
    return analysis_output


if __name__ == "__main__":
    analysis = main()

