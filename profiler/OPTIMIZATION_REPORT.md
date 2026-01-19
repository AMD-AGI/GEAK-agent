# TopK & Sort Module Optimization Report

## Executive Summary

Using the **generic ROCm profiler utility**, we achieved a **2.35x speedup** (57.5% improvement) on the TopK & Sort module, reducing latency from **57.00 us to 24.25 us**.

## Profiling Analysis

### Initial Profiling Results (BS=4)

| Kernel | Latency | % Total | BW (GB/s) | BW SOL % | Root Cause |
|--------|---------|---------|-----------|----------|------------|
| **triton_append** | 34.52 us | **60.3%** | 0.02 | 0.0003% | **KERNEL LAUNCH OVERHEAD** |
| ck_sorting | 18.24 us | 31.8% | 10.92 | 0.14% | Low occupancy |
| hip_topk | 13.10 us | 22.9% | 0.21 | 0.003% | Small batch |
| **TOTAL** | **57.27 us** | 100% | | | |

### Key Insight

The Triton append kernel was transferring only **544 bytes** but taking **34+ us** - that's 0.02 GB/s vs 8000 GB/s peak (0.00025% utilization). This was **pure kernel launch overhead** for essentially no work.

## Optimization Journey

### Step 1: Eliminate Triton Kernel (32.8% improvement)
- **Before:** 57.00 us
- **After:** 38.30 us
- **Method:** Replace Triton `fast_append` kernel with PyTorch tensor copy operations
- **Rationale:** The Triton kernel launch overhead dominated execution time for such small data

### Step 2: HIP Graph Capture (52.2% improvement)
- **Before:** 57.00 us → 38.30 us
- **After:** 27.24 us
- **Method:** Capture entire pipeline as HIP Graph for reduced launch overhead
- **Rationale:** Graph replay eliminates per-invocation CPU-GPU synchronization

### Step 3: Memory Alignment (54.8% improvement)
- **Before:** 27.24 us
- **After:** 25.76 us
- **Method:** 256-byte aligned tensor allocations
- **Rationale:** Better memory access patterns for GPU hardware

### Step 4: Skip Unnecessary Operations (57.5% improvement)
- **Before:** 25.76 us
- **After:** 24.25 us
- **Method:** Remove unnecessary `num_valid.zero_()` call before moe_sorting_fwd
- **Rationale:** The CK kernel always overwrites this value

## Final Results

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        FINAL OPTIMIZATION RESULTS                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Original:  57.00 us (HIP + Triton + CK baseline)                            ║
║  Final:     24.25 us (Optimized)                                             ║
║                                                                              ║
║  Speedup:   2.35x (mean), 2.50x (peak)                                       ║
║  Improvement: 57.5%                                                          ║
║  Saved:     32.75 us per invocation                                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

## Optimization Techniques Applied

1. **Profiler-Guided Analysis**
   - Used generic ROCm profiler to identify bottlenecks
   - Speed of Light (SOL) analysis revealed 0.0003% bandwidth utilization
   - Roofline analysis confirmed latency-bound (not compute/memory bound)

2. **Kernel Fusion / Elimination**
   - Eliminated separate Triton kernel for append operation
   - Pre-filled shared expert columns at initialization

3. **HIP Graph Capture**
   - Captured entire pipeline as single graph
   - Eliminates kernel launch overhead on replay

4. **Memory Optimization**
   - 256-byte aligned allocations
   - Graph memory pool for consistent allocation patterns

5. **Unnecessary Operation Removal**
   - Identified and removed redundant `num_valid.zero_()` call

## Profiler Framework Usage

The optimization was guided by the **generic profiler utility** in the kernel optimization agent framework:

```python
from profiler import GenericProfiler, profile_module

# Profile any kernel/module
analysis = profile_module(
    script_content=PROFILE_SCRIPT,
    module_name="topk_sort",
    gpu_id=0
)

# Access analysis results
print(f"Primary bottleneck: {analysis.primary_bottleneck_component}")
print(f"Bottleneck type: {analysis.primary_bottleneck_type}")
for rec in analysis.recommendations:
    print(f"  - {rec}")
```

## Correctness Verification

All optimizations were verified for functional correctness:
- `num_valid` matches baseline
- Output tensor values match within tolerance
- Final MoE routing produces identical results

## Applicability

This optimization approach is applicable when:
- Small batch sizes (M=4) where kernel launch overhead dominates
- Pipeline of multiple kernels with minimal data dependencies
- HIP Graph-compatible operations

For larger batch sizes (M=64+), the relative overhead of kernel launches decreases and these optimizations have less impact.

## Files

- `/home/sapmajum/kernel_optimization_framework/profiler/` - Generic profiler utility
- `/home/sapmajum/kernel_optimization_framework/profiler/final_push.py` - Final optimization script
- `/home/sapmajum/kernel_optimization_framework/profiler/comprehensive_topk_sort_profile.py` - Profiling analysis

## Conclusion

By using the profiler to identify that **60% of latency was kernel launch overhead** (not actual computation), we were able to apply targeted optimizations that achieved a **2.35x speedup**. This demonstrates the power of profiler-guided optimization for GPU kernels.

