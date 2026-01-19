# ROCm Kernel Profiler Utility

A generic profiler utility for the kernel optimization agent framework. This utility can profile any GPU kernel (Triton, HIP, CK, PyTorch) and provide bottleneck analysis with optimization recommendations.

## Overview

This profiler is designed to be **reusable across any module** without modification. It:

1. **Profiles kernel components** - Measures latency, bandwidth, and compute metrics
2. **Analyzes bottlenecks** - Identifies if kernels are compute-bound, memory-bound, or latency-bound
3. **Generates recommendations** - Provides prioritized optimization suggestions based on the roofline model
4. **Integrates with OpenEvolve** - Can be used as a fitness function for evolutionary optimization

## Quick Start

```python
from profiler.generic_profiler import GenericProfiler, profile_module

# Option 1: Use the profiler class directly
profiler = GenericProfiler(gpu_id=0)
results = profiler.profile_script(script_content)
analysis = profiler.analyze_results(results, module_name="my_module")
profiler.print_analysis(analysis)

# Option 2: Use the convenience function
analysis = profile_module(
    script_content=MY_PROFILE_SCRIPT,
    module_name="my_module",
    gpu_id=0,
    print_results=True
)
```

## Creating a Profile Script

Your profile script should output a JSON file with the following structure:

```python
results = {
    "config": {
        # Your module configuration
        "param1": value1,
        "param2": value2,
    },
    "components": {
        "kernel1_name": {
            "type": "triton",  # or "hip", "ck", "torch"
            "mean_us": 10.5,
            "std_us": 0.5,
            "min_us": 9.0,
            "max_us": 12.0,
            "p50_us": 10.3,
            "p95_us": 11.5,
            "p99_us": 11.9,
            "bytes_transferred": 1024000,  # Total memory IO in bytes
            "flops_estimated": 500000,     # Estimated FLOPs
        },
        "kernel2_name": {
            # Similar structure
        },
    },
    "full_pipeline": {
        "mean_us": 25.0,
        "std_us": 1.0,
        # ... other timing fields
    },
}

with open("/workspace/results.json", "w") as f:
    json.dump(results, f, indent=2)
```

## Components

### `generic_profiler.py`

Main profiler class with:
- `GenericProfiler` - Runs scripts in Docker and collects results
- `profile_module()` - Convenience function for quick profiling
- `KernelMetrics` - Data class for kernel metrics
- `ModuleAnalysis` - Data class for analysis results

### `bottleneck_analyzer.py`

Detailed bottleneck analysis:
- Identifies primary bottleneck type (compute, memory, latency, LDS, cache)
- Calculates utilization metrics
- Generates prioritized optimization list

### `roofline.py`

Roofline model implementation:
- Calculates arithmetic intensity
- Determines if kernel is compute-bound or memory-bound
- Provides ridge point for the target GPU

### `openevolve_integration.py`

Integration with OpenEvolve for evolutionary optimization:
- `ProfilingFitness` - Use profiler metrics as fitness function
- Supports multi-objective optimization (latency + correctness)

## GPU Support

Currently supports:
- **gfx942** (MI300X) - 1307 TFLOPS FP16, 5300 GB/s bandwidth
- **gfx950** (MI355X) - 1600 TFLOPS FP16, 8000 GB/s bandwidth

## Example: Profiling a New Module

```python
from profiler.generic_profiler import GenericProfiler

# Your profiling script that runs in Docker
MY_MODULE_SCRIPT = '''
import torch
import numpy as np
import json

# ... your kernel imports ...

# Warmup
for _ in range(1000):
    run_kernel()
torch.cuda.synchronize()

# Profile
times = []
for _ in range(500):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    run_kernel()
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) * 1000)  # us

results = {
    "components": {
        "my_kernel": {
            "type": "triton",
            "mean_us": np.mean(times),
            "std_us": np.std(times),
            "bytes_transferred": TOTAL_BYTES,
            "flops_estimated": TOTAL_FLOPS,
        }
    },
    "full_pipeline": {
        "mean_us": np.mean(times),
        "std_us": np.std(times),
    }
}

with open("/workspace/results.json", "w") as f:
    json.dump(results, f, indent=2)
'''

# Run profiling
profiler = GenericProfiler(gpu_id=0)
results = profiler.profile_script(MY_MODULE_SCRIPT)
analysis = profiler.analyze_results(results, "my_module")
profiler.print_analysis(analysis)
```

## Bottleneck Types

| Type | Description | Key Optimizations |
|------|-------------|-------------------|
| `MEMORY` | Bandwidth-limited | Vectorization, coalescing, fusion |
| `COMPUTE` | ALU-limited | Tensor cores, algorithm optimization |
| `LATENCY` | Launch overhead | Kernel fusion, persistent kernels |
| `LDS` | Shared memory conflicts | Padding, data layout |
| `CACHE` | Cache thrashing | Tiling, blocking, prefetching |

## Integration with Agent Framework

The profiler integrates with the kernel optimization agent framework:

```python
from profiler.openevolve_integration import ProfilingFitness

# Create fitness function for OpenEvolve
fitness = ProfilingFitness(
    profiler=GenericProfiler(gpu_id=0),
    profile_script_template=TEMPLATE,
    target_latency_us=20.0,
    correctness_weight=0.5,
)

# Use in OpenEvolve optimization
score = fitness.evaluate(candidate_code)
```

## Notes

- The profiler runs scripts in Docker to ensure consistent ROCm environment
- Heavy warmup (1000+ iterations) is recommended to eliminate JIT compilation overhead
- For Triton kernels, ensure the kernel is defined at module level for proper JIT caching
- Memory and FLOPS estimates are used for bottleneck analysis; more accurate with actual data sizes
