"""
bench-mcp: Reliable Performance Measurement

Role: Produce statistically meaningful performance results.
- Warmup and repeated sampling
- Statistical analysis (mean, min, max, std, percentiles)
- Derived metrics (TFLOPS, bandwidth, speedup vs baseline)

Used for: baseline creation, checkpoint comparison, rollback decisions.
"""

import subprocess
import tempfile
import json
import statistics
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class BenchmarkResult:
    """Result from benchmarking."""
    mean_latency_us: float
    min_latency_us: float
    max_latency_us: float
    std_latency_us: float
    p50_latency_us: float
    p95_latency_us: float
    p99_latency_us: float
    iterations: int
    warmup_iterations: int
    tflops: Optional[float] = None
    bandwidth_gbps: Optional[float] = None
    speedup_vs_baseline: Optional[float] = None


class BenchTool:
    """
    MCP Tool for reliable performance measurement.
    
    Provides statistically rigorous benchmarking with
    proper warmup and statistical analysis.
    """
    
    # Tool metadata for MCP
    TOOL_NAME = "bench"
    TOOL_DESCRIPTION = """Benchmark a GPU kernel with statistical rigor.
    
Returns:
- Latency statistics (mean, min, max, std, percentiles)
- Derived metrics (TFLOPS, bandwidth if applicable)
- Speedup vs baseline (if baseline provided)"""
    
    TOOL_SCHEMA = {
        "type": "object",
        "properties": {
            "kernel_code": {
                "type": "string",
                "description": "The kernel code to benchmark"
            },
            "warmup_iters": {
                "type": "integer",
                "description": "Number of warmup iterations",
                "default": 1000
            },
            "bench_iters": {
                "type": "integer",
                "description": "Number of benchmark iterations",
                "default": 3000
            },
            "baseline_latency_us": {
                "type": "number",
                "description": "Baseline latency for speedup calculation"
            },
            "flops": {
                "type": "integer",
                "description": "Total FLOPs for TFLOPS calculation"
            },
            "bytes": {
                "type": "integer",
                "description": "Total bytes transferred for bandwidth calculation"
            }
        },
        "required": ["kernel_code"]
    }
    
    def __init__(self, docker_image: str = None, gpu_device: str = "3"):
        self.docker_image = docker_image or "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x"
        self.gpu_device = gpu_device
        self.work_dir = Path(tempfile.mkdtemp(prefix="bench_"))
    
    def benchmark(self, kernel_code: str,
                  warmup_iters: int = 1000,
                  bench_iters: int = 3000,
                  baseline_latency_us: Optional[float] = None,
                  flops: Optional[int] = None,
                  bytes_transferred: Optional[int] = None) -> BenchmarkResult:
        """
        Benchmark a kernel with statistical analysis.
        
        This is the main entry point for the MCP tool.
        """
        # Generate benchmark script
        script = self._generate_bench_script(
            kernel_code, warmup_iters, bench_iters
        )
        
        script_path = self.work_dir / "bench_kernel.py"
        script_path.write_text(script)
        
        # Run benchmark
        result = self._run_benchmark(script_path)
        
        # Calculate derived metrics
        mean_latency = result.get("mean_us", 0)
        
        tflops = None
        if flops and mean_latency > 0:
            tflops = flops / (mean_latency * 1e-6) / 1e12
        
        bandwidth = None
        if bytes_transferred and mean_latency > 0:
            bandwidth = bytes_transferred / (mean_latency * 1e-6) / 1e9
        
        speedup = None
        if baseline_latency_us and mean_latency > 0:
            speedup = baseline_latency_us / mean_latency
        
        return BenchmarkResult(
            mean_latency_us=result.get("mean_us", 0),
            min_latency_us=result.get("min_us", 0),
            max_latency_us=result.get("max_us", 0),
            std_latency_us=result.get("std_us", 0),
            p50_latency_us=result.get("p50_us", 0),
            p95_latency_us=result.get("p95_us", 0),
            p99_latency_us=result.get("p99_us", 0),
            iterations=bench_iters,
            warmup_iterations=warmup_iters,
            tflops=tflops,
            bandwidth_gbps=bandwidth,
            speedup_vs_baseline=speedup,
        )
    
    def _generate_bench_script(self, kernel_code: str,
                               warmup_iters: int,
                               bench_iters: int) -> str:
        """Generate the benchmark script."""
        return f'''#!/usr/bin/env python3
"""Auto-generated benchmark script"""
import torch
import json
import statistics

torch.manual_seed(42)
device = "cuda"

# Kernel code
{kernel_code}

# Find the main function to benchmark
main_fn = None
for name in dir():
    obj = eval(name)
    if callable(obj) and name.startswith("run_"):
        main_fn = obj
        break

if main_fn is None:
    print("ERROR: No run_* function found")
    exit(1)

# Warmup
print(f"Warming up ({warmup_iters} iters)...")
for _ in range({warmup_iters}):
    main_fn()
torch.cuda.synchronize()

# Benchmark with individual timings for statistics
print(f"Benchmarking ({bench_iters} iters)...")
times = []

# First, get aggregate timing
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range({bench_iters}):
    main_fn()
end.record()
torch.cuda.synchronize()

mean_us = start.elapsed_time(end) * 1000 / {bench_iters}

# Then get individual timings for statistics (subset)
sample_size = min(500, {bench_iters})
for _ in range(sample_size):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    main_fn()
    e.record()
    torch.cuda.synchronize()
    times.append(s.elapsed_time(e) * 1000)  # Convert to us

# Calculate statistics
times_sorted = sorted(times)
results = {{
    "mean_us": mean_us,
    "min_us": min(times),
    "max_us": max(times),
    "std_us": statistics.stdev(times) if len(times) > 1 else 0,
    "p50_us": times_sorted[len(times_sorted) // 2],
    "p95_us": times_sorted[int(len(times_sorted) * 0.95)],
    "p99_us": times_sorted[int(len(times_sorted) * 0.99)],
    "sample_size": sample_size,
    "bench_iters": {bench_iters},
}}

print(f"Mean: {{mean_us:.2f}} us")
print(f"Min: {{results['min_us']:.2f}} us")
print(f"Max: {{results['max_us']:.2f}} us")
print(f"Std: {{results['std_us']:.2f}} us")

with open("/workspace/bench_result.json", "w") as f:
    json.dump(results, f, indent=2)
'''
    
    def _run_benchmark(self, script_path: Path) -> Dict[str, Any]:
        """Run benchmark in Docker."""
        cmd = [
            "docker", "run", "--rm",
            "--device=/dev/kfd", "--device=/dev/dri",
            "--ipc=host", "--group-add", "video",
            "-e", f"HIP_VISIBLE_DEVICES={self.gpu_device}",
            "-v", f"{self.work_dir}:/workspace",
            self.docker_image,
            "python3", f"/workspace/{script_path.name}"
        ]
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
            
            # Load results
            result_path = self.work_dir / "bench_result.json"
            if result_path.exists():
                with open(result_path) as f:
                    return json.load(f)
            
            return {"error": result.stderr}
            
        except subprocess.TimeoutExpired:
            return {"error": "timeout"}
        except Exception as e:
            return {"error": str(e)}
    
    # MCP Tool interface methods
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return MCP tool definition."""
        return {
            "name": self.TOOL_NAME,
            "description": self.TOOL_DESCRIPTION,
            "inputSchema": self.TOOL_SCHEMA,
        }
    
    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given arguments."""
        result = self.benchmark(
            kernel_code=arguments["kernel_code"],
            warmup_iters=arguments.get("warmup_iters", 1000),
            bench_iters=arguments.get("bench_iters", 3000),
            baseline_latency_us=arguments.get("baseline_latency_us"),
            flops=arguments.get("flops"),
            bytes_transferred=arguments.get("bytes"),
        )
        
        return {
            "mean_latency_us": result.mean_latency_us,
            "min_latency_us": result.min_latency_us,
            "max_latency_us": result.max_latency_us,
            "std_latency_us": result.std_latency_us,
            "p50_latency_us": result.p50_latency_us,
            "p95_latency_us": result.p95_latency_us,
            "p99_latency_us": result.p99_latency_us,
            "tflops": result.tflops,
            "bandwidth_gbps": result.bandwidth_gbps,
            "speedup_vs_baseline": result.speedup_vs_baseline,
        }


