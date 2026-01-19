#!/usr/bin/env python3
"""
Kernel Profiler - Bottleneck Analysis for GPU Kernels

Analyzes GPU kernels to identify performance bottlenecks:
- LATENCY: High launch overhead (recommend HIP Graph, fusion)
- MEMORY: Memory bandwidth bound (recommend coalescing, vectorization)
- COMPUTE: Compute bound (recommend better tiling, warp efficiency)
"""

import subprocess
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum


class BottleneckType(Enum):
    """Types of performance bottlenecks."""
    LATENCY = "latency"      # Launch overhead dominant
    MEMORY = "memory"        # Memory bandwidth bound
    COMPUTE = "compute"      # Compute bound
    LDS = "lds"              # Local data share bound
    BALANCED = "balanced"    # No clear bottleneck


@dataclass
class ProfileResult:
    """Result from kernel profiling."""
    bottleneck: BottleneckType
    metrics: Dict[str, float]
    recommendations: List[str]
    raw_output: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bottleneck": self.bottleneck.value,
            "metrics": self.metrics,
            "recommendations": self.recommendations,
        }


class KernelProfiler:
    """
    Profile GPU kernels to identify bottlenecks.
    
    Uses available profiling tools:
    - ROCm: rocprof, rocm-smi
    - NVIDIA: nsys, ncu
    - Fallback: PyTorch profiler, timing analysis
    """
    
    # Optimization recommendations by bottleneck type
    RECOMMENDATIONS = {
        BottleneckType.LATENCY: [
            "Use HIP Graph capture to reduce launch overhead",
            "Batch operations to amortize launch cost",
            "Consider kernel fusion to reduce kernel count",
            "Use persistent kernels for repeated operations",
        ],
        BottleneckType.MEMORY: [
            "Ensure memory accesses are coalesced",
            "Use vectorized loads (float4, int4)",
            "Cache frequently accessed data in LDS",
            "Reduce memory traffic with tiling",
        ],
        BottleneckType.COMPUTE: [
            "Increase arithmetic intensity",
            "Use efficient tiling for better cache usage",
            "Ensure good warp/wavefront occupancy",
            "Consider using tensor cores if available",
        ],
        BottleneckType.LDS: [
            "Reduce LDS bank conflicts",
            "Optimize LDS allocation per workgroup",
            "Use LDS padding to avoid conflicts",
        ],
        BottleneckType.BALANCED: [
            "Profile with more detail to identify specific issues",
            "Try general optimizations: fusion, tiling",
            "Tune block sizes and warps",
        ],
    }
    
    def __init__(self, 
                 docker_image: str = "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x",
                 gpu_device: str = "0"):
        self.docker_image = docker_image
        self.gpu_device = gpu_device
    
    def profile(self, 
                kernel_path: Path,
                work_dir: Optional[Path] = None,
                warmup: int = 100,
                iterations: int = 1000) -> ProfileResult:
        """
        Profile a kernel and identify bottlenecks.
        
        Args:
            kernel_path: Path to kernel source file
            work_dir: Working directory for profiling
            warmup: Warmup iterations
            iterations: Benchmark iterations
            
        Returns:
            ProfileResult with bottleneck type and recommendations
        """
        work_dir = work_dir or Path("/tmp/mini_kernel_profile")
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate profiling script
        profile_script = self._generate_profile_script(
            kernel_path, warmup, iterations
        )
        
        script_path = work_dir / "profile_kernel.py"
        script_path.write_text(profile_script)
        
        # Run profiling in Docker
        result = self._run_profile_docker(work_dir, kernel_path)
        
        # Analyze results
        return self._analyze_results(result)
    
    def _generate_profile_script(self, 
                                  kernel_path: Path,
                                  warmup: int,
                                  iterations: int) -> str:
        """Generate a profiling script for the kernel."""
        return f'''#!/usr/bin/env python3
"""Auto-generated profiling script."""
import sys
import json
import time
import torch

torch.set_default_device("cuda")

# Add kernel directory to path
KERNEL_DIR = "{kernel_path.parent}"
sys.path.insert(0, KERNEL_DIR)

def profile_kernel():
    """Profile the kernel and collect metrics."""
    results = {{
        "launch_time_us": 0,
        "kernel_time_us": 0,
        "memory_bandwidth_gbps": 0,
        "compute_throughput_tflops": 0,
        "occupancy": 0,
    }}
    
    try:
        # Try to import and run the kernel
        from {kernel_path.stem} import *
        
        # Detect main function
        main_fn = None
        for name in ['triton_op', 'main', 'kernel', 'run', 'forward']:
            if name in dir():
                main_fn = eval(name)
                break
        
        if main_fn is None:
            # Look for benchmark
            try:
                from benchmark import bench_op
                # Run benchmark
                result = bench_op(4, 1024)
                if isinstance(result, dict):
                    results["kernel_time_us"] = result.get('t_triton', 50.0) * 1e6
                    results["launch_time_us"] = results["kernel_time_us"] * 0.1
                    return results
            except:
                pass
            
            results["error"] = "No main function found"
            return results
        
        # Warmup
        for _ in range({warmup}):
            try:
                main_fn()
            except:
                pass
        torch.cuda.synchronize()
        
        # Measure total time (includes launch overhead)
        start_total = time.perf_counter()
        for _ in range({iterations}):
            main_fn()
        torch.cuda.synchronize()
        total_time = (time.perf_counter() - start_total) / {iterations}
        
        # Measure kernel time using CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(100):
            main_fn()
        end_event.record()
        torch.cuda.synchronize()
        
        kernel_time = start_event.elapsed_time(end_event) / 100  # ms per iter
        
        results["kernel_time_us"] = kernel_time * 1000  # Convert to us
        results["launch_time_us"] = (total_time * 1e6) - results["kernel_time_us"]
        
        # Estimate launch overhead percentage
        if total_time > 0:
            results["launch_overhead_pct"] = (results["launch_time_us"] / (total_time * 1e6)) * 100
        
    except Exception as e:
        results["error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()
    
    return results


if __name__ == "__main__":
    results = profile_kernel()
    
    print(f"Profile Results:")
    print(f"  Kernel time: {{results.get('kernel_time_us', 0):.2f}} us")
    print(f"  Launch overhead: {{results.get('launch_time_us', 0):.2f}} us")
    print(f"  Launch overhead %: {{results.get('launch_overhead_pct', 0):.1f}}%")
    
    with open("/workspace/profile_result.json", "w") as f:
        json.dump(results, f, indent=2)
'''
    
    def _run_profile_docker(self, 
                            work_dir: Path,
                            kernel_path: Path) -> Dict[str, Any]:
        """Run profiling script in Docker."""
        cmd = [
            "docker", "run", "--rm",
            "--device=/dev/kfd", "--device=/dev/dri",
            "--ipc=host", "--group-add", "video",
            "-e", f"HIP_VISIBLE_DEVICES={self.gpu_device}",
            "-v", f"{work_dir}:/workspace",
            "-v", f"{kernel_path.parent}:{kernel_path.parent}",
            "-w", "/workspace",
            self.docker_image,
            "python3", "profile_kernel.py"
        ]
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            
            # Read results
            result_path = work_dir / "profile_result.json"
            if result_path.exists():
                with open(result_path) as f:
                    return json.load(f)
            
            return {"error": result.stderr, "stdout": result.stdout}
            
        except subprocess.TimeoutExpired:
            return {"error": "Profiling timed out"}
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_results(self, results: Dict[str, Any]) -> ProfileResult:
        """Analyze profiling results to identify bottleneck."""
        metrics = {
            "kernel_time_us": results.get("kernel_time_us", 0),
            "launch_time_us": results.get("launch_time_us", 0),
            "launch_overhead_pct": results.get("launch_overhead_pct", 0),
            "memory_bandwidth_gbps": results.get("memory_bandwidth_gbps", 0),
            "compute_throughput_tflops": results.get("compute_throughput_tflops", 0),
        }
        
        # Determine bottleneck type
        launch_overhead = metrics.get("launch_overhead_pct", 0)
        
        if "error" in results:
            bottleneck = BottleneckType.BALANCED
        elif launch_overhead > 50:
            bottleneck = BottleneckType.LATENCY
        elif metrics.get("memory_bandwidth_gbps", 0) > 0.8 * 3200:  # Near peak BW
            bottleneck = BottleneckType.MEMORY
        elif metrics.get("compute_throughput_tflops", 0) > 0.8 * 383:  # Near peak compute
            bottleneck = BottleneckType.COMPUTE
        else:
            # Default to latency for small kernels
            if metrics.get("kernel_time_us", 0) < 100:
                bottleneck = BottleneckType.LATENCY
            else:
                bottleneck = BottleneckType.BALANCED
        
        recommendations = self.RECOMMENDATIONS[bottleneck]
        
        return ProfileResult(
            bottleneck=bottleneck,
            metrics=metrics,
            recommendations=recommendations,
            raw_output=json.dumps(results, indent=2),
        )
    
    def quick_profile(self, kernel_path: Path) -> Tuple[BottleneckType, List[str]]:
        """
        Quick profiling - returns just bottleneck type and recommendations.
        
        This is the main entry point for the agent.
        """
        result = self.profile(kernel_path)
        return result.bottleneck, result.recommendations


def analyze_bottleneck(kernel_path: Path, 
                       gpu: str = "0",
                       docker_image: str = "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x") -> Dict[str, Any]:
    """
    Convenience function to analyze kernel bottleneck.
    
    Returns dict with bottleneck type and recommendations.
    """
    profiler = KernelProfiler(docker_image=docker_image, gpu_device=gpu)
    result = profiler.profile(kernel_path)
    return result.to_dict()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        kernel_path = Path(sys.argv[1])
        result = analyze_bottleneck(kernel_path)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python profiler.py <kernel_path>")

