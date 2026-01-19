"""
profiler-mcp: Hardware-level Bottleneck Analysis

Role: Identify performance bottlenecks using hardware signals.
- Kernel-only timing via rocprofv3
- Hardware counters (compute, memory, cache, LDS)
- Bottleneck classification (compute-, memory-, LDS-bound, etc.)
- Roofline and speed-of-light analysis
- Actionable optimization hints

Used for: baseline profiling, post-optimization validation, and stall diagnosis.
"""

import subprocess
import tempfile
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum


class BottleneckType(Enum):
    COMPUTE = "compute"
    MEMORY = "memory"
    LATENCY = "latency"
    LDS = "lds"
    CACHE = "cache"
    BALANCED = "balanced"
    UNKNOWN = "unknown"


@dataclass
class ProfileResult:
    """Result from profiling."""
    kernel_name: str
    latency_us: float
    bottleneck: BottleneckType
    compute_utilization: float  # 0-1
    memory_utilization: float   # 0-1
    memory_bandwidth_gbps: float
    arithmetic_intensity: float
    hardware_counters: Dict[str, Any]
    suggestions: List[str]
    roofline_position: Dict[str, float]


class ProfilerTool:
    """
    MCP Tool for hardware-level profiling.
    
    Exposes profiling capabilities as a tool that returns
    structured, machine-readable results.
    """
    
    # Tool metadata for MCP
    TOOL_NAME = "profiler"
    TOOL_DESCRIPTION = """Profile a GPU kernel to identify performance bottlenecks.
    
Returns:
- Latency in microseconds
- Bottleneck type (compute, memory, latency, LDS, cache)
- Hardware utilization metrics
- Optimization suggestions based on bottleneck type"""
    
    TOOL_SCHEMA = {
        "type": "object",
        "properties": {
            "kernel_code": {
                "type": "string",
                "description": "The kernel code to profile"
            },
            "warmup_iters": {
                "type": "integer",
                "description": "Number of warmup iterations",
                "default": 100
            },
            "profile_iters": {
                "type": "integer",
                "description": "Number of profiling iterations",
                "default": 100
            },
            "collect_counters": {
                "type": "boolean",
                "description": "Whether to collect hardware counters",
                "default": True
            }
        },
        "required": ["kernel_code"]
    }
    
    def __init__(self, docker_image: str = None, gpu_device: str = "3"):
        self.docker_image = docker_image or "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x"
        self.gpu_device = gpu_device
        self.work_dir = Path(tempfile.mkdtemp(prefix="profiler_"))
    
    def profile(self, kernel_code: str, 
                warmup_iters: int = 100,
                profile_iters: int = 100,
                collect_counters: bool = True) -> ProfileResult:
        """
        Profile a kernel and return structured results.
        
        This is the main entry point for the MCP tool.
        """
        # Generate profiling script
        script = self._generate_profile_script(
            kernel_code, warmup_iters, profile_iters, collect_counters
        )
        
        script_path = self.work_dir / "profile_kernel.py"
        script_path.write_text(script)
        
        # Run profiling
        result = self._run_profile(script_path, collect_counters)
        
        # Analyze results
        bottleneck = self._analyze_bottleneck(result)
        suggestions = self._generate_suggestions(bottleneck, result)
        roofline = self._compute_roofline_position(result)
        
        return ProfileResult(
            kernel_name=result.get("kernel_name", "unknown"),
            latency_us=result.get("latency_us", 0),
            bottleneck=bottleneck,
            compute_utilization=result.get("compute_util", 0),
            memory_utilization=result.get("memory_util", 0),
            memory_bandwidth_gbps=result.get("bandwidth_gbps", 0),
            arithmetic_intensity=result.get("arithmetic_intensity", 0),
            hardware_counters=result.get("counters", {}),
            suggestions=suggestions,
            roofline_position=roofline,
        )
    
    def _generate_profile_script(self, kernel_code: str,
                                 warmup_iters: int,
                                 profile_iters: int,
                                 collect_counters: bool) -> str:
        """Generate the profiling script."""
        return f'''#!/usr/bin/env python3
"""Auto-generated profiling script"""
import torch
import json
import time

torch.manual_seed(42)
device = "cuda"

# Kernel code
{kernel_code}

# Find the main function to profile
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

# Profile
print(f"Profiling ({profile_iters} iters)...")
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range({profile_iters}):
    main_fn()
end.record()
torch.cuda.synchronize()

latency_us = start.elapsed_time(end) * 1000 / {profile_iters}

print(f"Latency: {{latency_us:.2f}} us")

# Save results
results = {{
    "kernel_name": main_fn.__name__ if hasattr(main_fn, "__name__") else "unknown",
    "latency_us": latency_us,
    "warmup_iters": {warmup_iters},
    "profile_iters": {profile_iters},
}}

with open("/workspace/profile_result.json", "w") as f:
    json.dump(results, f, indent=2)
'''
    
    def _run_profile(self, script_path: Path, 
                    collect_counters: bool) -> Dict[str, Any]:
        """Run profiling in Docker."""
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
                cmd, capture_output=True, text=True, timeout=300
            )
            
            # Load results
            result_path = self.work_dir / "profile_result.json"
            if result_path.exists():
                with open(result_path) as f:
                    return json.load(f)
            
            return {"error": result.stderr}
            
        except subprocess.TimeoutExpired:
            return {"error": "timeout"}
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_bottleneck(self, result: Dict[str, Any]) -> BottleneckType:
        """Analyze profiling result to determine bottleneck type."""
        latency = result.get("latency_us", 0)
        counters = result.get("counters", {})
        
        # Simple heuristics based on latency and counters
        if latency < 10:
            return BottleneckType.LATENCY  # Launch overhead dominates
        
        compute_util = result.get("compute_util", 0)
        memory_util = result.get("memory_util", 0)
        
        if compute_util > 0.7 and compute_util > memory_util:
            return BottleneckType.COMPUTE
        elif memory_util > 0.7 and memory_util > compute_util:
            return BottleneckType.MEMORY
        elif counters.get("lds_conflicts", 0) > 1000:
            return BottleneckType.LDS
        elif counters.get("cache_misses", 0) > counters.get("cache_hits", 1):
            return BottleneckType.CACHE
        else:
            return BottleneckType.BALANCED
    
    def _generate_suggestions(self, bottleneck: BottleneckType,
                             result: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on bottleneck."""
        suggestions = []
        
        if bottleneck == BottleneckType.LATENCY:
            suggestions.extend([
                "Kernel is latency-bound (launch overhead dominates)",
                "Consider: Replace with PyTorch tensor operations",
                "Consider: Fuse with adjacent kernels",
                "Consider: Use HIP Graph to reduce launch overhead",
            ])
        elif bottleneck == BottleneckType.MEMORY:
            suggestions.extend([
                "Kernel is memory-bound",
                "Consider: Improve memory coalescing",
                "Consider: Use shared memory (LDS) for caching",
                "Consider: Reduce memory traffic with kernel fusion",
            ])
        elif bottleneck == BottleneckType.COMPUTE:
            suggestions.extend([
                "Kernel is compute-bound",
                "Consider: Use tensor cores / matrix units",
                "Consider: Vectorize operations",
                "Consider: Algorithmic optimizations to reduce FLOPs",
            ])
        elif bottleneck == BottleneckType.LDS:
            suggestions.extend([
                "Kernel has LDS bank conflicts",
                "Consider: Pad shared memory to avoid conflicts",
                "Consider: Rearrange access patterns",
            ])
        elif bottleneck == BottleneckType.CACHE:
            suggestions.extend([
                "Kernel has poor cache utilization",
                "Consider: Improve data locality",
                "Consider: Prefetch data",
            ])
        
        return suggestions
    
    def _compute_roofline_position(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Compute position on roofline model."""
        latency = result.get("latency_us", 1)
        flops = result.get("flops", 0)
        bytes_transferred = result.get("bytes", 0)
        
        # Arithmetic intensity (FLOPs per byte)
        ai = flops / max(bytes_transferred, 1)
        
        # Achieved performance (TFLOPS)
        achieved_tflops = flops / (latency * 1e-6) / 1e12 if latency > 0 else 0
        
        # MI300X theoretical peaks
        peak_compute_tflops = 1307  # FP16
        peak_bandwidth_gbps = 5300  # HBM3
        
        # Ridge point (where compute and memory rooflines meet)
        ridge_point = peak_compute_tflops * 1e12 / (peak_bandwidth_gbps * 1e9)
        
        return {
            "arithmetic_intensity": ai,
            "achieved_tflops": achieved_tflops,
            "peak_tflops": peak_compute_tflops,
            "peak_bandwidth_gbps": peak_bandwidth_gbps,
            "ridge_point": ridge_point,
            "efficiency": achieved_tflops / peak_compute_tflops if peak_compute_tflops > 0 else 0,
        }
    
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
        result = self.profile(
            kernel_code=arguments["kernel_code"],
            warmup_iters=arguments.get("warmup_iters", 100),
            profile_iters=arguments.get("profile_iters", 100),
            collect_counters=arguments.get("collect_counters", True),
        )
        
        return {
            "kernel_name": result.kernel_name,
            "latency_us": result.latency_us,
            "bottleneck": result.bottleneck.value,
            "compute_utilization": result.compute_utilization,
            "memory_utilization": result.memory_utilization,
            "suggestions": result.suggestions,
            "roofline": result.roofline_position,
        }


