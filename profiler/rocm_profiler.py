#!/usr/bin/env python3
"""
ROCm Kernel Profiler - Generic profiling utility for GPU kernels.

Supports:
- Triton kernels
- HIP kernels  
- CK (Composable Kernel) kernels
- Any GPU kernel callable from Python

Uses rocprofv3 (rocprofiler-sdk) for hardware performance counters.
"""

import os
import json
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
import time


class KernelType(Enum):
    TRITON = "triton"
    HIP = "hip"
    CK = "ck"
    TORCH = "torch"
    UNKNOWN = "unknown"


@dataclass
class KernelProfile:
    """Profile data for a single kernel."""
    kernel_name: str
    kernel_type: KernelType
    
    # Timing metrics (in microseconds)
    duration_us: float
    duration_std: float = 0.0
    
    # Hardware counters
    waves: int = 0
    valu_insts: int = 0
    salu_insts: int = 0
    lds_bank_conflicts: int = 0
    l1_cache_hits: int = 0
    l1_cache_misses: int = 0
    l2_cache_hits: int = 0
    l2_cache_misses: int = 0
    
    # Memory metrics
    global_load_bytes: int = 0
    global_store_bytes: int = 0
    lds_bytes: int = 0
    
    # Launch configuration
    grid_size: tuple = (0, 0, 0)
    block_size: tuple = (0, 0, 0)
    shared_mem_bytes: int = 0
    
    # Derived metrics (computed after profiling)
    valu_utilization: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    arithmetic_intensity: float = 0.0
    
    # Raw profiler output
    raw_counters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileAnalysis:
    """Analysis results from profiling."""
    kernel_profile: KernelProfile
    
    # Bottleneck identification
    bottleneck_type: str = "unknown"  # compute, memory, latency, balanced
    bottleneck_confidence: float = 0.0
    
    # Performance metrics
    achieved_tflops: float = 0.0
    achieved_bandwidth_gbps: float = 0.0
    theoretical_peak_tflops: float = 0.0
    theoretical_peak_bandwidth_gbps: float = 0.0
    
    # Optimization suggestions
    suggestions: List[str] = field(default_factory=list)
    
    # Roofline position
    roofline_x: float = 0.0  # Arithmetic intensity (FLOP/byte)
    roofline_y: float = 0.0  # Performance (TFLOP/s)


class ROCmProfiler:
    """
    Generic ROCm kernel profiler using rocprofv3.
    
    Usage:
        profiler = ROCmProfiler(gpu_id=0)
        
        # Profile a kernel function
        profile = profiler.profile_kernel(
            kernel_fn=my_kernel,
            args=(x, y, z),
            num_warmup=100,
            num_iterations=1000
        )
        
        # Analyze results
        analysis = profiler.analyze(profile)
        print(analysis.bottleneck_type)
        print(analysis.suggestions)
    """
    
    # MI300X/MI355X specs (gfx942/gfx950)
    GPU_SPECS = {
        "gfx942": {
            "peak_tflops_fp16": 1307.0,
            "peak_tflops_fp32": 653.0,
            "peak_bandwidth_gbps": 5300.0,
            "num_cus": 304,
            "waves_per_cu": 32,
            "lds_per_cu_kb": 64,
        },
        "gfx950": {
            "peak_tflops_fp16": 1600.0,  # Estimated for MI355X
            "peak_tflops_fp32": 800.0,
            "peak_bandwidth_gbps": 8000.0,
            "num_cus": 304,
            "waves_per_cu": 32,
            "lds_per_cu_kb": 64,
        },
    }
    
    # Performance counters to collect
    DEFAULT_COUNTERS = [
        "SQ_WAVES",
        "SQ_INSTS_VALU",
        "SQ_INSTS_SALU", 
        "SQ_WAIT_INST_LDS",
        "TCC_HIT",
        "TCC_MISS",
        "TCP_TCC_READ_REQ",
        "TCP_TCC_WRITE_REQ",
        "SQ_LDS_BANK_CONFLICT",
        "SQ_ACTIVE_INST_VALU",
        "GRBM_COUNT",
        "GRBM_GUI_ACTIVE",
    ]
    
    def __init__(
        self,
        gpu_id: int = 0,
        docker_image: str = "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x",
        work_dir: Optional[str] = None,
        gpu_arch: str = "gfx950",
    ):
        """
        Initialize the ROCm profiler.
        
        Args:
            gpu_id: GPU device ID to profile on
            docker_image: Docker image with ROCm and profiler tools
            work_dir: Working directory for temporary files
            gpu_arch: GPU architecture (gfx942 for MI300X, gfx950 for MI355X)
        """
        self.gpu_id = gpu_id
        self.docker_image = docker_image
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="rocm_profiler_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_arch = gpu_arch
        self.specs = self.GPU_SPECS.get(gpu_arch, self.GPU_SPECS["gfx950"])
        
    def profile_kernel(
        self,
        kernel_fn: Callable,
        args: tuple = (),
        kwargs: dict = None,
        num_warmup: int = 100,
        num_iterations: int = 1000,
        kernel_name: Optional[str] = None,
        kernel_type: KernelType = KernelType.UNKNOWN,
        collect_counters: bool = True,
        counters: List[str] = None,
    ) -> KernelProfile:
        """
        Profile a kernel function.
        
        Args:
            kernel_fn: The kernel function to profile
            args: Positional arguments to pass to kernel
            kwargs: Keyword arguments to pass to kernel
            num_warmup: Number of warmup iterations
            num_iterations: Number of profiling iterations
            kernel_name: Name for the kernel (auto-detected if not provided)
            kernel_type: Type of kernel (Triton, HIP, CK, etc.)
            collect_counters: Whether to collect hardware performance counters
            counters: List of specific counters to collect
            
        Returns:
            KernelProfile with profiling results
        """
        kwargs = kwargs or {}
        counters = counters or self.DEFAULT_COUNTERS
        
        # Auto-detect kernel name
        if kernel_name is None:
            kernel_name = getattr(kernel_fn, '__name__', str(kernel_fn))
            
        # Auto-detect kernel type
        if kernel_type == KernelType.UNKNOWN:
            kernel_type = self._detect_kernel_type(kernel_fn)
            
        # Create profile script
        script_path = self._create_profile_script(
            kernel_fn, args, kwargs, num_warmup, num_iterations, kernel_name
        )
        
        # Run profiling
        timing_results = self._run_timing_profile(script_path)
        
        counter_results = {}
        if collect_counters:
            counter_results = self._run_counter_profile(script_path, counters)
            
        # Parse results
        profile = self._parse_results(
            kernel_name, kernel_type, timing_results, counter_results
        )
        
        return profile
    
    def profile_script(
        self,
        script_content: str,
        kernel_name: str = "kernel",
        kernel_type: KernelType = KernelType.UNKNOWN,
        collect_counters: bool = True,
        counters: List[str] = None,
    ) -> KernelProfile:
        """
        Profile a kernel defined in a script string.
        
        This is useful when you need to define custom setup/teardown logic
        or when profiling kernels that require special initialization.
        
        Args:
            script_content: Python script content that runs the kernel
            kernel_name: Name for the kernel
            kernel_type: Type of kernel
            collect_counters: Whether to collect hardware counters
            counters: Specific counters to collect
            
        Returns:
            KernelProfile with results
        """
        counters = counters or self.DEFAULT_COUNTERS
        
        # Write script to file
        script_path = self.work_dir / "profile_target.py"
        script_path.write_text(script_content)
        
        # Run profiling
        timing_results = self._run_timing_profile(script_path)
        
        counter_results = {}
        if collect_counters:
            counter_results = self._run_counter_profile(script_path, counters)
            
        # Parse results
        profile = self._parse_results(
            kernel_name, kernel_type, timing_results, counter_results
        )
        
        return profile
    
    def analyze(self, profile: KernelProfile) -> ProfileAnalysis:
        """
        Analyze a kernel profile to identify bottlenecks and suggest optimizations.
        
        Args:
            profile: KernelProfile from profiling
            
        Returns:
            ProfileAnalysis with bottleneck identification and suggestions
        """
        analysis = ProfileAnalysis(kernel_profile=profile)
        
        # Calculate achieved performance
        if profile.valu_insts > 0 and profile.duration_us > 0:
            # Estimate FLOPS (2 FLOPs per VALU instruction is a rough estimate)
            flops = profile.valu_insts * 2
            analysis.achieved_tflops = (flops / profile.duration_us) / 1e6
            
        # Calculate memory bandwidth
        total_bytes = profile.global_load_bytes + profile.global_store_bytes
        if total_bytes > 0 and profile.duration_us > 0:
            analysis.achieved_bandwidth_gbps = (total_bytes / profile.duration_us) / 1e3
            
        # Set theoretical peaks
        analysis.theoretical_peak_tflops = self.specs["peak_tflops_fp16"]
        analysis.theoretical_peak_bandwidth_gbps = self.specs["peak_bandwidth_gbps"]
        
        # Calculate roofline position
        if total_bytes > 0:
            analysis.roofline_x = (profile.valu_insts * 2) / total_bytes  # Arithmetic intensity
        analysis.roofline_y = analysis.achieved_tflops
        
        # Identify bottleneck
        analysis.bottleneck_type, analysis.bottleneck_confidence = self._identify_bottleneck(
            profile, analysis
        )
        
        # Generate optimization suggestions
        analysis.suggestions = self._generate_suggestions(profile, analysis)
        
        return analysis
    
    def _detect_kernel_type(self, kernel_fn: Callable) -> KernelType:
        """Detect the type of kernel from its module."""
        module = getattr(kernel_fn, '__module__', '')
        
        if 'triton' in module.lower():
            return KernelType.TRITON
        elif 'hip' in module.lower() or 'cuda' in module.lower():
            return KernelType.HIP
        elif 'ck' in module.lower() or 'composable' in module.lower():
            return KernelType.CK
        elif 'torch' in module.lower():
            return KernelType.TORCH
        else:
            return KernelType.UNKNOWN
    
    def _create_profile_script(
        self,
        kernel_fn: Callable,
        args: tuple,
        kwargs: dict,
        num_warmup: int,
        num_iterations: int,
        kernel_name: str,
    ) -> Path:
        """Create a profiling script for the kernel."""
        # This is a simplified version - in practice you'd serialize the kernel
        # and its arguments properly
        
        script = f'''#!/usr/bin/env python3
"""Auto-generated profiling script for {kernel_name}"""
import torch
import time
import json
import os
os.environ['TRITON_PRINT_AUTOTUNING'] = '0'

# Import kernel (this would need to be customized per kernel)
# For now, this is a placeholder

device = "cuda:{self.gpu_id}"

# Warmup
print("Warming up...")
for _ in range({num_warmup}):
    pass  # kernel_fn(*args, **kwargs)
torch.cuda.synchronize()

# Timing
print("Profiling...")
times = []
for _ in range({num_iterations}):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    pass  # kernel_fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) * 1000)  # us

import numpy as np
results = {{
    "mean_us": np.mean(times),
    "std_us": np.std(times),
    "min_us": np.min(times),
    "max_us": np.max(times),
    "kernel_name": "{kernel_name}",
}}

with open("/workspace/timing_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Mean: {{results['mean_us']:.3f}} us")
print("Profiling complete!")
'''
        
        script_path = self.work_dir / "profile_target.py"
        script_path.write_text(script)
        return script_path
    
    def _run_timing_profile(self, script_path: Path) -> Dict:
        """Run basic timing profiling with rocprofv3 kernel trace."""
        results_dir = self.work_dir / "timing_results"
        results_dir.mkdir(exist_ok=True)
        
        cmd = [
            "docker", "run", "--rm",
            "--device=/dev/kfd", "--device=/dev/dri", "--ipc=host",
            "--cap-add=SYS_PTRACE", "--security-opt", "seccomp=unconfined",
            "-e", f"HIP_VISIBLE_DEVICES={self.gpu_id}",
            "-v", f"{self.work_dir}:/workspace",
            self.docker_image,
            "bash", "-c",
            f"rocprofv3 --kernel-trace -o /workspace/trace -- python3 /workspace/{script_path.name}"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            # Parse results
            timing_file = self.work_dir / "timing_results.json"
            if timing_file.exists():
                with open(timing_file) as f:
                    return json.load(f)
                    
            # Try to parse rocprofv3 output
            trace_file = self.work_dir / "trace_results.db"
            if trace_file.exists():
                return self._parse_rocprofv3_trace(trace_file)
                
        except subprocess.TimeoutExpired:
            print("Profiling timed out")
        except Exception as e:
            print(f"Profiling error: {e}")
            
        return {}
    
    def _run_counter_profile(self, script_path: Path, counters: List[str]) -> Dict:
        """Run hardware counter profiling."""
        # Create counter input file
        counter_file = self.work_dir / "counters.txt"
        counter_content = "\n".join([f"pmc: {c}" for c in counters])
        counter_file.write_text(counter_content)
        
        cmd = [
            "docker", "run", "--rm",
            "--device=/dev/kfd", "--device=/dev/dri", "--ipc=host",
            "--cap-add=SYS_PTRACE", "--security-opt", "seccomp=unconfined",
            "-e", f"HIP_VISIBLE_DEVICES={self.gpu_id}",
            "-v", f"{self.work_dir}:/workspace",
            self.docker_image,
            "bash", "-c",
            f"rocprofv3 -i /workspace/counters.txt -o /workspace/counters -- python3 /workspace/{script_path.name}"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            # Parse counter results
            counter_results = self.work_dir / "counters_results.csv"
            if counter_results.exists():
                return self._parse_counter_csv(counter_results)
                
        except subprocess.TimeoutExpired:
            print("Counter profiling timed out")
        except Exception as e:
            print(f"Counter profiling error: {e}")
            
        return {}
    
    def _parse_rocprofv3_trace(self, trace_file: Path) -> Dict:
        """Parse rocprofv3 SQLite trace database."""
        try:
            import sqlite3
            conn = sqlite3.connect(str(trace_file))
            cursor = conn.cursor()
            
            # Query kernel dispatch info
            cursor.execute("""
                SELECT kernel_name, duration_ns, workgroup_size_x, workgroup_size_y, workgroup_size_z,
                       grid_size_x, grid_size_y, grid_size_z
                FROM rocpd_kernel_dispatch
            """)
            
            results = {"kernels": []}
            for row in cursor.fetchall():
                results["kernels"].append({
                    "name": row[0],
                    "duration_us": row[1] / 1000.0,
                    "block_size": (row[2], row[3], row[4]),
                    "grid_size": (row[5], row[6], row[7]),
                })
                
            conn.close()
            return results
            
        except Exception as e:
            print(f"Error parsing trace: {e}")
            return {}
    
    def _parse_counter_csv(self, counter_file: Path) -> Dict:
        """Parse rocprofv3 counter CSV output."""
        try:
            import csv
            results = {}
            
            with open(counter_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key, value in row.items():
                        if key not in results:
                            results[key] = []
                        try:
                            results[key].append(float(value))
                        except (ValueError, TypeError):
                            results[key].append(value)
                            
            return results
            
        except Exception as e:
            print(f"Error parsing counters: {e}")
            return {}
    
    def _parse_results(
        self,
        kernel_name: str,
        kernel_type: KernelType,
        timing_results: Dict,
        counter_results: Dict,
    ) -> KernelProfile:
        """Parse profiling results into a KernelProfile."""
        profile = KernelProfile(
            kernel_name=kernel_name,
            kernel_type=kernel_type,
            duration_us=timing_results.get("mean_us", 0.0),
            duration_std=timing_results.get("std_us", 0.0),
        )
        
        # Parse hardware counters
        if counter_results:
            profile.waves = int(sum(counter_results.get("SQ_WAVES", [0])))
            profile.valu_insts = int(sum(counter_results.get("SQ_INSTS_VALU", [0])))
            profile.salu_insts = int(sum(counter_results.get("SQ_INSTS_SALU", [0])))
            profile.lds_bank_conflicts = int(sum(counter_results.get("SQ_LDS_BANK_CONFLICT", [0])))
            profile.l2_cache_hits = int(sum(counter_results.get("TCC_HIT", [0])))
            profile.l2_cache_misses = int(sum(counter_results.get("TCC_MISS", [0])))
            profile.raw_counters = counter_results
            
        # Parse kernel info
        kernels = timing_results.get("kernels", [])
        if kernels:
            kernel_info = kernels[0]
            profile.grid_size = kernel_info.get("grid_size", (0, 0, 0))
            profile.block_size = kernel_info.get("block_size", (0, 0, 0))
            
        return profile
    
    def _identify_bottleneck(
        self,
        profile: KernelProfile,
        analysis: ProfileAnalysis,
    ) -> tuple:
        """Identify the performance bottleneck."""
        # Ridge point: where compute roof meets memory roof
        ridge_point = (
            analysis.theoretical_peak_tflops / 
            (analysis.theoretical_peak_bandwidth_gbps / 1000)
        )
        
        if analysis.roofline_x < ridge_point * 0.5:
            # Low arithmetic intensity -> memory bound
            return "memory", 0.8
        elif analysis.roofline_x > ridge_point * 2.0:
            # High arithmetic intensity -> compute bound
            return "compute", 0.8
        else:
            # Near ridge point -> balanced or latency bound
            if profile.duration_us < 10:
                return "latency", 0.6  # Very short kernel, launch overhead dominates
            return "balanced", 0.5
    
    def _generate_suggestions(
        self,
        profile: KernelProfile,
        analysis: ProfileAnalysis,
    ) -> List[str]:
        """Generate optimization suggestions based on profile."""
        suggestions = []
        
        # Memory-bound optimizations
        if analysis.bottleneck_type == "memory":
            suggestions.append("Memory-bound: Consider improving data locality")
            suggestions.append("Use vectorized loads/stores (e.g., float4)")
            suggestions.append("Increase arithmetic intensity by fusing operations")
            
            if profile.l2_cache_misses > profile.l2_cache_hits:
                suggestions.append("High L2 cache miss rate: Optimize memory access patterns")
                
            if profile.lds_bank_conflicts > 0:
                suggestions.append(f"LDS bank conflicts detected ({profile.lds_bank_conflicts}): Pad shared memory")
                
        # Compute-bound optimizations
        elif analysis.bottleneck_type == "compute":
            suggestions.append("Compute-bound: Already memory-efficient")
            
            if profile.valu_insts > 0 and profile.salu_insts > 0:
                ratio = profile.valu_insts / (profile.valu_insts + profile.salu_insts)
                if ratio < 0.9:
                    suggestions.append(f"VALU utilization {ratio:.1%}: Reduce scalar operations")
                    
            suggestions.append("Consider using tensor cores (MFMA instructions)")
            
        # Latency-bound optimizations
        elif analysis.bottleneck_type == "latency":
            suggestions.append("Latency-bound: Kernel is too short, launch overhead dominates")
            suggestions.append("Fuse with adjacent kernels to increase work per launch")
            suggestions.append("Use persistent kernels for small workloads")
            
        # General suggestions
        if profile.waves < self.specs["num_cus"] * 4:
            suggestions.append(f"Low occupancy ({profile.waves} waves): Increase parallelism")
            
        return suggestions
    
    def get_gpu_specs(self) -> Dict:
        """Get GPU specifications."""
        return self.specs.copy()
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)


# Convenience function for quick profiling
def quick_profile(
    script: str,
    kernel_name: str = "kernel",
    gpu_id: int = 0,
) -> ProfileAnalysis:
    """
    Quick one-liner profiling.
    
    Args:
        script: Python script content that runs the kernel
        kernel_name: Name for the kernel
        gpu_id: GPU device ID
        
    Returns:
        ProfileAnalysis with results
    """
    profiler = ROCmProfiler(gpu_id=gpu_id)
    try:
        profile = profiler.profile_script(script, kernel_name=kernel_name)
        analysis = profiler.analyze(profile)
        return analysis
    finally:
        profiler.cleanup()

