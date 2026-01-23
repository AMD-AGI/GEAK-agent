#!/usr/bin/env python3
"""
Generic Kernel Profiler - A reusable utility for profiling any GPU kernel.

This is the main entry point for the profiler utility in the agent framework.
It can profile:
- Triton kernels
- HIP kernels
- CK (Composable Kernel) kernels
- PyTorch operations
- Any custom GPU operation

Usage:
    from profiler.generic_profiler import GenericProfiler
    
    profiler = GenericProfiler(gpu_id=0)
    results = profiler.profile_module(
        module_script="path/to/profile_script.py",
        module_name="my_module"
    )
    profiler.print_analysis(results)
"""

import subprocess
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
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
class KernelMetrics:
    """Metrics for a single kernel/component."""
    name: str
    kernel_type: str  # "triton", "hip", "ck", "torch"
    
    # Timing
    mean_us: float
    std_us: float
    min_us: float = 0.0
    max_us: float = 0.0
    p50_us: float = 0.0
    p95_us: float = 0.0
    p99_us: float = 0.0
    
    # Memory
    bytes_transferred: int = 0
    achieved_bandwidth_gbps: float = 0.0
    
    # Compute
    flops_estimated: int = 0
    arithmetic_intensity: float = 0.0
    
    # Analysis
    bottleneck: BottleneckType = BottleneckType.UNKNOWN
    bandwidth_utilization_pct: float = 0.0
    compute_utilization_pct: float = 0.0
    
    # Raw data
    raw_times: List[float] = field(default_factory=list)


@dataclass
class ModuleAnalysis:
    """Analysis results for a complete module."""
    module_name: str
    
    # Components
    components: Dict[str, KernelMetrics] = field(default_factory=dict)
    
    # Full pipeline
    total_latency_us: float = 0.0
    total_std_us: float = 0.0
    
    # Bottleneck summary
    primary_bottleneck_component: str = ""
    primary_bottleneck_type: BottleneckType = BottleneckType.UNKNOWN
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Config
    config: Dict[str, Any] = field(default_factory=dict)


class GenericProfiler:
    """
    Generic profiler for any GPU kernel/module.
    
    This profiler is designed to be reusable across different kernel types
    and modules without modification.
    """
    
    # GPU specifications
    GPU_SPECS = {
        "gfx942": {  # MI300X
            "peak_tflops_fp16": 1307.0,
            "peak_tflops_fp32": 653.0,
            "peak_bandwidth_gbps": 5300.0,
            "num_cus": 304,
            "clock_mhz": 2100,
        },
        "gfx950": {  # MI355X
            "peak_tflops_fp16": 1600.0,
            "peak_tflops_fp32": 800.0,
            "peak_bandwidth_gbps": 8000.0,
            "num_cus": 304,
            "clock_mhz": 2500,
        },
    }
    
    def __init__(
        self,
        gpu_id: int = 0,
        docker_image: str = "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x",
        gpu_arch: str = "gfx950",
        work_dir: Optional[str] = None,
    ):
        """
        Initialize the profiler.
        
        Args:
            gpu_id: GPU device ID
            docker_image: Docker image with ROCm environment
            gpu_arch: GPU architecture (gfx942 for MI300X, gfx950 for MI355X)
            work_dir: Working directory for temporary files
        """
        self.gpu_id = gpu_id
        self.docker_image = docker_image
        self.gpu_arch = gpu_arch
        self.specs = self.GPU_SPECS.get(gpu_arch, self.GPU_SPECS["gfx950"])
        
        if work_dir:
            self.work_dir = Path(work_dir)
        else:
            self.work_dir = Path(tempfile.mkdtemp(prefix="kernel_profiler_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Ridge point for roofline model
        self.ridge_point = (
            self.specs["peak_tflops_fp16"] * 1e12 /
            (self.specs["peak_bandwidth_gbps"] * 1e9)
        )
    
    def profile_script(
        self,
        script_content: str,
        output_json: str = "results.json",
        timeout: int = 900,
    ) -> Dict:
        """
        Profile a script that outputs JSON results.
        
        Args:
            script_content: Python script content
            output_json: Name of output JSON file
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with profiling results
        """
        # Write script
        script_path = self.work_dir / "profile_script.py"
        script_path.write_text(script_content)
        
        # Run in Docker
        cmd = [
            "docker", "run", "--rm",
            "--device=/dev/kfd", "--device=/dev/dri", "--ipc=host",
            "-e", f"HIP_VISIBLE_DEVICES={self.gpu_id}",
            "-v", f"{self.work_dir}:/workspace",
            self.docker_image,
            "python3", "/workspace/profile_script.py"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        # Load results
        results_path = self.work_dir / output_json
        if results_path.exists():
            with open(results_path) as f:
                return json.load(f)
        
        return {"error": "No results produced", "stdout": result.stdout, "stderr": result.stderr}
    
    def analyze_results(self, results: Dict, module_name: str = "module") -> ModuleAnalysis:
        """
        Analyze profiling results and generate recommendations.
        
        Args:
            results: Raw profiling results (dict with components)
            module_name: Name of the module
            
        Returns:
            ModuleAnalysis with detailed analysis
        """
        analysis = ModuleAnalysis(module_name=module_name)
        
        if "config" in results:
            analysis.config = results["config"]
        
        # Process components
        components = results.get("components", {})
        for name, data in components.items():
            metrics = self._create_metrics(name, data)
            analysis.components[name] = metrics
        
        # Full pipeline metrics
        if "full_pipeline" in results:
            analysis.total_latency_us = results["full_pipeline"].get("mean_us", 0)
            analysis.total_std_us = results["full_pipeline"].get("std_us", 0)
        else:
            # Sum components
            analysis.total_latency_us = sum(
                m.mean_us for m in analysis.components.values()
            )
        
        # Find primary bottleneck
        if analysis.components:
            bottleneck_comp = max(
                analysis.components.items(),
                key=lambda x: x[1].mean_us
            )
            analysis.primary_bottleneck_component = bottleneck_comp[0]
            analysis.primary_bottleneck_type = bottleneck_comp[1].bottleneck
        
        # Generate recommendations
        analysis.recommendations = self._generate_recommendations(analysis)
        
        return analysis
    
    def _create_metrics(self, name: str, data: Dict) -> KernelMetrics:
        """Create KernelMetrics from raw data."""
        metrics = KernelMetrics(
            name=name,
            kernel_type=data.get("type", "unknown").lower(),
            mean_us=data.get("mean_us", 0),
            std_us=data.get("std_us", 0),
            min_us=data.get("min_us", 0),
            max_us=data.get("max_us", 0),
            p50_us=data.get("p50_us", 0),
            p95_us=data.get("p95_us", 0),
            p99_us=data.get("p99_us", 0),
            bytes_transferred=data.get("bytes_transferred", 0),
            flops_estimated=data.get("flops_estimated", 0),
        )
        
        # Calculate derived metrics
        if metrics.mean_us > 0:
            if metrics.bytes_transferred > 0:
                metrics.achieved_bandwidth_gbps = (
                    metrics.bytes_transferred / (metrics.mean_us * 1e3)
                )
                metrics.bandwidth_utilization_pct = (
                    metrics.achieved_bandwidth_gbps / 
                    self.specs["peak_bandwidth_gbps"] * 100
                )
            
            if metrics.flops_estimated > 0 and metrics.bytes_transferred > 0:
                metrics.arithmetic_intensity = (
                    metrics.flops_estimated / metrics.bytes_transferred
                )
        
        # Determine bottleneck
        metrics.bottleneck = self._classify_bottleneck(metrics)
        
        return metrics
    
    def _classify_bottleneck(self, metrics: KernelMetrics) -> BottleneckType:
        """Classify the kernel bottleneck."""
        # Very short kernels are latency bound
        if metrics.mean_us < 5.0:
            return BottleneckType.LATENCY
        
        ai = metrics.arithmetic_intensity
        
        if ai == 0:
            # Pure memory operation
            return BottleneckType.MEMORY
        elif ai < self.ridge_point / 10:
            return BottleneckType.MEMORY
        elif ai < self.ridge_point:
            return BottleneckType.MEMORY
        elif ai > self.ridge_point * 2:
            return BottleneckType.COMPUTE
        else:
            return BottleneckType.BALANCED
    
    def _generate_recommendations(self, analysis: ModuleAnalysis) -> List[str]:
        """Generate optimization recommendations."""
        recs = []
        
        bottleneck_type = analysis.primary_bottleneck_type
        bottleneck_comp = analysis.primary_bottleneck_component
        
        if bottleneck_comp:
            metrics = analysis.components[bottleneck_comp]
            pct = metrics.mean_us / analysis.total_latency_us * 100
            recs.append(f"Primary bottleneck: {bottleneck_comp} ({metrics.mean_us:.2f} us, {pct:.1f}% of total)")
        
        if bottleneck_type == BottleneckType.MEMORY:
            recs.append("[HIGH] Vectorize memory accesses (use float4/int4)")
            recs.append("[HIGH] Improve memory coalescing patterns")
            recs.append("[MED] Fuse with adjacent operations to increase arithmetic intensity")
            recs.append("[MED] Use shared memory for frequently accessed data")
            
        elif bottleneck_type == BottleneckType.LATENCY:
            recs.append("[HIGH] Fuse adjacent kernels to reduce launch overhead")
            recs.append("[HIGH] Use persistent kernels for small workloads")
            recs.append("[MED] Batch multiple operations together")
            recs.append("[MED] Consider CUDAGraphs/HIPGraphs for launch optimization")
            
        elif bottleneck_type == BottleneckType.COMPUTE:
            recs.append("[HIGH] Use tensor cores (MFMA instructions on AMD)")
            recs.append("[MED] Reduce unnecessary computation")
            recs.append("[MED] Consider algorithmic optimizations")
            recs.append("[LOW] Already memory-efficient")
            
        elif bottleneck_type == BottleneckType.LDS:
            recs.append("[HIGH] Pad shared memory to avoid bank conflicts")
            recs.append("[MED] Reduce shared memory usage")
            recs.append("[MED] Reorganize data layout in LDS")
            
        elif bottleneck_type == BottleneckType.CACHE:
            recs.append("[HIGH] Improve data locality")
            recs.append("[HIGH] Use blocking/tiling strategies")
            recs.append("[MED] Prefetch data")
            
        else:
            recs.append("[MED] Profile with hardware counters for detailed analysis")
            recs.append("[MED] Consider both compute and memory optimizations")
        
        return recs
    
    def print_analysis(self, analysis: ModuleAnalysis):
        """Print a formatted analysis report."""
        print("=" * 80)
        print(f"MODULE ANALYSIS: {analysis.module_name}")
        print("=" * 80)
        
        if analysis.config:
            print("\nConfiguration:")
            for k, v in analysis.config.items():
                print(f"  {k}: {v}")
        
        print("\n" + "-" * 80)
        print("COMPONENT BREAKDOWN")
        print("-" * 80)
        print(f"{'Component':<25} {'Type':<10} {'Latency':<15} {'% Total':<10} {'Bottleneck':<15}")
        print("-" * 80)
        
        for name, metrics in analysis.components.items():
            pct = metrics.mean_us / analysis.total_latency_us * 100 if analysis.total_latency_us > 0 else 0
            print(f"{name:<25} {metrics.kernel_type:<10} {metrics.mean_us:>6.2f} ± {metrics.std_us:>4.2f} us  {pct:>5.1f}%     {metrics.bottleneck.value:<15}")
        
        print("-" * 80)
        print(f"{'TOTAL':<25} {'mixed':<10} {analysis.total_latency_us:>6.2f} ± {analysis.total_std_us:>4.2f} us  {'100.0%':>8}")
        
        print("\n" + "-" * 80)
        print("DETAILED METRICS")
        print("-" * 80)
        
        for name, metrics in analysis.components.items():
            print(f"\n{name} ({metrics.kernel_type}):")
            print(f"  Timing:      P50={metrics.p50_us:.2f} / P95={metrics.p95_us:.2f} / P99={metrics.p99_us:.2f} us")
            print(f"  Bandwidth:   {metrics.achieved_bandwidth_gbps:.2f} GB/s ({metrics.bandwidth_utilization_pct:.3f}% of peak)")
            print(f"  AI:          {metrics.arithmetic_intensity:.4f} FLOP/byte")
            print(f"  Bottleneck:  {metrics.bottleneck.value}")
        
        print("\n" + "-" * 80)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("-" * 80)
        
        for i, rec in enumerate(analysis.recommendations, 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "=" * 80)
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)


# Convenience function for quick profiling
def profile_module(
    script_content: str,
    module_name: str = "module",
    gpu_id: int = 0,
    print_results: bool = True,
) -> ModuleAnalysis:
    """
    One-liner function to profile a module.
    
    Args:
        script_content: Python script that outputs profiling results to /workspace/results.json
        module_name: Name of the module
        gpu_id: GPU device ID
        print_results: Whether to print the analysis
        
    Returns:
        ModuleAnalysis with results
    """
    profiler = GenericProfiler(gpu_id=gpu_id)
    try:
        results = profiler.profile_script(script_content, output_json="results.json")
        analysis = profiler.analyze_results(results, module_name=module_name)
        if print_results:
            profiler.print_analysis(analysis)
        return analysis
    finally:
        profiler.cleanup()

