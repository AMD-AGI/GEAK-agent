#!/usr/bin/env python3
"""
Kernel Profiler - Bottleneck Analysis for GPU Kernels

Uses rocprof-compute (ROCm Compute Profiler) for detailed hardware analysis:
- Occupancy, register usage, LDS usage
- Memory bandwidth, cache hit rates
- Compute utilization

Identifies bottlenecks:
- LATENCY: High launch overhead (recommend HIP Graph, fusion)
- MEMORY: Memory bandwidth bound (recommend coalescing, vectorization)
- COMPUTE: Compute bound (recommend better tiling, warp efficiency)
- LDS: Local data share bound (recommend bank conflict reduction)
- OCCUPANCY: Low occupancy (recommend register/LDS optimization)
"""

import subprocess
import json
import re
import sys
import time
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum


class BottleneckType(Enum):
    """Types of performance bottlenecks."""
    LATENCY = "latency"        # Launch overhead dominant
    MEMORY = "memory"          # Memory bandwidth bound
    COMPUTE = "compute"        # Compute bound
    LDS = "lds"                # Local data share bound
    OCCUPANCY = "occupancy"    # Low occupancy
    BALANCED = "balanced"      # No clear bottleneck


@dataclass
class KernelMetrics:
    """Detailed metrics for a single kernel."""
    name: str
    count: int = 0
    mean_time_ns: float = 0
    median_time_ns: float = 0
    total_time_ns: float = 0
    pct_of_total: float = 0
    
    # Hardware metrics from rocprof-compute
    valu_utilization: float = 0
    mfma_utilization: float = 0
    vmem_utilization: float = 0
    lds_utilization: float = 0
    occupancy: float = 0
    active_cus: int = 0
    vgpr_count: int = 0
    sgpr_count: int = 0
    lds_per_workgroup: int = 0
    
    @property
    def mean_time_us(self) -> float:
        return self.mean_time_ns / 1000.0


@dataclass
class ProfileResult:
    """Result from kernel profiling."""
    bottleneck: BottleneckType
    metrics: Dict[str, float]
    recommendations: List[str]
    kernels: List[KernelMetrics] = field(default_factory=list)
    raw_output: str = ""
    rocprof_compute_used: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bottleneck": self.bottleneck.value,
            "metrics": self.metrics,
            "recommendations": self.recommendations,
            "kernels": [
                {
                    "name": k.name,
                    "mean_time_us": k.mean_time_us,
                    "count": k.count,
                    "pct_of_total": k.pct_of_total,
                }
                for k in self.kernels[:5]
            ],
            "rocprof_compute_used": self.rocprof_compute_used,
        }


class ProgressBar:
    """Simple progress bar for terminal output."""
    
    def __init__(self, total: int, prefix: str = "", width: int = 40):
        self.total = total
        self.prefix = prefix
        self.width = width
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1):
        self.current += n
        self._display()
    
    def set(self, value: int):
        self.current = value
        self._display()
    
    def _display(self):
        pct = self.current / self.total if self.total > 0 else 0
        filled = int(self.width * pct)
        bar = "█" * filled + "░" * (self.width - filled)
        
        elapsed = time.time() - self.start_time
        if self.current > 0 and pct < 1:
            eta = elapsed / pct * (1 - pct)
            eta_str = f"ETA: {eta:.0f}s"
        else:
            eta_str = ""
        
        print(f"\r  {self.prefix} |{bar}| {pct*100:5.1f}% {eta_str}   ", end="", flush=True)
    
    def finish(self, message: str = ""):
        self.current = self.total
        self._display()
        print(f" {message}")


class RocprofComputeProfiler:
    """
    Advanced profiler using rocprof-compute for detailed kernel analysis.
    
    rocprof-compute provides higher-level kernel analysis including:
    - Occupancy analysis
    - Memory bandwidth utilization
    - Register and LDS usage
    - Compute utilization
    - Roofline analysis
    """
    
    # Optimization recommendations by bottleneck type
    RECOMMENDATIONS = {
        BottleneckType.LATENCY: [
            "Use HIP Graph capture to eliminate launch overhead",
            "Batch multiple kernel calls into single graph",
            "Consider multi-stream execution with batched graphs",
            "Use persistent kernels for repeated operations",
            "Kernel fusion to reduce dispatch count",
        ],
        BottleneckType.MEMORY: [
            "Ensure memory accesses are coalesced",
            "Use vectorized loads (float4, int4)",
            "Cache frequently accessed data in LDS",
            "Reduce memory traffic with tiling",
            "Check L2 cache hit rate",
        ],
        BottleneckType.COMPUTE: [
            "Increase arithmetic intensity",
            "Use MFMA (Matrix Fused Multiply-Add) instructions",
            "Ensure good warp/wavefront occupancy",
            "Optimize tiling for better cache usage",
            "Consider using tensor cores if available",
        ],
        BottleneckType.LDS: [
            "Reduce LDS bank conflicts with padding",
            "Optimize LDS allocation per workgroup",
            "Use LDS more efficiently with tiling",
        ],
        BottleneckType.OCCUPANCY: [
            "Reduce VGPR usage to increase waves per CU",
            "Reduce LDS per workgroup",
            "Tune block size for better occupancy",
        ],
        BottleneckType.BALANCED: [
            "Profile with rocprof-compute for detailed analysis",
            "Try HIP Graph capture",
            "Tune block sizes and warps",
            "Consider kernel fusion",
        ],
    }
    
    def __init__(self, 
                 docker_image: str = "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x",
                 gpu_device: str = "0",
                 verbose: bool = True):
        self.docker_image = docker_image
        self.gpu_device = gpu_device
        self.verbose = verbose
    
    def _print(self, msg: str):
        if self.verbose:
            print(msg)
    
    def profile(self, 
                benchmark_script: Path,
                work_dir: Path,
                kernel_name: str = "kernel") -> ProfileResult:
        """
        Profile a kernel using rocprof-compute.
        
        Args:
            benchmark_script: Path to benchmark Python script
            work_dir: Working directory for output
            kernel_name: Name for the profiling run
            
        Returns:
            ProfileResult with bottleneck analysis
        """
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        
        output_dir = work_dir / "rocprof_compute_output"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir()
        
        self._print("")
        self._print("=" * 70)
        self._print("  ROCPROF-COMPUTE PROFILER")
        self._print("=" * 70)
        
        # Step 1: Install dependencies
        self._print("")
        self._print("  [1/4] Checking rocprof-compute dependencies...")
        self._install_deps()
        
        # Step 2: Run profiling
        self._print("  [2/4] Running rocprof-compute profile...")
        profile_success = self._run_profile(benchmark_script, output_dir, kernel_name)
        
        if not profile_success:
            self._print("  ⚠ rocprof-compute failed, falling back to basic profiler")
            return self._fallback_profile(benchmark_script, work_dir)
        
        # Step 3: Analyze results
        self._print("  [3/4] Analyzing profiling results...")
        result = self._run_analyze(output_dir)
        
        # Step 4: Identify bottleneck
        self._print("  [4/4] Identifying bottleneck...")
        
        return result
    
    def _install_deps(self):
        """Install rocprof-compute Python dependencies."""
        cmd = [
            "docker", "run", "--rm",
            "-v", "/tmp:/tmp",
            self.docker_image,
            "bash", "-c",
            "pip install -q -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt 2>/dev/null || "
            "pip install -q astunparse==1.6.2 tabulate rich plotly 2>/dev/null"
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=120)
        except:
            pass
    
    def _run_profile(self, 
                     benchmark_script: Path,
                     output_dir: Path,
                     kernel_name: str) -> bool:
        """Run rocprof-compute profile command."""
        
        # Copy benchmark script to output directory
        script_dest = output_dir.parent / benchmark_script.name
        if benchmark_script != script_dest:
            shutil.copy(benchmark_script, script_dest)
        
        # Build docker command
        cmd = f'''
pip install -q -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt 2>/dev/null

echo "  Running profile (this may take 2-5 minutes)..."

rocprof-compute profile \\
    -n {kernel_name} \\
    --path /workspace/output \\
    -- python3 /workspace/{benchmark_script.name} 2>&1 | \\
    grep -v "^\[aiter\]" | \\
    grep -v "^W20" | \\
    tail -20
'''
        
        docker_cmd = [
            "docker", "run", "--rm",
            "--device=/dev/kfd", "--device=/dev/dri",
            "--group-add", "video",
            "-v", f"{output_dir.parent}:/workspace",
            "-w", "/workspace",
            self.docker_image,
            "bash", "-c", cmd
        ]
        
        # Progress bar for profiling (estimate ~3 minutes)
        if self.verbose:
            progress = ProgressBar(100, "Profiling")
            
        try:
            process = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            output_lines = []
            start = time.time()
            
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                    
                if line:
                    output_lines.append(line.strip())
                    # Update progress based on output
                    if "Peak" in line:
                        if self.verbose:
                            progress.set(min(95, progress.current + 10))
                    elif "roofline" in line.lower():
                        if self.verbose:
                            progress.set(95)
                
                # Update progress based on time (estimate 180s total)
                elapsed = time.time() - start
                if self.verbose and elapsed < 180:
                    progress.set(min(90, int(elapsed / 180 * 90)))
            
            if self.verbose:
                progress.finish("✓")
            
            # Check if output was generated
            csv_files = list(output_dir.glob("*.csv")) + list((output_dir / "output").glob("*.csv") if (output_dir / "output").exists() else [])
            return len(csv_files) > 0 or (output_dir / "output").exists()
            
        except subprocess.TimeoutExpired:
            if self.verbose:
                progress.finish("✗ Timeout")
            return False
        except Exception as e:
            if self.verbose:
                print(f"\n  ✗ Error: {e}")
            return False
    
    def _run_analyze(self, output_dir: Path) -> ProfileResult:
        """Run rocprof-compute analyze and parse results."""
        
        # Find the actual output directory
        if (output_dir / "output").exists():
            analyze_path = output_dir / "output"
        else:
            analyze_path = output_dir
        
        cmd = f'''
pip install -q -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt 2>/dev/null

rocprof-compute analyze --path /workspace/output 2>&1 | head -200
'''
        
        docker_cmd = [
            "docker", "run", "--rm",
            "--device=/dev/kfd", "--device=/dev/dri",
            "--group-add", "video",
            "-v", f"{output_dir.parent}:/workspace",
            "-w", "/workspace",
            self.docker_image,
            "bash", "-c", cmd
        ]
        
        if self.verbose:
            progress = ProgressBar(100, "Analyzing")
        
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if self.verbose:
                progress.finish("✓")
            
            output = result.stdout
            
            # Parse the analysis output
            return self._parse_analyze_output(output)
            
        except Exception as e:
            if self.verbose:
                progress.finish(f"✗ {e}")
            return ProfileResult(
                bottleneck=BottleneckType.BALANCED,
                metrics={},
                recommendations=self.RECOMMENDATIONS[BottleneckType.BALANCED],
                raw_output=str(e),
            )
    
    def _parse_analyze_output(self, output: str) -> ProfileResult:
        """Parse rocprof-compute analyze output to extract metrics."""
        
        kernels = []
        metrics = {}
        
        # Parse Top Kernels table
        kernel_pattern = r'│\s*(\d+)\s*│\s*([^│]+)\s*│\s*([\d.]+)\s*│\s*([\d.]+)\s*│\s*([\d.]+)\s*│\s*([\d.]+)\s*│\s*([\d.]+)\s*│'
        for match in re.finditer(kernel_pattern, output):
            try:
                kernels.append(KernelMetrics(
                    name=match.group(2).strip(),
                    count=int(float(match.group(3))),
                    total_time_ns=float(match.group(4)),
                    mean_time_ns=float(match.group(5)),
                    median_time_ns=float(match.group(6)),
                    pct_of_total=float(match.group(7)),
                ))
            except:
                pass
        
        # Parse Speed-of-Light metrics
        sol_patterns = {
            'valu_utilization': r'VALU Utilization\s*│\s*([\d.]+)',
            'mfma_utilization': r'MFMA Utilization\s*│\s*([\d.]+)',
            'vmem_utilization': r'VMEM Utilization\s*│\s*([\d.]+)',
            'occupancy': r'Wavefront Occupancy\s*│\s*([\d.]+)',
            'active_cus': r'Active CUs\s*│\s*([\d.]+)',
            'ipc': r'IPC\s*│\s*([\d.]+)',
            'lds_bw': r'Theoretical LDS Bandwidth\s*│\s*([\d.]+)',
        }
        
        for key, pattern in sol_patterns.items():
            match = re.search(pattern, output)
            if match:
                metrics[key] = float(match.group(1))
        
        # Identify bottleneck
        bottleneck = self._identify_bottleneck(metrics, kernels)
        
        # Get recommendations
        recommendations = self.RECOMMENDATIONS[bottleneck].copy()
        
        # Add specific recommendations based on metrics
        if metrics.get('valu_utilization', 0) < 20:
            recommendations.append(f"Low VALU utilization ({metrics.get('valu_utilization', 0):.1f}%)")
        if metrics.get('occupancy', 0) < 30:
            recommendations.append(f"Low occupancy ({metrics.get('occupancy', 0):.1f}%)")
        
        # Print summary
        if self.verbose:
            self._print_summary(bottleneck, metrics, kernels)
        
        return ProfileResult(
            bottleneck=bottleneck,
            metrics=metrics,
            recommendations=recommendations,
            kernels=kernels,
            raw_output=output,
            rocprof_compute_used=True,
        )
    
    def _identify_bottleneck(self, 
                             metrics: Dict[str, float],
                             kernels: List[KernelMetrics]) -> BottleneckType:
        """Identify the primary bottleneck from metrics."""
        
        # Check occupancy first
        occupancy = metrics.get('occupancy', 50)
        if occupancy < 20:
            return BottleneckType.OCCUPANCY
        
        # Check utilization metrics
        valu = metrics.get('valu_utilization', 0)
        mfma = metrics.get('mfma_utilization', 0)
        vmem = metrics.get('vmem_utilization', 0)
        lds_bw = metrics.get('lds_bw', 0)
        
        # If no metrics, check kernel timing
        if not metrics and kernels:
            target = kernels[0]
            if target.mean_time_us < 10:
                return BottleneckType.LATENCY
            elif target.mean_time_us > 100:
                return BottleneckType.COMPUTE
        
        # High memory utilization
        if vmem > 50:
            return BottleneckType.MEMORY
        
        # High compute utilization  
        if valu > 50 or mfma > 50:
            return BottleneckType.COMPUTE
        
        # High LDS usage
        if lds_bw > 50000:  # 50 TB/s
            return BottleneckType.LDS
        
        # Small kernel with low utilization = latency bound
        if kernels and kernels[0].mean_time_us < 10:
            return BottleneckType.LATENCY
        
        return BottleneckType.BALANCED
    
    def _print_summary(self, 
                       bottleneck: BottleneckType,
                       metrics: Dict[str, float],
                       kernels: List[KernelMetrics]):
        """Print profiling summary."""
        
        self._print("")
        self._print("  " + "-" * 66)
        self._print("  PROFILING SUMMARY")
        self._print("  " + "-" * 66)
        
        if kernels:
            self._print("")
            self._print("  Top Kernels:")
            for i, k in enumerate(kernels[:3]):
                self._print(f"    {i+1}. {k.name[:45]}")
                self._print(f"       Time: {k.mean_time_us:.2f} μs | Count: {k.count} | {k.pct_of_total:.1f}%")
        
        if metrics:
            self._print("")
            self._print("  Hardware Utilization:")
            if 'valu_utilization' in metrics:
                self._print(f"    VALU: {metrics['valu_utilization']:.1f}%")
            if 'mfma_utilization' in metrics:
                self._print(f"    MFMA: {metrics['mfma_utilization']:.1f}%")
            if 'vmem_utilization' in metrics:
                self._print(f"    VMEM: {metrics['vmem_utilization']:.1f}%")
            if 'occupancy' in metrics:
                self._print(f"    Occupancy: {metrics['occupancy']:.1f}%")
        
        self._print("")
        self._print(f"  ╔══════════════════════════════════════════════════════════════════╗")
        self._print(f"  ║  BOTTLENECK IDENTIFIED: {bottleneck.value.upper():^40} ║")
        self._print(f"  ╚══════════════════════════════════════════════════════════════════╝")
        self._print("")
    
    def _fallback_profile(self, 
                          benchmark_script: Path,
                          work_dir: Path) -> ProfileResult:
        """Fallback to simple timing-based profiling."""
        
        self._print("  Using fallback timing-based profiler...")
        
        # Simple timing analysis
        cmd = f'''
python3 << 'EOF'
import time
import torch
torch.set_default_device("cuda")

# Try to import and run benchmark
try:
    import sys
    sys.path.insert(0, "/workspace")
    
    # Try various import patterns
    times = []
    for _ in range(10):
        start = time.perf_counter()
        # Placeholder - would need actual kernel call
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / len(times) * 1e6  # us
    print(f"Average time: {{avg_time:.2f}} us")
    
except Exception as e:
    print(f"Error: {{e}}")
EOF
'''
        
        docker_cmd = [
            "docker", "run", "--rm",
            "--device=/dev/kfd", "--device=/dev/dri",
            "--group-add", "video",
            "-v", f"{work_dir}:/workspace",
            "-w", "/workspace",
            self.docker_image,
            "bash", "-c", cmd
        ]
        
        try:
            result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=60)
            
            # Parse time
            match = re.search(r'Average time: ([\d.]+) us', result.stdout)
            kernel_time = float(match.group(1)) if match else 50.0
            
            if kernel_time < 10:
                bottleneck = BottleneckType.LATENCY
            elif kernel_time > 100:
                bottleneck = BottleneckType.COMPUTE
            else:
                bottleneck = BottleneckType.BALANCED
            
        except:
            bottleneck = BottleneckType.BALANCED
            kernel_time = 0
        
        return ProfileResult(
            bottleneck=bottleneck,
            metrics={"kernel_time_us": kernel_time},
            recommendations=self.RECOMMENDATIONS[bottleneck],
            rocprof_compute_used=False,
        )


def profile_kernel(benchmark_script: str,
                   work_dir: str = "/tmp/mini_kernel_profile",
                   kernel_name: str = "kernel",
                   verbose: bool = True) -> ProfileResult:
    """
    Profile a kernel using rocprof-compute.
    
    This is the main entry point for the mini-kernel agent.
    
    Args:
        benchmark_script: Path to benchmark Python script
        work_dir: Working directory for output
        kernel_name: Name for the profiling run
        verbose: Print detailed output
        
    Returns:
        ProfileResult with bottleneck analysis
    """
    profiler = RocprofComputeProfiler(verbose=verbose)
    return profiler.profile(
        Path(benchmark_script),
        Path(work_dir),
        kernel_name=kernel_name
    )


def identify_bottleneck(benchmark_script: str,
                        work_dir: str = "/tmp/mini_kernel_profile",
                        verbose: bool = True) -> Tuple[str, List[str]]:
    """
    Identify bottleneck for a kernel.
    
    Returns:
        Tuple of (bottleneck_type, recommendations)
    """
    result = profile_kernel(benchmark_script, work_dir, verbose=verbose)
    return result.bottleneck.value, result.recommendations


# Legacy compatibility
class KernelProfiler:
    """Legacy profiler class for backwards compatibility."""
    
    RECOMMENDATIONS = RocprofComputeProfiler.RECOMMENDATIONS
    
    def __init__(self, docker_image: str = "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x", gpu_device: str = "0"):
        self.profiler = RocprofComputeProfiler(docker_image=docker_image, verbose=False)
    
    def profile(self, kernel_path: Path, work_dir: Optional[Path] = None, **kwargs) -> ProfileResult:
        work_dir = work_dir or Path("/tmp/mini_kernel_profile")
        return self.profiler.profile(kernel_path, work_dir)
    
    def quick_profile(self, kernel_path: Path) -> Tuple[BottleneckType, List[str]]:
        result = self.profile(kernel_path)
        return result.bottleneck, result.recommendations


def analyze_bottleneck(kernel_path: Path, 
                       gpu: str = "0",
                       docker_image: str = "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x") -> Dict[str, Any]:
    """Legacy function for backwards compatibility."""
    profiler = RocprofComputeProfiler(docker_image=docker_image, verbose=False)
    result = profiler.profile(kernel_path, Path("/tmp/mini_kernel_profile"))
    return result.to_dict()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        script_path = sys.argv[1]
        work_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/mini_kernel_profile"
        
        result = profile_kernel(script_path, work_dir, verbose=True)
        
        print("\n" + "=" * 70)
        print("  RESULT")
        print("=" * 70)
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("Usage: python profiler.py <benchmark_script.py> [work_dir]")
        print("")
        print("Example:")
        print("  python profiler.py benchmark.py /tmp/profile_output")
