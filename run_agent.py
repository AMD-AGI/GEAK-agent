#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  MINI-KERNEL AGENT - Autonomous GPU Kernel Optimization
═══════════════════════════════════════════════════════════════════════════════

This is the MAIN ENTRY POINT for all kernel optimization use cases:

  1. Single kernel file:     python run_agent.py kernel.py
  2. Module folder:          python run_agent.py /path/to/module/
  3. Multiple kernels:       python run_agent.py /path/to/kernels/

The agent will automatically:
  - Detect all kernels in the given path
  - Run rocprof-compute profiling
  - Analyze bottlenecks with LLM (Claude)
  - Use OpenEvolve to generate and evolve optimizations
  - Evaluate each candidate and find the best solution

Environment Variables:
  MINI_KERNEL_API_KEY   - Required. Your API key for AMD LLM Gateway
  MINI_KERNEL_API_URL   - Optional. Custom API endpoint
  MINI_KERNEL_DOCKER    - Optional. Docker image to use

Usage:
  # Set your API key first
  export MINI_KERNEL_API_KEY="your-api-key-here"
  
  # Single kernel
  python run_agent.py examples/add_kernel/kernel.py
  
  # Module folder
  python run_agent.py /path/to/mla_module/
  
  # With options
  python run_agent.py kernel.py --generations 5 --population 5 --target 2.0

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import subprocess
import json
import re
import time
import tempfile
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Add mini_kernel to path
sys.path.insert(0, str(Path(__file__).parent))

# =============================================================================
# CONFIGURATION
# =============================================================================

DOCKER_IMAGE = os.environ.get("MINI_KERNEL_DOCKER", "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x")

# Patterns to identify kernel files
KERNEL_PATTERNS = [
    "**/kernel.py",
    "**/kernel_*.py",
    "**/*_kernel.py",
    "**/triton_*.py",
    "**/fused_*.py",
]

# Patterns to identify benchmark files
BENCHMARK_PATTERNS = [
    "benchmark.py",
    "bench.py",
    "test.py",
    "run.py",
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def banner(title: str, char: str = "═"):
    """Print a banner."""
    width = 70
    print()
    print("╔" + char * (width - 2) + "╗")
    print("║" + f"  {title}".ljust(width - 2) + "║")
    print("╚" + char * (width - 2) + "╝")
    print()


def step_header(num: int, total: int, title: str):
    """Print step header."""
    print()
    print("=" * 70)
    print(f"  [{num}/{total}] {title}")
    print("=" * 70)


def log(msg: str, indent: int = 0):
    """Print a log message."""
    prefix = "  " * indent
    print(f"{prefix}{msg}")


# =============================================================================
# KERNEL DETECTION
# =============================================================================

class KernelDetector:
    """Detects and analyzes kernels in a given path."""
    
    def __init__(self, path: Path):
        self.path = path
        self.kernels: List[Dict[str, Any]] = []
        self.benchmark_file: Optional[Path] = None
        self.is_single_file = path.is_file()
        self.is_module = path.is_dir()
        
    def detect(self) -> List[Dict[str, Any]]:
        """Detect all kernels in the path."""
        
        log(f"Scanning: {self.path}")
        
        if self.is_single_file:
            # Single kernel file
            self.kernels = [self._analyze_kernel_file(self.path)]
            self._find_benchmark(self.path.parent)
            
        elif self.is_module:
            # Directory - find all kernels
            self._scan_directory(self.path)
            self._find_benchmark(self.path)
        
        log(f"Found {len(self.kernels)} kernel(s)")
        for k in self.kernels:
            log(f"  • {k['name']} ({k['type']})", indent=1)
        
        if self.benchmark_file:
            log(f"Benchmark: {self.benchmark_file.name}")
        
        return self.kernels
    
    def _scan_directory(self, directory: Path):
        """Scan directory for kernel files."""
        
        # Look for kernel files
        for pattern in KERNEL_PATTERNS:
            for path in directory.glob(pattern):
                if path.is_file() and not path.name.startswith("_"):
                    kernel_info = self._analyze_kernel_file(path)
                    if kernel_info:
                        self.kernels.append(kernel_info)
        
        # If no kernels found with patterns, try all .py files
        if not self.kernels:
            for path in directory.glob("*.py"):
                if path.name not in BENCHMARK_PATTERNS and not path.name.startswith("_"):
                    kernel_info = self._analyze_kernel_file(path)
                    if kernel_info and kernel_info.get("has_kernel"):
                        self.kernels.append(kernel_info)
    
    def _analyze_kernel_file(self, path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a single kernel file."""
        
        try:
            content = path.read_text()
            
            # Detect kernel type
            kernel_type = "unknown"
            if "@triton.jit" in content or "triton.language" in content:
                kernel_type = "triton"
            elif "__global__" in content or "hipLaunch" in content:
                kernel_type = "hip"
            elif "torch.autograd.Function" in content:
                kernel_type = "pytorch"
            elif "ck::" in content or "composable_kernel" in content:
                kernel_type = "ck"
            
            # Find kernel functions
            kernel_funcs = []
            
            # Triton kernels
            triton_matches = re.findall(r'@triton\.jit\s*\ndef\s+(\w+)', content)
            kernel_funcs.extend(triton_matches)
            
            # CUDA/HIP kernels
            hip_matches = re.findall(r'__global__\s+\w+\s+(\w+)\s*\(', content)
            kernel_funcs.extend(hip_matches)
            
            # Check for run/benchmark functions
            has_benchmark = any(f"def {name}" in content for name in ["run_baseline", "benchmark", "bench_op", "main"])
            
            return {
                "path": path,
                "name": path.stem,
                "type": kernel_type,
                "kernel_functions": kernel_funcs,
                "has_kernel": len(kernel_funcs) > 0 or kernel_type != "unknown",
                "has_benchmark": has_benchmark,
                "content": content,
            }
            
        except Exception as e:
            log(f"Warning: Could not analyze {path}: {e}")
            return None
    
    def _find_benchmark(self, directory: Path):
        """Find benchmark file in directory."""
        for name in BENCHMARK_PATTERNS:
            bench_path = directory / name
            if bench_path.exists():
                self.benchmark_file = bench_path
                break


# =============================================================================
# DOCKER EXECUTION
# =============================================================================

class DockerRunner:
    """Runs scripts in Docker with live output."""
    
    def __init__(self, work_dir: Path, kernel_dir: Path, image: str = DOCKER_IMAGE):
        self.work_dir = work_dir
        self.kernel_dir = kernel_dir
        self.image = image
        
    def run(self, script: str, timeout: int = 600, show_output: bool = True) -> str:
        """Run a Python script in Docker."""
        
        script_path = self.work_dir / "docker_script.py"
        script_path.write_text(script)
        
        cmd = [
            "docker", "run", "--rm",
            "--device=/dev/kfd", "--device=/dev/dri",
            "--group-add", "video",
            "-v", f"{self.work_dir}:/workspace",
            "-v", f"{self.kernel_dir}:/kernel",
            "-w", "/workspace",
            self.image,
            "python3", "-u", "/workspace/docker_script.py"
        ]
        
        try:
            if show_output:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                output_lines = []
                for line in process.stdout:
                    print(f"  {line}", end="")
                    output_lines.append(line)
                
                process.wait(timeout=timeout)
                return "".join(output_lines)
            else:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                return result.stdout + result.stderr
                
        except subprocess.TimeoutExpired:
            return "TIMEOUT"
        except Exception as e:
            return f"ERROR: {e}"


# =============================================================================
# PROFILER
# =============================================================================

class Profiler:
    """Runs rocprof-compute profiling."""
    
    def __init__(self, docker: DockerRunner):
        self.docker = docker
        
    def profile(self, kernel_info: Dict[str, Any]) -> Dict[str, Any]:
        """Profile a kernel and return analysis."""
        
        log("Starting rocprof-compute profiler...")
        
        # Generate benchmark script
        benchmark_script = self._generate_benchmark(kernel_info)
        
        script = f'''#!/usr/bin/env python3
import subprocess
import sys
import os
import shutil

print()
print("=" * 60)
print("  ROCPROF-COMPUTE PROFILER")
print("=" * 60)
print()

# Install dependencies
print("[1/4] Installing dependencies (this may take 1-2 minutes)...")
print("  Installing rocprof-compute requirements...")
dep_result = subprocess.run([
    sys.executable, "-m", "pip", "install",
    "-r", "/opt/rocm/libexec/rocprofiler-compute/requirements.txt"
], capture_output=True, text=True)
# Show any errors
if dep_result.returncode != 0:
    print("  Warning: " + dep_result.stderr[-500:])
print("  ✓ Dependencies installed")

# Create benchmark
print()
print("[2/4] Creating benchmark...")
benchmark = """{benchmark_script}"""
with open("/workspace/bench.py", "w") as f:
    f.write(benchmark)
print("  ✓ Benchmark created")

# Clean old output
if os.path.exists("/workspace/profile_output"):
    shutil.rmtree("/workspace/profile_output")

# Run profiling
print()
print("[3/4] Running rocprof-compute profile...")
print("  (This may take 1-2 minutes)")
print()

import sys as _sys
process = subprocess.Popen([
    "rocprof-compute", "profile",
    "-n", "kernel_profile",
    "--path", "/workspace/profile_output",
    "--", "python3", "/workspace/bench.py"
], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

# Stream output in real-time
stdout_lines = []
for line in process.stdout:
    stdout_lines.append(line)
    # Show progress indicators
    if "%" in line or "Peak" in line or "roofline" in line.lower() or "Collecting" in line or "profiling" in line.lower():
        print(f"  {{line.strip()}}")
        _sys.stdout.flush()

process.wait()
result_stdout = "".join(stdout_lines)

print()
print("[4/4] Analyzing results...")
print()

process = subprocess.Popen([
    "rocprof-compute", "analyze",
    "--path", "/workspace/profile_output"
], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

# Stream analyze output
analyze_lines = []
for line in process.stdout:
    analyze_lines.append(line)
    print(line, end="")
    _sys.stdout.flush()

process.wait()
result_stdout = "".join(analyze_lines)

# Save output
with open("/workspace/profiler_output.txt", "w") as f:
    f.write(result_stdout)

print()
print("  ✓ Profiling complete")
'''
        
        output = self.docker.run(script, timeout=600)
        
        # Load saved output
        output_file = self.docker.work_dir / "profiler_output.txt"
        if output_file.exists():
            return {
                "output": output_file.read_text(),
                "success": True
            }
        
        return {"output": output, "success": False}
    
    def _generate_benchmark(self, kernel_info: Dict[str, Any]) -> str:
        """Generate a benchmark script for the kernel."""
        
        kernel_name = kernel_info["name"]
        kernel_type = kernel_info["type"]
        
        # If kernel has its own benchmark, use it
        if kernel_info.get("has_benchmark"):
            return f'''
import torch
torch.set_default_device("cuda")
import sys
sys.path.insert(0, "/kernel")
from {kernel_name} import *

# Try to find and run benchmark
if 'run_baseline' in dir():
    for _ in range(10): run_baseline()
    torch.cuda.synchronize()
    for _ in range(20): run_baseline()
    torch.cuda.synchronize()
elif 'benchmark' in dir():
    benchmark()
elif 'triton_op' in dir():
    x = torch.randn(1024*1024, device='cuda')
    y = torch.randn(1024*1024, device='cuda')
    for _ in range(10): triton_op(x, y)
    torch.cuda.synchronize()
    for _ in range(20): triton_op(x, y)
    torch.cuda.synchronize()
print("Benchmark complete")
'''
        
        # Generate generic benchmark based on kernel type
        if kernel_type == "triton":
            return f'''
import torch
torch.set_default_device("cuda")
import sys
sys.path.insert(0, "/kernel")
from {kernel_name} import *

# Find triton op
triton_fn = None
for name in dir():
    obj = eval(name)
    if callable(obj) and not name.startswith('_'):
        if 'triton' in str(type(obj)).lower() or name.endswith('_op'):
            triton_fn = obj
            break

if triton_fn:
    # Generate test inputs
    x = torch.randn(1024*1024, device='cuda', dtype=torch.float32)
    y = torch.randn(1024*1024, device='cuda', dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        try:
            triton_fn(x, y)
        except:
            triton_fn(x)
    torch.cuda.synchronize()
    
    # Profile
    for _ in range(20):
        try:
            triton_fn(x, y)
        except:
            triton_fn(x)
    torch.cuda.synchronize()

print("Benchmark complete")
'''
        
        # Default fallback
        return f'''
import torch
torch.set_default_device("cuda")
import sys
sys.path.insert(0, "/kernel")

# Import kernel module
try:
    from {kernel_name} import *
except Exception as e:
    print(f"Import error: {{e}}")

print("Benchmark complete")
'''


# =============================================================================
# BASELINE BENCHMARK
# =============================================================================

class Benchmarker:
    """Measures baseline performance."""
    
    def __init__(self, docker: DockerRunner):
        self.docker = docker
        
    def measure_baseline(self, kernel_info: Dict[str, Any]) -> float:
        """Measure baseline latency in microseconds."""
        
        kernel_name = kernel_info["name"]
        
        script = f'''#!/usr/bin/env python3
import torch
torch.set_default_device("cuda")
import sys
sys.path.insert(0, "/kernel")

print("  Loading kernel...")
from {kernel_name} import *

print("  Creating test data...")

# Find the main function to benchmark
benchmark_fn = None
test_args = None

if 'run_baseline' in dir():
    benchmark_fn = run_baseline
    test_args = ()
elif 'triton_op' in dir():
    benchmark_fn = triton_op
    size = 1024 * 1024
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    y = torch.randn(size, device='cuda', dtype=torch.float32)
    test_args = (x, y)
elif 'benchmark' in dir():
    benchmark_fn = benchmark
    test_args = ()

if benchmark_fn is None:
    print("  ✗ No benchmark function found")
    print("BASELINE_LATENCY_US:0.0")
    sys.exit(1)

print("  Warming up...")
for _ in range(100):
    benchmark_fn(*test_args) if test_args else benchmark_fn()
torch.cuda.synchronize()

print("  Benchmarking (1000 iterations)...")
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

iterations = 1000
start.record()
for _ in range(iterations):
    benchmark_fn(*test_args) if test_args else benchmark_fn()
end.record()
torch.cuda.synchronize()

latency_us = start.elapsed_time(end) * 1000 / iterations
print()
print(f"  ✓ Baseline latency: {{latency_us:.2f}} μs")
print(f"BASELINE_LATENCY_US:{{latency_us:.4f}}")
'''
        
        output = self.docker.run(script, show_output=True)
        
        match = re.search(r'BASELINE_LATENCY_US:([\d.]+)', output)
        if match:
            return float(match.group(1))
        return 0.0


# =============================================================================
# MAIN AGENT
# =============================================================================

class MiniKernelAgent:
    """
    The main autonomous kernel optimization agent.
    
    Orchestrates:
    1. Kernel detection
    2. Profiling
    3. LLM analysis
    4. OpenEvolve optimization
    5. Results reporting
    """
    
    def __init__(self, 
                 path: Path,
                 generations: int = 3,
                 population: int = 3,
                 target_speedup: float = 1.5,
                 verbose: bool = True):
        
        self.path = path
        self.generations = generations
        self.population = population
        self.target_speedup = target_speedup
        self.verbose = verbose
        
        # Setup work directory
        self.work_dir = Path(tempfile.mkdtemp(prefix="mini_kernel_"))
        
        # Determine kernel directory
        if path.is_file():
            self.kernel_dir = path.parent
        else:
            self.kernel_dir = path
        
        # Initialize components
        self.docker = DockerRunner(self.work_dir, self.kernel_dir)
        self.detector = KernelDetector(path)
        self.profiler = Profiler(self.docker)
        self.benchmarker = Benchmarker(self.docker)
        
        # Results
        self.results: Dict[str, Any] = {}
        
    def run(self) -> Dict[str, Any]:
        """Run the full optimization pipeline."""
        
        # Check API key
        api_key = os.environ.get("MINI_KERNEL_API_KEY")
        if not api_key:
            banner("⚠️  ERROR: API KEY NOT SET")
            log("Please set your API key:")
            log("  export MINI_KERNEL_API_KEY='your-api-key-here'")
            log("")
            return {"error": "API key not set"}
        
        # Banner
        banner("🚀 MINI-KERNEL AGENT")
        
        log(f"Path:        {self.path}")
        log(f"Docker:      {DOCKER_IMAGE}")
        log(f"Work dir:    {self.work_dir}")
        log(f"API Key:     {api_key[:8]}...{api_key[-4:]}")
        log(f"Generations: {self.generations}")
        log(f"Population:  {self.population}")
        log(f"Target:      {self.target_speedup:.1f}x speedup")
        
        # =====================================================================
        # STEP 1: Detect kernels
        # =====================================================================
        step_header(1, 5, "DETECTING KERNELS")
        
        kernels = self.detector.detect()
        
        if not kernels:
            log("✗ No kernels found!")
            return {"error": "No kernels found"}
        
        # For now, optimize the first/main kernel
        # Future: optimize all kernels
        kernel_info = kernels[0]
        
        # =====================================================================
        # STEP 2: Profile with rocprof-compute
        # =====================================================================
        step_header(2, 5, "PROFILING WITH ROCPROF-COMPUTE")
        
        profile_result = self.profiler.profile(kernel_info)
        profiler_output = profile_result.get("output", "")
        
        # =====================================================================
        # STEP 3: Measure baseline
        # =====================================================================
        step_header(3, 5, "MEASURING BASELINE PERFORMANCE")
        
        baseline_us = self.benchmarker.measure_baseline(kernel_info)
        
        if baseline_us == 0:
            log("✗ Failed to measure baseline!")
            return {"error": "Baseline measurement failed"}
        
        # =====================================================================
        # STEP 4: LLM Analysis + OpenEvolve
        # =====================================================================
        step_header(4, 5, "LLM ANALYSIS + OPENEVOLVE OPTIMIZATION")
        
        try:
            from mini_kernel.llm_brain import LLMBrain, OpenEvolveBrain
            
            # Read kernel code
            kernel_code = kernel_info["content"]
            
            # Initialize LLM
            log("")
            llm = LLMBrain(verbose=True)
            
            # Analyze bottleneck
            log("")
            analysis = llm.analyze_profiler_output(profiler_output, kernel_code)
            
            bottleneck = analysis.get("primary_bottleneck", "balanced")
            metrics = analysis.get("key_metrics", {})
            
            # Create evaluator
            evaluate_fn = self._create_evaluator(kernel_info, baseline_us)
            
            # Run OpenEvolve
            log("")
            evolver = OpenEvolveBrain(
                population_size=self.population,
                generations=self.generations,
                mutation_rate=0.3,
                crossover_rate=0.2,
                verbose=True
            )
            
            best = evolver.evolve(
                kernel_code=kernel_code,
                bottleneck=bottleneck,
                profiler_metrics=metrics,
                evaluate_fn=evaluate_fn,
                baseline_time_us=baseline_us,
                target_fitness=self.target_speedup
            )
            
            # Store results
            self.results = {
                "kernel": kernel_info["name"],
                "baseline_us": baseline_us,
                "best_time_us": baseline_us / best.fitness if best.fitness > 0 else baseline_us,
                "speedup": best.fitness,
                "strategy": best.strategy_name,
                "generation": best.generation,
                "bottleneck": bottleneck,
                "target_achieved": best.fitness >= self.target_speedup,
                "history": evolver.get_evolution_history(),
            }
            
        except ImportError as e:
            log(f"✗ Import error: {e}")
            log("Make sure to install: pip install openai tenacity")
            return {"error": str(e)}
        except Exception as e:
            log(f"✗ Optimization error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
        
        # =====================================================================
        # STEP 5: Final Report
        # =====================================================================
        step_header(5, 5, "FINAL REPORT")
        
        self._print_report(best, baseline_us)
        
        # Save results
        self._save_results(best)
        
        return self.results
    
    def _create_evaluator(self, kernel_info: Dict[str, Any], baseline_us: float):
        """Create an evaluation function."""
        
        kernel_name = kernel_info["name"]
        
        def evaluate(code: str) -> float:
            """Evaluate optimization code and return speedup."""
            
            # Save the code
            opt_path = self.work_dir / "optimization.py"
            opt_path.write_text(code)
            
            # Create evaluation script
            eval_script = f'''#!/usr/bin/env python3
import torch
import sys
import traceback

torch.set_default_device("cuda")
sys.path.insert(0, "/kernel")

try:
    # First import the original kernel
    from {kernel_name} import *
    
    # Then load and run the optimization
    exec(open("/workspace/optimization.py").read())
    
    # Try to find and run the optimization
    for cls_name in ['OptimizedKernel', 'Optimizer', 'HIPGraphOptimizer', 'MultiStreamOptimizer']:
        if cls_name in dir():
            cls = eval(cls_name)
            obj = cls()
            if hasattr(obj, 'optimize'):
                latency = obj.optimize()
                speedup = {baseline_us} / latency if latency > 0 else 0
                print(f"SPEEDUP:{{speedup:.4f}}")
                sys.exit(0)
            elif hasattr(obj, 'benchmark'):
                latency = obj.benchmark()
                speedup = {baseline_us} / latency if latency > 0 else 0
                print(f"SPEEDUP:{{speedup:.4f}}")
                sys.exit(0)
    
    # Fallback
    print("SPEEDUP:1.0")
    
except Exception as e:
    print(f"EVAL_ERROR:{{e}}")
    traceback.print_exc()
    print("SPEEDUP:0.0")
'''
            
            eval_path = self.work_dir / "eval_script.py"
            eval_path.write_text(eval_script)
            
            # Run evaluation
            output = self.docker.run(
                open(eval_path).read(),
                timeout=120,
                show_output=False
            )
            
            match = re.search(r'SPEEDUP:([\d.]+)', output)
            if match:
                return float(match.group(1))
            return 0.0
        
        return evaluate
    
    def _print_report(self, best, baseline_us: float):
        """Print the final report."""
        
        log(f"  Kernel:          {self.results['kernel']}")
        log(f"  Bottleneck:      {self.results['bottleneck']}")
        log(f"  Baseline:        {baseline_us:.2f} μs")
        log(f"  Best:            {self.results['best_time_us']:.2f} μs")
        log(f"  Speedup:         {self.results['speedup']:.2f}x")
        log(f"  Strategy:        {self.results['strategy']}")
        log(f"  Generation:      {self.results['generation']}")
        log("")
        
        if self.results['target_achieved']:
            log(f"  ✅ TARGET ACHIEVED! ({self.target_speedup:.1f}x)")
        else:
            gap = (self.target_speedup - self.results['speedup']) / self.target_speedup * 100
            log(f"  ⚠️  Gap to target: {gap:.1f}%")
    
    def _save_results(self, best):
        """Save optimization results."""
        
        # Save best solution
        output_file = self.work_dir / "best_optimization.py"
        output_file.write_text(best.code)
        log(f"")
        log(f"  Best solution: {output_file}")
        
        # Save results JSON
        results_file = self.work_dir / "results.json"
        results_file.write_text(json.dumps(self.results, indent=2, default=str))
        log(f"  Results JSON:  {results_file}")
        
        # Save history
        history_file = self.work_dir / "evolution_history.json"
        history_file.write_text(json.dumps(self.results.get('history', []), indent=2))
        log(f"  History:       {history_file}")
        
        log("")
        log("=" * 70)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Mini-Kernel Agent - Autonomous GPU Kernel Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single kernel file
  python run_agent.py examples/add_kernel/kernel.py
  
  # Module folder
  python run_agent.py /path/to/mla_module/
  
  # With options
  python run_agent.py kernel.py --generations 5 --population 5 --target 2.0

Environment Variables:
  MINI_KERNEL_API_KEY   - Required. Your API key for AMD LLM Gateway
  MINI_KERNEL_API_URL   - Optional. Custom API endpoint
  MINI_KERNEL_DOCKER    - Optional. Docker image to use
"""
    )
    
    parser.add_argument(
        "path",
        type=str,
        help="Path to kernel file or module directory"
    )
    
    parser.add_argument(
        "--generations", "-g",
        type=int,
        default=3,
        help="Number of evolution generations (default: 3)"
    )
    
    parser.add_argument(
        "--population", "-p",
        type=int,
        default=3,
        help="Population size per generation (default: 3)"
    )
    
    parser.add_argument(
        "--target", "-t",
        type=float,
        default=1.5,
        help="Target speedup (default: 1.5)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    
    args = parse_args()
    
    # Resolve path
    path = Path(args.path).resolve()
    
    if not path.exists():
        print(f"Error: Path not found: {path}")
        return 1
    
    # Create and run agent
    agent = MiniKernelAgent(
        path=path,
        generations=args.generations,
        population=args.population,
        target_speedup=args.target,
        verbose=not args.quiet
    )
    
    result = agent.run()
    
    if "error" in result:
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
