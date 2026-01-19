#!/usr/bin/env python3
"""
Test the LLM-powered optimization on the add_kernel example.

This script demonstrates the full mini-kernel agent pipeline:
1. Run rocprof-compute profiling
2. Use LLM (Claude) to analyze bottlenecks
3. Use OpenEvolve-style evolution to generate optimizations
4. Evaluate and return best result
"""

import sys
import os
import subprocess
import json
import re
import time
from pathlib import Path

# Add mini_kernel to path
sys.path.insert(0, str(Path(__file__).parent))

# Configuration
DOCKER_IMAGE = "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x"
KERNEL_PATH = Path(__file__).parent / "examples" / "add_kernel" / "kernel.py"


def print_banner(title: str):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def step(num: int, total: int, msg: str):
    print()
    print(f"[{num}/{total}] {msg}")
    print("-" * 50)


def run_in_docker(script: str, work_dir: Path, timeout: int = 300) -> str:
    """Run a Python script in Docker and return output."""
    
    script_path = work_dir / "run_script.py"
    script_path.write_text(script)
    
    cmd = [
        "docker", "run", "--rm",
        "--device=/dev/kfd", "--device=/dev/dri",
        "--group-add", "video",
        "-v", f"{work_dir}:/workspace",
        "-v", f"{KERNEL_PATH.parent}:/kernel",
        "-w", "/workspace",
        DOCKER_IMAGE,
        "python3", "/workspace/run_script.py"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"


def run_profiler(work_dir: Path) -> str:
    """Run rocprof-compute profiler on the kernel."""
    
    script = '''#!/usr/bin/env python3
import subprocess
import sys

# Install rocprof-compute dependencies
print("Installing rocprof-compute dependencies...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "-r", "/opt/rocm/libexec/rocprofiler-compute/requirements.txt"
], capture_output=True)

# Create benchmark script
benchmark = """
import torch
torch.set_default_device("cuda")
import sys
sys.path.insert(0, "/kernel")
from kernel import triton_add

# Create test data
size = 1024 * 1024
x = torch.randn(size, device='cuda', dtype=torch.float32)
y = torch.randn(size, device='cuda', dtype=torch.float32)

# Warmup
for _ in range(10):
    _ = triton_add(x, y)
torch.cuda.synchronize()

# Profile runs
for _ in range(20):
    _ = triton_add(x, y)
torch.cuda.synchronize()
print("Benchmark complete")
"""

with open("/workspace/bench.py", "w") as f:
    f.write(benchmark)

# Run profiling
print("\\nRunning rocprof-compute profile...")
result = subprocess.run([
    "rocprof-compute", "profile",
    "-n", "add_kernel",
    "--path", "/workspace/profile_output",
    "--", "python3", "/workspace/bench.py"
], capture_output=True, text=True)

print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

# Run analysis
print("\\nRunning rocprof-compute analyze...")
result = subprocess.run([
    "rocprof-compute", "analyze",
    "--path", "/workspace/profile_output"
], capture_output=True, text=True)

print(result.stdout[:5000])
'''
    
    return run_in_docker(script, work_dir, timeout=600)


def run_baseline_benchmark(work_dir: Path) -> float:
    """Run baseline benchmark and return latency in microseconds."""
    
    script = '''#!/usr/bin/env python3
import torch
torch.set_default_device("cuda")
import sys
sys.path.insert(0, "/kernel")
from kernel import triton_add

# Create test data
size = 1024 * 1024
x = torch.randn(size, device='cuda', dtype=torch.float32)
y = torch.randn(size, device='cuda', dtype=torch.float32)

# Warmup
for _ in range(100):
    _ = triton_add(x, y)
torch.cuda.synchronize()

# Benchmark
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

iterations = 1000
start.record()
for _ in range(iterations):
    _ = triton_add(x, y)
end.record()
torch.cuda.synchronize()

latency_us = start.elapsed_time(end) * 1000 / iterations  # Convert ms to us
print(f"BASELINE_LATENCY_US:{latency_us:.4f}")
'''
    
    output = run_in_docker(script, work_dir)
    
    match = re.search(r'BASELINE_LATENCY_US:([\d.]+)', output)
    if match:
        return float(match.group(1))
    return 0.0


def test_llm_brain():
    """Test the LLM brain connectivity and analysis."""
    
    print_banner("TESTING LLM BRAIN")
    
    try:
        from mini_kernel.llm_brain import LLMBrain
        
        brain = LLMBrain(verbose=True)
        
        # Test simple call
        print("\n  Testing API connectivity...")
        response = brain._call_llm(
            "Say 'LLM Brain is working!' in exactly those words.",
            temperature=0.0,
            max_tokens=50
        )
        print(f"  Response: {response}")
        
        if "working" in response.lower():
            print("  ✓ LLM Brain is connected and working!")
            return True
        else:
            print("  ⚠ Unexpected response")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_profiler_analysis():
    """Test profiler output analysis with LLM."""
    
    print_banner("TESTING PROFILER ANALYSIS")
    
    try:
        from mini_kernel.llm_brain import LLMBrain
        
        brain = LLMBrain(verbose=True)
        
        # Sample profiler output
        sample_output = """
        ================================================================================
        rocprof-compute analyze
        ================================================================================
        
        Top Kernels:
        ╒═════╤══════════════════════════════════════════╤═════════╤═══════════════════╤════════════════╤═════════════════╤═════════════════╕
        │ idx │                                   Kernel │   Count │   Avg Duration(ns)│   Total Time(%)│       Pct       │    Type         │
        ╞═════╪══════════════════════════════════════════╪═════════╪═══════════════════╪════════════════╪═════════════════╪═════════════════╡
        │   0 │ add_kernel_0d1d2d3d4c                    │      20 │           3840.00 │          98.45 │            VALU │      Triton     │
        ╘═════╧══════════════════════════════════════════╧═════════╧═══════════════════╧════════════════╧═════════════════╧═════════════════╛
        
        Hardware Metrics:
        - VALU Utilization: 8.5%
        - VMEM Utilization: 62.3%
        - Wavefront Occupancy: 75.2%
        - Active CUs: 110/110
        """
        
        print("  Analyzing profiler output with LLM...")
        analysis = brain.analyze_profiler_output(sample_output)
        
        print(f"\n  Analysis Results:")
        print(f"    Bottleneck: {analysis.get('primary_bottleneck', 'unknown')}")
        print(f"    Confidence: {analysis.get('confidence', 0):.0%}")
        print(f"    Analysis: {analysis.get('analysis', 'N/A')}")
        print(f"    Recommendations:")
        for i, rec in enumerate(analysis.get('top_3_recommendations', []), 1):
            print(f"      {i}. {rec}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimization_generation():
    """Test optimization code generation with LLM."""
    
    print_banner("TESTING OPTIMIZATION GENERATION")
    
    try:
        from mini_kernel.llm_brain import LLMBrain
        
        brain = LLMBrain(verbose=True)
        
        # Read kernel code
        kernel_code = KERNEL_PATH.read_text()
        
        print("  Generating optimization for LATENCY bottleneck...")
        
        candidate = brain.generate_optimization(
            kernel_code=kernel_code,
            bottleneck="latency",
            profiler_metrics={
                "kernel_time_us": 3.84,
                "valu_utilization": 8.5,
                "vmem_utilization": 62.3,
                "occupancy": 75.2
            }
        )
        
        print(f"\n  Generated Optimization:")
        print(f"    Strategy: {candidate.strategy_name}")
        print(f"    Description: {candidate.description}")
        print(f"    Expected: {candidate.expected_improvement}")
        print(f"    Code length: {len(candidate.code)} chars")
        print(f"\n  Code preview (first 500 chars):")
        print("  " + "-" * 40)
        for line in candidate.code[:500].split('\n'):
            print(f"  {line}")
        print("  " + "-" * 40)
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the full optimization pipeline."""
    
    print_banner("FULL PIPELINE TEST")
    
    work_dir = Path("/tmp/mini_kernel_test")
    work_dir.mkdir(exist_ok=True)
    
    # Step 1: Run profiler
    step(1, 4, "Running rocprof-compute profiler...")
    profiler_output = run_profiler(work_dir)
    print(profiler_output[:2000])
    
    # Step 2: Get baseline
    step(2, 4, "Measuring baseline performance...")
    baseline_us = run_baseline_benchmark(work_dir)
    print(f"  Baseline latency: {baseline_us:.2f} μs")
    
    # Step 3: Analyze with LLM
    step(3, 4, "Analyzing with LLM...")
    try:
        from mini_kernel.llm_brain import LLMBrain
        brain = LLMBrain(verbose=True)
        
        analysis = brain.analyze_profiler_output(profiler_output)
        bottleneck = analysis.get("primary_bottleneck", "balanced")
        print(f"  Identified bottleneck: {bottleneck}")
        print(f"  Recommendations: {analysis.get('top_3_recommendations', [])}")
    except Exception as e:
        print(f"  ⚠ LLM analysis failed: {e}")
        bottleneck = "latency"  # Default assumption for small kernels
    
    # Step 4: Generate optimization
    step(4, 4, "Generating optimization with LLM...")
    try:
        kernel_code = KERNEL_PATH.read_text()
        candidate = brain.generate_optimization(
            kernel_code=kernel_code,
            bottleneck=bottleneck,
            profiler_metrics=analysis.get("key_metrics", {})
        )
        
        print(f"  Strategy: {candidate.strategy_name}")
        print(f"  Description: {candidate.description}")
        
        # Save generated code
        opt_path = work_dir / "optimized_kernel.py"
        opt_path.write_text(candidate.code)
        print(f"  Saved to: {opt_path}")
        
    except Exception as e:
        print(f"  ⚠ Optimization generation failed: {e}")
    
    # Summary
    print_banner("PIPELINE COMPLETE")
    print(f"  Kernel: {KERNEL_PATH.name}")
    print(f"  Baseline: {baseline_us:.2f} μs")
    print(f"  Bottleneck: {bottleneck}")
    print(f"  Work dir: {work_dir}")


def main():
    """Main test runner."""
    
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  Mini-Kernel Agent - LLM Integration Test".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print(f"  Kernel: {KERNEL_PATH}")
    print(f"  Docker: {DOCKER_IMAGE}")
    print()
    
    # Check if kernel exists
    if not KERNEL_PATH.exists():
        print(f"ERROR: Kernel not found at {KERNEL_PATH}")
        return 1
    
    # Test 1: LLM Brain connectivity
    if not test_llm_brain():
        print("\n⚠ LLM Brain test failed. Check API credentials.")
        # Continue anyway to test other components
    
    # Test 2: Profiler analysis
    test_profiler_analysis()
    
    # Test 3: Optimization generation
    test_optimization_generation()
    
    # Test 4: Full pipeline (optional, slower)
    print("\n" + "=" * 70)
    response = input("Run full pipeline test with Docker? (y/N): ").strip().lower()
    if response == 'y':
        test_full_pipeline()
    
    print("\n" + "=" * 70)
    print("  TEST COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

