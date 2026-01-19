#!/usr/bin/env python3
"""
OpenEvolve Integration for ROCm Profiler.

This module provides integration between the ROCm profiler and OpenEvolve,
allowing the evolutionary optimization framework to use profiling data
to guide kernel optimization.
"""

import json
import os
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass


@dataclass
class ProfilingFitness:
    """Fitness metrics from profiling for evolutionary optimization."""
    latency_us: float
    throughput_tflops: float
    memory_bandwidth_gbps: float
    efficiency: float  # 0.0 to 1.0
    
    # Penalty factors
    correctness_penalty: float = 0.0  # 0 if correct, large if incorrect
    
    # Combined fitness score (higher is better)
    fitness_score: float = 0.0
    
    def __post_init__(self):
        # Calculate combined fitness
        # Lower latency is better, so invert
        latency_score = 1.0 / (self.latency_us + 1e-6)
        
        # Higher throughput/efficiency is better
        perf_score = self.throughput_tflops * self.efficiency
        
        # Apply correctness penalty
        self.fitness_score = (latency_score * 1e6 + perf_score) * (1.0 - self.correctness_penalty)


class OpenEvolveProfilerIntegration:
    """
    Integration layer between OpenEvolve and ROCm Profiler.
    
    Usage with OpenEvolve:
        
        integration = OpenEvolveProfilerIntegration(
            baseline_kernel=baseline_kernel_code,
            test_inputs=test_inputs,
            gpu_id=0
        )
        
        # In OpenEvolve evaluator:
        def evaluate(candidate_kernel):
            fitness = integration.evaluate_kernel(candidate_kernel)
            return fitness.fitness_score
    """
    
    def __init__(
        self,
        baseline_kernel: str,
        test_inputs: Dict[str, Any],
        gpu_id: int = 0,
        docker_image: str = "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x",
        work_dir: Optional[str] = None,
        correctness_tolerance: float = 1e-5,
    ):
        """
        Initialize the integration.
        
        Args:
            baseline_kernel: Baseline kernel code (for correctness comparison)
            test_inputs: Dictionary of test input tensors/values
            gpu_id: GPU device ID
            docker_image: Docker image with ROCm
            work_dir: Working directory
            correctness_tolerance: Tolerance for output comparison
        """
        self.baseline_kernel = baseline_kernel
        self.test_inputs = test_inputs
        self.gpu_id = gpu_id
        self.docker_image = docker_image
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="openevolve_profiler_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.correctness_tolerance = correctness_tolerance
        
        # Cache baseline outputs
        self.baseline_outputs = None
        self.baseline_latency = None
        
    def evaluate_kernel(
        self,
        kernel_code: str,
        num_warmup: int = 100,
        num_iterations: int = 500,
    ) -> ProfilingFitness:
        """
        Evaluate a candidate kernel.
        
        Args:
            kernel_code: Kernel code to evaluate
            num_warmup: Warmup iterations
            num_iterations: Benchmark iterations
            
        Returns:
            ProfilingFitness with metrics
        """
        # Create evaluation script
        script = self._create_evaluation_script(
            kernel_code, num_warmup, num_iterations
        )
        
        # Write and run
        script_path = self.work_dir / "evaluate.py"
        script_path.write_text(script)
        
        results_path = self.work_dir / "results.json"
        
        cmd = [
            "docker", "run", "--rm",
            "--device=/dev/kfd", "--device=/dev/dri", "--ipc=host",
            "-e", f"HIP_VISIBLE_DEVICES={self.gpu_id}",
            "-v", f"{self.work_dir}:/workspace",
            self.docker_image,
            "python3", "/workspace/evaluate.py"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if results_path.exists():
                with open(results_path) as f:
                    data = json.load(f)
                    
                return ProfilingFitness(
                    latency_us=data.get("latency_us", float('inf')),
                    throughput_tflops=data.get("throughput_tflops", 0.0),
                    memory_bandwidth_gbps=data.get("bandwidth_gbps", 0.0),
                    efficiency=data.get("efficiency", 0.0),
                    correctness_penalty=data.get("correctness_penalty", 1.0),
                )
            else:
                # Execution failed
                return ProfilingFitness(
                    latency_us=float('inf'),
                    throughput_tflops=0.0,
                    memory_bandwidth_gbps=0.0,
                    efficiency=0.0,
                    correctness_penalty=1.0,
                )
                
        except subprocess.TimeoutExpired:
            return ProfilingFitness(
                latency_us=float('inf'),
                throughput_tflops=0.0,
                memory_bandwidth_gbps=0.0,
                efficiency=0.0,
                correctness_penalty=1.0,
            )
        except Exception as e:
            print(f"Evaluation error: {e}")
            return ProfilingFitness(
                latency_us=float('inf'),
                throughput_tflops=0.0,
                memory_bandwidth_gbps=0.0,
                efficiency=0.0,
                correctness_penalty=1.0,
            )
    
    def _create_evaluation_script(
        self,
        kernel_code: str,
        num_warmup: int,
        num_iterations: int,
    ) -> str:
        """Create the evaluation script."""
        
        script = f'''#!/usr/bin/env python3
"""Auto-generated kernel evaluation script."""
import torch
import numpy as np
import json
import traceback
import os
os.environ['TRITON_PRINT_AUTOTUNING'] = '0'

device = "cuda"
results = {{
    "latency_us": float('inf'),
    "throughput_tflops": 0.0,
    "bandwidth_gbps": 0.0,
    "efficiency": 0.0,
    "correctness_penalty": 1.0,
    "error": None,
}}

try:
    # Kernel code
{self._indent_code(kernel_code, 4)}

    # Test inputs (would be injected from test_inputs dict)
    # This is a placeholder - actual implementation would serialize/deserialize inputs
    
    # Warmup
    for _ in range({num_warmup}):
        pass  # Run kernel here
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range({num_iterations}):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        pass  # Run kernel here
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # us
    
    results["latency_us"] = np.mean(times)
    
    # TODO: Add correctness check against baseline
    results["correctness_penalty"] = 0.0  # Assume correct for now
    
    # TODO: Calculate throughput from actual FLOP count
    results["efficiency"] = 1.0 - results["correctness_penalty"]
    
except Exception as e:
    results["error"] = str(e)
    results["traceback"] = traceback.format_exc()

# Save results
with open("/workspace/results.json", "w") as f:
    json.dump(results, f, indent=2)
'''
        return script
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces."""
        indent = " " * spaces
        return "\n".join(indent + line for line in code.split("\n"))
    
    def get_baseline_fitness(self) -> ProfilingFitness:
        """Get fitness of baseline kernel."""
        return self.evaluate_kernel(self.baseline_kernel)


class ProfilerEvaluator:
    """
    Evaluator class for use with OpenEvolve's evaluation pipeline.
    
    This can be used as a custom evaluator in OpenEvolve configuration.
    """
    
    def __init__(
        self,
        baseline_code: str,
        correctness_fn: Optional[Callable] = None,
        gpu_id: int = 0,
        docker_image: str = "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x",
    ):
        """
        Initialize evaluator.
        
        Args:
            baseline_code: Baseline kernel for comparison
            correctness_fn: Optional function to verify correctness
            gpu_id: GPU device ID
            docker_image: Docker image to use
        """
        self.baseline_code = baseline_code
        self.correctness_fn = correctness_fn
        self.gpu_id = gpu_id
        self.docker_image = docker_image
        self.work_dir = Path(tempfile.mkdtemp(prefix="profiler_eval_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate(self, candidate: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a candidate kernel.
        
        This method is called by OpenEvolve during evaluation.
        
        Args:
            candidate: Dictionary containing 'code' key with kernel code
            
        Returns:
            Dictionary of metric scores
        """
        kernel_code = candidate.get('code', '')
        
        # Create and run evaluation
        integration = OpenEvolveProfilerIntegration(
            baseline_kernel=self.baseline_code,
            test_inputs={},
            gpu_id=self.gpu_id,
            docker_image=self.docker_image,
            work_dir=str(self.work_dir),
        )
        
        fitness = integration.evaluate_kernel(kernel_code)
        
        return {
            'latency': -fitness.latency_us,  # Negative because lower is better
            'throughput': fitness.throughput_tflops,
            'efficiency': fitness.efficiency,
            'fitness': fitness.fitness_score,
            'correctness': 1.0 - fitness.correctness_penalty,
        }
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)


def create_openevolve_config(
    baseline_kernel: str,
    kernel_name: str = "optimized_kernel",
    population_size: int = 20,
    num_generations: int = 50,
    gpu_id: int = 0,
) -> Dict:
    """
    Create OpenEvolve configuration for kernel optimization.
    
    Args:
        baseline_kernel: Baseline kernel code
        kernel_name: Name for the optimized kernel
        population_size: Evolution population size
        num_generations: Number of generations
        gpu_id: GPU device ID
        
    Returns:
        OpenEvolve configuration dictionary
    """
    config = {
        "program": {
            "path": baseline_kernel,
            "name": kernel_name,
        },
        "evaluation": {
            "type": "profiler",
            "gpu_id": gpu_id,
            "num_warmup": 100,
            "num_iterations": 500,
            "metrics": ["latency", "throughput", "correctness"],
            "weights": {
                "latency": 0.4,
                "throughput": 0.3,
                "correctness": 0.3,
            }
        },
        "evolution": {
            "population_size": population_size,
            "num_generations": num_generations,
            "mutation_rate": 0.3,
            "crossover_rate": 0.7,
            "selection": "tournament",
            "tournament_size": 3,
        },
        "islands": {
            "num_islands": 4,
            "migration_interval": 5,
            "migration_size": 2,
        },
        "llm": {
            "model": "anthropic/claude-sonnet",
            "mutation_prompt_template": """
You are optimizing a GPU kernel for AMD MI300X/MI355X.

Current kernel:
{code}

Performance metrics:
- Latency: {latency_us:.2f} us
- Efficiency: {efficiency:.1%}
- Bottleneck: {bottleneck}

Suggestions from profiler:
{suggestions}

Generate an optimized version that improves performance while maintaining correctness.
Focus on: {focus_area}
""",
        }
    }
    
    return config

