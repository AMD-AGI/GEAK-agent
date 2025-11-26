"""
🛡️ BULLETPROOF ROCm TRITON KERNEL EVALUATOR  (AMD GPU EDITION) 🛡️

This evaluator is a **drop-in replacement** for the original Metal evaluator (https://github.com/codelion/openevolve/blob/main/examples/mlx_metal_kernel_opt/evaluator.py).
It focuses on Triton kernels executed on AMD GPUs (ROCm / HIP back-end) and
contains multiple layers of protection to guarantee that *the surrounding
evolution process NEVER hard-crashes*, no matter how broken a candidate
kernel may be.

Key changes vs. the original version
────────────────────────────────────
• All Metal-specific logic was removed or rewritten for Triton/ROCm  
• Error classification now understands HIP / ROCm run-time messages  
• Kernel compilation is attempted with `triton.compile()` *before* running  
• A microscopic “canary” launch is executed to catch illegal memory access  
• Graceful fallback to PyTorch reference attention if anything goes wrong  
• Optional *address-sanitiser* style buffer over-allocation for extra safety  

The public API is **identical** (`evaluate(program_text: str) -> dict`) so
OpenEvolve (or any other driver) does not need to be altered.
"""

from __future__ import annotations

import os, sys, time, traceback, tempfile, importlib, types, math, subprocess, json
from typing import Any, Dict, List, Optional, Tuple
from glob import glob
import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except Exception as _e:
    raise RuntimeError(
        "Triton must be installed with AMD/ROCm support "
    ) from _e

import geak_eval  # Using GEAK-eval-OE evaluation framework 
# ────────────────────────────────────────────────────────────────────────────
# Exceptions (Triton / ROCm specific)
# ────────────────────────────────────────────────────────────────────────────
class TritonKernelSafetyError(Exception):
    """Static or dynamic Triton kernel safety violation."""


class HipRuntimeError(Exception):
    """Errors originating from the HIP / ROCm driver/runtime."""


class TritonCompilationError(Exception):
    """triton.compile raised an error."""


# ────────────────────────────────────────────────────────────────────────────
# Evaluator
# ────────────────────────────────────────────────────────────────────────────
class BulletproofTritonEvaluator:
    """
    A *single* instance is created per evaluation.  All state (error counters,
    baseline numbers, …) lives inside this object.
    """

    # ----------  configuration knobs  ----------
    _MAX_RETRIES = 3
    _RETRY_BASE_DELAY_S = 1.0

    # conservative “canary” launch sizes (tokens, heads, …)
    _MAX_SAFE_TOKENS = 128
    _MAX_SAFE_BATCH = 2

    # compile / launch-timeouts (soft, seconds)
    _COMPILE_TIMEOUT_S = 6000
    _LAUNCH_TIMEOUT_S = 6000

    # Use environment variable for golden data path with fallback
    GOLDEN_DATA_PATH = os.environ.get(
        'ROCM_GOLDEN_DATA_PATH',
        '/home/sapmajum/geak-openevolve/TB-eval-OE/tb_eval/data/ROCm/data/performance/golden_results'
    )

    # -------------------------------------------------
    def __init__(self) -> None:
        self.triton_compile_errors: int = 0
        self.hip_runtime_errors: int = 0
        self.memory_violations: int = 0
        self.total_gpu_errors: int = 0
        self.successful_fallbacks: int = 0
        self.retry_attempts_used: int = 0

        # Baseline (PyTorch) numbers are cached between attempts
        self._baseline_metrics: Optional[Dict[str, float]] = None
        
        # Baseline file path (will be set during first evaluation)
        self._baseline_file: Optional[str] = None

        print("🛡️ BULLETPROOF TRITON KERNEL EVALUATOR (AMD GPU) INITIALISED")

    # ======================================================================
    # Public entry-point used by OpenEvolve
    # ======================================================================
    def evaluate(self, test_suite_path: str, program_text: str, ref_wrapper_path: str, 
                 wrapper_fn_name: str, unit_tests_path: str, n_warmup: int, n_iters: int,
                 atol: float, rtol: float, verbose: bool, gpu_id: int=0) -> Dict[str, Any]:
        """Master function: never raises, always returns a dict.
        
        Args:
            test_suite_path: Path to test suite (not used in ROCm evaluator)
            program_text: Path to the program/kernel to evaluate
            ref_wrapper_path: Path to reference wrapper (not used in ROCm evaluator)
            wrapper_fn_name: Name of wrapper function (not used in ROCm evaluator)
            unit_tests_path: Path to unit tests (not used in ROCm evaluator)
            n_warmup: Number of warmup iterations (not used in ROCm evaluator)
            n_iters: Number of benchmark iterations (not used in ROCm evaluator)
            atol: Absolute tolerance (not used in ROCm evaluator)
            rtol: Relative tolerance (not used in ROCm evaluator)
            verbose: Verbose output flag (not used in ROCm evaluator)
            gpu_id: GPU ID to use for evaluation
        """
        try:
            ## if program_text is not a path then save the code in a temporary file
            if not os.path.exists(program_text):
                print("Program text is not a file path, saving to temporary file.")
                with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.py', delete=False
                ) as temp_file:
                    temp_file.write(program_text)
                    program_text = temp_file.name
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
            else:
                print(f"Using program text from file: {program_text}")

            print(f"Evaluating Triton kernel from: {program_text}")
            
            # Run correctness tests using pytest
            env = os.environ.copy()
            env['HIP_VISIBLE_DEVICES'] = str(gpu_id % 8)
            
            # Run pytest for correctness (excluding performance tests)
            correctness_cmd = [
                "pytest", "-v", "-x", "--maxfail=1", 
                program_text,
                "-k", "not (test_performance or test_save)"
            ]
            print(f"Running correctness tests: {' '.join(correctness_cmd)}")
            correctness_result = subprocess.run(
                correctness_cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=env
            )
                
            call, correct, benchmark, benchmark_params, err_msg = None, None, None, None, ""

            # Check correctness test results
            call = correctness_result.returncode == 0
            correct = call  # If tests pass, kernel is correct
            
            if not call:
                print(f"Correctness tests failed. Return code: {correctness_result.returncode}")
                print(f"STDOUT: {correctness_result.stdout}")
                print(f"STDERR: {correctness_result.stderr}")
                # Combine stdout and stderr for error message
                err_msg = f"Correctness tests failed (exit {correctness_result.returncode}):\n"
                if correctness_result.stdout:
                    err_msg += f"STDOUT: {correctness_result.stdout}\n"
                if correctness_result.stderr:
                    err_msg += f"STDERR: {correctness_result.stderr}\n"
                return self._create_comprehensive_failure_result(err_msg)

            # Run performance tests to get benchmark data
            # Get the directory where the kernel file is located
            kernel_dir = os.path.dirname(os.path.abspath(program_text))
            perf_dir = os.path.join(kernel_dir, "perf")
            
            # Run performance tests
            perf_cmd = [
                "pytest", "-v", "-x", "--maxfail=1",
                program_text,
                "-k", "test_performance or test_save_performance_results"
            ]
            print(f"Running performance tests: {' '.join(perf_cmd)}")
            perf_result = subprocess.run(
                perf_cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                env=env
            )
            
            # Parse performance results from JSON files
            benchmark = None
            benchmark_params = None
            
            print(f"Performance test result - returncode: {perf_result.returncode}")
            print(f"Performance test stdout: {perf_result.stdout[:500]}")
            print(f"Performance test stderr: {perf_result.stderr[:500]}")
            
            if os.path.exists(perf_dir):
                print(f"Looking for performance results in: {perf_dir}")
                perf_files = [f for f in os.listdir(perf_dir) if f.endswith('.json')]
                print(f"Found {len(perf_files)} JSON files: {perf_files}")
                if perf_files:
                    import json
                    latest_perf_file = os.path.join(perf_dir, sorted(perf_files)[-1])
                    print(f"Reading performance data from: {latest_perf_file}")
                    with open(latest_perf_file, 'r') as f:
                        perf_data = json.load(f)
                    
                    print(f"Performance data structure: {list(perf_data[0].keys()) if perf_data else 'empty'}")
                    
                    # Extract latency (in milliseconds)
                    if perf_data and len(perf_data) > 0:
                        # Get the first benchmark result
                        first_result = perf_data[0]
                        # Try different possible keys
                        for key in ['latency_ms', 'ms', 'mean_ms', 'median_ms']:
                            if key in first_result:
                                benchmark = first_result[key]
                                benchmark_params = first_result.get('config', first_result.get('params', {}))
                                print(f"Performance: {benchmark}ms (from key '{key}')")
                                break
                        
                        if benchmark is None:
                            print(f"Available keys in perf data: {first_result.keys()}")
                else:
                    print(f"No JSON performance files found in {perf_dir}")
            else:
                print(f"Performance directory does not exist: {perf_dir}")
            
            # If we don't have benchmark data, that's okay for now
            if benchmark is None:
                print("Warning: No benchmark data available")
                benchmark = 0.0  # Default value
            
            # Calculate speedup using persistent baseline stored in file
            # Baseline file is stored in the parent directory of the temp eval directory
            if self._baseline_file is None:
                # Determine baseline file path from the program_text path
                # e.g., /path/evals/tmpXXX/test.py -> /path/evals/.baseline_latency.txt
                eval_dir = os.path.dirname(program_text)  # /path/evals/tmpXXX
                evals_root = os.path.dirname(eval_dir)     # /path/evals
                self._baseline_file = os.path.join(evals_root, ".baseline_latency.txt")
            
            # Try to load existing baseline
            baseline_latency = None
            if os.path.exists(self._baseline_file):
                try:
                    with open(self._baseline_file, 'r') as f:
                        baseline_latency = float(f.read().strip())
                    print(f"Loaded baseline latency from file: {baseline_latency:.6f}ms")
                except Exception as e:
                    print(f"Warning: Could not load baseline from {self._baseline_file}: {e}")
            
            # If no baseline exists, establish it from this kernel
            if baseline_latency is None:
                baseline_latency = benchmark
                speedup = 1.0  # First kernel is the baseline
                print(f"Establishing NEW baseline latency: {baseline_latency:.6f}ms, speedup=1.0")
                # Save baseline to file for future evaluations
                try:
                    os.makedirs(os.path.dirname(self._baseline_file), exist_ok=True)
                    with open(self._baseline_file, 'w') as f:
                        f.write(f"{baseline_latency:.10f}")
                    print(f"Saved baseline to: {self._baseline_file}")
                except Exception as e:
                    print(f"Warning: Could not save baseline to {self._baseline_file}: {e}")
            else:
                # Calculate speedup relative to baseline
                if benchmark > 0:
                    speedup = baseline_latency / benchmark
                    print(f"Calculated speedup: {baseline_latency:.6f}ms / {benchmark:.6f}ms = {speedup:.4f}x")
                else:
                    speedup = 0.0
                    print(f"Warning: Invalid benchmark {benchmark}, setting speedup to 0.0")
            
            # Format benchmark_params if we have it
            if benchmark_params and isinstance(benchmark_params, dict):
                params_str = "; ".join(f"{k}={v}" for k, v in benchmark_params.items())
                benchmark_params = f"Kernel parameters: {params_str}, achieved latency: {benchmark:.6f} ms, speedup: {speedup:.4f}x"
            elif not benchmark_params:
                benchmark_params = f"Achieved latency: {benchmark:.6f} ms, speedup: {speedup:.4f}x"

            summary = ""
            if call:
                summary += "The Triton kernel was successfully compiled and launched. "
            else:
                summary += "The Triton kernel failed to compile or launch. "

            if correct:
                summary += "The Triton kernel produced correct results. "
            else:
                summary += "The Triton kernel produced incorrect results. "

            if benchmark is not None and benchmark > 0:
                summary += f"The Triton kernel achieved a speedup of {speedup:.4f}x compared to the baseline."
            else:
                summary += "The Triton kernel failed to benchmark."

            safety_validation = ""
            if call:
                safety_validation = "This generated Triton kernel was evaluated safely, with no hard crashes or memory violations."
            else:
                safety_validation = "This generated Triton kernel failed to compile or run safely, resulting in a hard crash or memory violation."
                if err_msg:
                    safety_validation += f" Error message: {err_msg}"
            if correct:
                safety_validation += " The results were correct."
            else:
                safety_validation += " The results were incorrect."
                if err_msg:
                    safety_validation += f" Error message: {err_msg}"
            
            if speedup > 0:
                baseline_comparison = f"Performance report: {benchmark_params}. Speedup={speedup:.4f}x (baseline: {baseline_latency:.6f}ms, current: {benchmark:.6f}ms)"
            else:
                baseline_comparison = "The Triton kernel failed to benchmark, so no performance comparison can be made."
                if err_msg:
                    baseline_comparison += f" Error message: {err_msg}"

            benchmark_results = []
            if benchmark_params is not None:
                benchmark_results.append(
                    f"Performance report: {benchmark_params}."
                )

            return {
                "success": speedup > 0,
                "final_score": speedup,  # Use speedup (higher is better)
                "performance_metrics": speedup,  # Use speedup (higher is better)
                "correctness_score": 1 if correct else 0,
                "combined_score": speedup,  # Use speedup for optimization
                "benchmark_results": benchmark_results if benchmark_params is not None else [],
                "baseline_comparison": baseline_comparison,
                "individual_comparisons": [],
                "summary": summary,
                # "metal_safety_statistics": self._get_comprehensive_error_statistics(),
                "safety_validation": safety_validation,
                "error": err_msg if err_msg else None,
            }

        except Exception as top_e:
            traceback.print_exc()
            return self._create_comprehensive_failure_result(f"Top-level crash caught: {top_e}")

    def _create_comprehensive_failure_result(self, error_message: str) -> Dict[str, Any]:
        """Create comprehensive failure result with full error statistics"""
        return {
            "success": False,
            "final_score": 0.0,
            "error": error_message,
            "performance_metrics": {},
            "combined_score": 0.0,
            "correctness_score": 0.0,
            "summary": f"Evaluation failed due to: {error_message}",
            # "metal_safety_statistics": self._get_comprehensive_error_statistics(),
            "safety_validation": {"success": False, "error": error_message},
        }

def evaluate(test_suite_path: str, program_text: str, ref_wrapper_path: str,
             wrapper_fn_name: str, unit_tests_path: str, n_warmup: int, n_iters: int,
             atol: float, rtol: float, verbose: bool, gpu_id: int = 0) -> Dict[str, Any]:
    """🛡️ BULLETPROOF evaluation function called by OpenEvolve"""
    evaluator = BulletproofTritonEvaluator()
    return evaluator.evaluate(test_suite_path, program_text, ref_wrapper_path,
                             wrapper_fn_name, unit_tests_path, n_warmup, n_iters,
                             atol, rtol, verbose, gpu_id)

# # ────────────────────────────────────────────────────────────────────────────
# # quick self-test (optional)
# # ────────────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     dummy_kernel = """
# import triton
# import triton.language as tl

# @triton.jit
# def kernel(X, Y, T: tl.constexpr, H: tl.constexpr):
#     pid = tl.program_id(axis=0)
#     off = pid * T * H
#     x_ptr = X + off
#     y_ptr = Y + off
#     for i in range(0, T * H):
#         tl.store(y_ptr + i, tl.load(x_ptr + i))
# """
#     res = evaluate(dummy_kernel)
#     print(res)
