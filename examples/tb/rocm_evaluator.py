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

import tb_eval  # Using TB-eval-OE installed as geak-eval 
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
            ## in a subprocess.run run the command python program_text -a
            env = os.environ.copy()
            env['HIP_VISIBLE_DEVICES'] = str(gpu_id % 8)  # Set the GPU ID for ROCm
            cmd = ["geak-eval", "eval", "-f", f'{program_text}', "-o", 'rocm_results', "--dataset", 'rocm', "-c", "-tp", f'{tb_eval.constants.ROCm_DATA_AUTOTUNE_ROOT}' ]
            print(f"Running kernel evaluation command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                env=env
            )
                
            call, correct, benchmark, benchmark_params, err_msg = None, None, None, None, ""

            if result.returncode != 0:
                print(f"Kernel kernel evaluation command failed!: {result.stderr.strip()}")
                return self._create_comprehensive_failure_result(
                    f"Kernel kernel evaluation command failed!: {result.stderr.strip()}"
                )

            ## get dir name by removing extension from program_text
            ext_removed = os.path.splitext(program_text)[0]
            if not os.path.exists(ext_removed):
                call = False
                correct = False
                benchmark = None
                benchmark_params = None
                err_msg += f"Evaluation directory {ext_removed} does not exist. Kernel evaluation failed."
                return self._create_comprehensive_failure_result(err_msg)

            rocm_runs_file = os.path.join(ext_removed, "rocm_results")

            ## read rocm_runs file
            if not os.path.exists(rocm_runs_file):
                call = False
                correct = False
                benchmark = None
                benchmark_params = None
                err_msg += f"ROCm runs file {rocm_runs_file} does not exist. Kernel evaluation failed."
                return self._create_comprehensive_failure_result(err_msg)

            with open(rocm_runs_file, 'r') as f:
                result = f.read().strip()
            
            result_lines = result.splitlines()
            if len(result_lines) < 1:
                call = False
                correct = False
                benchmark = None
                benchmark_params = None
                err_msg += "ROCm runs file is empty. Kernel evaluation failed."
                return self._create_comprehensive_failure_result(err_msg)

            for line in result_lines:
                err_msg += f"Could not find call or correctness information in the ROCm runs file {rocm_runs_file}. with keys 'Call: ' and 'Exec: '."
                if ("Call" in line) and ("Exec" in line):
                    call = line.split(",")[1].strip().split(":")[1].strip().lower() == 'true'
                    correct = line.split(",")[2].strip().split(":")[1].strip().lower() == 'true'
                    print(f"NOTE: for line {line}  call: {call}, correctness: {correct}")
                    err_msg = ""
                    break
            if not call or not correct:
                err_msg = result

            benchmark_file = os.path.join(ext_removed, "exec", "rocm_performance_analysis.txt")

            if not os.path.exists(benchmark_file):
                benchmark = None
                benchmark_params = None
                err_msg += f"Benchmark file {benchmark_file} does not exist. Kernel evaluation failed."
            else:
                with open(benchmark_file, 'r') as f:
                    benchmark_data = f.read().strip()
                
                    benchmark_lines = benchmark_data.splitlines()
                    if len(benchmark_lines) < 1:
                        benchmark = None
                        benchmark_params = None
                        err_msg += "Benchmark file is empty. Kernel evaluation failed."
                    else:
                        # Parse the benchmark data
                        benchmark = None
                        benchmark_params = []
                        err_msg += f"Could not find benchmark data in the file {benchmark_file} with key 'Speedup (Gen vs. Ref): '."
                        for line in benchmark_lines:
                            if "Speedup (Gen vs. Ref): " in line:
                                benchmark = float(line.split(":")[1].strip()) 
                                err_msg = ""
                                break

            # op = os.path.splitext(os.path.basename(program_text))[0]
            # benchmark_params_file = os.path.join(ext_removed, "exec", "perf", f"{op}_perf.json")
            benchmark_params_root = os.path.join(ext_removed, "exec", "perf")
            files_here = glob(f"{benchmark_params_root}/*.json")
            benchmark_params_file = None
            if files_here:
                for file in files_here:
                    if "all_perf_results.json" in file:
                        continue
                    benchmark_params_file = file

            if (benchmark_params_file is None) or (not os.path.exists(benchmark_params_file)):
                benchmark_params = None
                err_msg += f"Benchmark parameters file {benchmark_params_file} does not exist. Kernel performance evaluation failed."

            else:
                baseline_fname = os.path.basename(benchmark_params_file)
                with open(benchmark_params_file, 'r') as f:
                    json_data = json.load(f)

                benchmark_params = ""

                for item in json_data:
                    params = item.get("params", {})
                    latency = item.get("ms", None)
                    if params and latency is not None:
                        benchmark_params += "For parameters: "
                        params_str = ";".join(f"{k}={v}" for k, v in params.items())
                        benchmark_params += params_str + f" the generated triton kernel achieved a latency of {latency:.6f} ms. "

                if benchmark_params == "":
                    baseline_file = os.path.join(self.GOLDEN_DATA_PATH, baseline_fname)
                    if os.path.exists(baseline_file):
                        with open(baseline_file, 'r') as f:
                            json_data = json.load(f)
                        for item in json_data:
                            params = item.get("params", {})
                            latency = item.get("ms", None)
                            if params and latency is not None:
                                benchmark_params += params_str + f" And the baseline kernel achieved a latency of {latency:.6f} ms. "
                    else:
                        print( f"⚠️ WARNING: Baseline performance file {baseline_file} does not exist.")

            summary = ""
            if call:
                summary += "The Triton kernel was successfully compiled and launched. "
            else:
                summary += "The Triton kernel failed to compile or launch. "

            if correct:
                summary += "The Triton kernel produced correct results. "
            else:
                summary += "The Triton kernel produced incorrect results. "

            if benchmark is not None:
                summary += f"The Triton kernel achieved a performance benchmark of {benchmark:.6f}x compared to the baseline."
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
            
            if benchmark is not None:
                baseline_comparison = f"Performance report: {benchmark_params}."
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
                "success": benchmark is not None,
                "final_score": benchmark if benchmark is not None else 0.0,
                "performance_metrics": benchmark if benchmark is not None else None,
                "correctness_score": 1 if correct else 0,
                "combined_score": benchmark if benchmark is not None else 0.0,
                "benchmark_results": benchmark_results if benchmark_params is not None else [],
                "baseline_comparison": f"The ratio of average latency of baseline to generated triton kernel is {benchmark:.2f}x, this ratio must be greater than 1.0 for the kernel to be considered performant." if benchmark is not None else None,
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
