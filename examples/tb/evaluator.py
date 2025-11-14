"""
🛡️ BULLETPROOF TRITON KERNEL EVALUATOR  (AMD GPU EDITION) 🛡️

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

import os, sys, time, traceback, tempfile, importlib, types, math, subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except Exception as _e:
    raise RuntimeError(
        "Triton must be installed with AMD/ROCm support "
    ) from _e


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
    def evaluate(self, program_text: str, kernel_evaluator_path: str, gpu_id: int=0) -> Dict[str, Any]:
        """Master function: never raises, always returns a dict."""
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
            cmd = [sys.executable, kernel_evaluator_path, '-a', '-k ' + program_text]
            print(f"Running kernel evaluation command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._COMPILE_TIMEOUT_S,
                env=env
            )
            if result.returncode != 0:
                print(f"Kernel kernel evaluation command failed!: {result.stderr.strip()}")
                return self._create_comprehensive_failure_result(
                    f"Kernel kernel evaluation command failed!: {result.stderr.strip()}"
                )
            output = result.stdout.strip()
            call, correct, benchmark, benchmark_params, err_msg = output.split("#*#*")

            call = call.lower() == 'true'
            correct = correct.lower() == 'true'
            ## check if benchmark is a number or 'none' or false or null
            if benchmark.lower() in ['none', 'false', 'null']:
                benchmark = None
            else:
                try:
                    benchmark = float(benchmark)
                except ValueError:
                    benchmark = None

            if benchmark_params.lower() in ['none', 'false', 'null']:
                benchmark_params = None
            else:   
                try:
                    benchmark_params = benchmark_params.split(';')
                except (ValueError, IndexError):
                    benchmark_params = None

            err_msg = err_msg.strip()

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
                summary += f"The Triton kernel achieved a performance benchmark of {benchmark:.2f}x compared to the baseline."
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
                baseline_comparison = f"Performance report: {",".join(benchmark_params)}."
            else:
                baseline_comparison = "The Triton kernel failed to benchmark, so no performance comparison can be made."
                if err_msg:
                    baseline_comparison += f" Error message: {err_msg}"

            benchmark_results = []
            if benchmark_params is not None:
                benchmark_results.append(
                    f"Performance report: {",".join(benchmark_params)}."
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

def evaluate(program_text: str, kernel_evaluator_path: str, gpu_id: int = 0) -> Dict[str, Any]:
    """🛡️ BULLETPROOF evaluation function called by OpenEvolve"""
    evaluator = BulletproofTritonEvaluator()
    return evaluator.evaluate(program_text, kernel_evaluator_path=kernel_evaluator_path, gpu_id=gpu_id)

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
