# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/algorithmicsuperintelligence/openevolve  - Apache License 2.0

from __future__ import annotations

import os, sys, time, traceback, tempfile, importlib, types, math, subprocess, json, re
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


    # ======================================================================
    # Public entry-point used by OpenEvolve
    # ======================================================================
    def evaluate(
                    self, 
                    test_suite_path: str, 
                    program_text: str, 
                    ref_wrapper_path:str, 
                    wrapper_fn_name: str, 
                    unit_tests_path:str, 
                    n_warmup: int=1000, 
                    n_iters: int=500,
                    atol: float=1e-3, 
                    rtol: float=1e-3, 
                    verbose: bool=False,
                    gpu_id: int=0
                ) -> Dict[str, Any]:
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
            # python test_suite.py --wrapper_ref_path templates/sample.py --wrapper_path templates/kernel.py --wrapper_name add_wrapper --device cuda --n_warmup 10 --n_iters 100 --atol 1e-3 --rtol 1e-3 --verbose
            cmd = [
                sys.executable,
                test_suite_path,
                "--wrapper_ref_path", ref_wrapper_path,
                "--wrapper_path", program_text,
                "--wrapper_name", wrapper_fn_name,
                "--unit_test_path", unit_tests_path,
                "--device", "cuda",
                "--n_warmup", str(n_warmup),
                "--n_iters", str(n_iters),
                "--atol", str(atol),
                "--rtol", str(rtol),
            ]

            if verbose:
                cmd.append("--verbose")

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


            # Parse the JSON output from the command
            try:
                ## extract ```json \n ... \n``` from result.stdout
                json_output_match = re.search(r"```json\s*(\{.*?\})\s*```", result.stdout, re.DOTALL)
                if not json_output_match:
                    raise json.JSONDecodeError("No JSON output found", result.stdout, 0)
                output = json.loads(json_output_match.group(1))

                status = output.get("status")
                benchmark = output.get("avg_speedup")
                benchmark_params = output.get("benchmark")
                failed_tests = output.get("failed_tests")
                num_tests = output.get("num_tests")
                msg = output.get("msg")

                call = True
                correct = status
                
            except json.JSONDecodeError as e:
                err_msg = f"Failed to parse JSON output: {e} with stdout: {result.stdout.strip()}"
                print(err_msg)

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

def evaluate(
                test_suite_path: str, 
                program_text: str, 
                ref_wrapper_path:str, 
                wrapper_fn_name: str, 
                unit_tests_path:str, 
                n_warmup: int=1000, 
                n_iters: int=500,
                atol: float=1e-3, 
                rtol: float=1e-3, 
                verbose: bool=False,
                gpu_id: int=0
            ) -> Dict[str, Any]:
    """🛡️ BULLETPROOF evaluation function called by OpenEvolve"""
    evaluator = BulletproofTritonEvaluator()

    return evaluator.evaluate(
                                test_suite_path=test_suite_path, 
                                program_text=program_text, 
                                ref_wrapper_path=ref_wrapper_path, 
                                wrapper_fn_name=wrapper_fn_name, 
                                unit_tests_path=unit_tests_path, 
                                n_warmup=n_warmup, 
                                n_iters=n_iters, 
                                atol=atol, 
                                rtol=rtol, 
                                verbose=verbose, 
                                gpu_id=gpu_id
                            )
