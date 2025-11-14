# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

"""
Mixed precision tests for matmul (tl.dot) with cast (tl.to)

issue: https://github.com/triton-lang/triton/issues/2523
"""



######################################## Imports#######################################
import pytest
import torch

import triton
import triton.language as tl
######################################## Imports#######################################

######################################## HELPERS utils ######################################## 
input_dtypes = ["float16", "float32", "float64"]
out_dtypes = ["float16", "float32"]
######################################## HELPERS utils ######################################## 


@triton.jit
def matmul_kernel(A, B, C, M, N, K,  #
                  stride_am, stride_ak,  #
                  stride_bk, stride_bn,  #
                  stride_cm, stride_cn,  #
                  dot_out_dtype: tl.constexpr,  #
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,  #
                  BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    # matrix multiplication
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
        a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
        b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        a = a.to(C.dtype.element_ty)
        b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=dot_out_dtype)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, acc, mask=mask)

##################################################################################################################################################  
import numpy as np
import random
import torch 
import os
from tb_eval.perf.ROCm.performance_utils_pytest import PytestBenchmarker, do_bench_config, save_all_benchmark_results
from typing import Dict


result_gold = {}
######################################## HELPERS for Eval ########################################
def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility across multiple libraries and configure PyTorch for deterministic behavior.

    Args:
        seed (int): The seed value to set. Default is 42.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)
    # Set seed for NumPy
    np.random.seed(seed)
    # Set seed for PyTorch on CPU
    torch.manual_seed(seed)
    # Set seed for PyTorch on all GPUs (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variable for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)

######################################## HELPERS for Eval ########################################


@pytest.mark.parametrize("M, K, N, w_dtype, x_dtype, out_dtype",
                         [(M, K, N, w, x, o)  #
                          for (M, K, N) in [(128, 128, 128), (1280, 768, 1024)]  #
                          for w in input_dtypes
                          for x in input_dtypes  #
                          for o in out_dtypes])
def test_cast_matmul(M, K, N, w_dtype, x_dtype, out_dtype, request):
    set_seed()

    if x_dtype == w_dtype:
        pytest.skip("skip the same input dtype")
    device = torch.cuda.current_device()
    x_dtype = getattr(torch, x_dtype)
    w_dtype = getattr(torch, w_dtype)
    a = torch.randn((M, K), device=device, dtype=x_dtype)
    b = torch.randn((K, N), device=device, dtype=w_dtype)
    torch_dtype = getattr(torch, out_dtype)
    triton_dtype = getattr(tl, out_dtype)  # <- here force dot_out_dtype
    out_torch = torch.matmul(a.to(torch_dtype), b.to(torch_dtype))
    out_triton = torch.empty((M, N), device=device, dtype=torch_dtype)

    # launch kernel
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 32
    grid = ((triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)), 1)

    matmul_kernel[grid](
        a, b, out_triton, M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        out_triton.stride(0), out_triton.stride(1), dot_out_dtype=triton_dtype,  #
        # GROUP_M=8,  #
        # BLOCK_M=BLOCK_M,  #
        # BLOCK_N=BLOCK_N,  #
        # BLOCK_K=BLOCK_K
        )

    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = out_triton.clone().detach().cpu()
    ###################################################################

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    torch.testing.assert_close(out_torch, out_triton, atol=0.3, rtol=0.01)

# --- Python wrapper for the kernel ---
def cast_matmul_triton_wrapper(a_tensor, b_tensor, out_tensor, # out_tensor is C
                                M, N, K, # runtime dims
                                triton_dot_out_dtype, # tl.dtype for dot_out_dtype
                                BLOCK_M_const, BLOCK_N_const, BLOCK_K_const, GROUP_M_const, # constexpr
                                num_warps_arg # For launch, if kernel was autotuned for it
                               ):
    grid = (triton.cdiv(M, BLOCK_M_const) * triton.cdiv(N, BLOCK_N_const), 1) # As per original test
    # The original test calculates grid based on M, N, BLOCK_M, BLOCK_N.
    # The kernel itself uses GROUP_M for pid reordering.
    # The grid should be (num_programs_total, 1, 1) or just (num_programs_total,).
    # num_programs_total = group_count * group_width = cdiv(M, BLOCK_M*GROUP_M) * (GROUP_M * cdiv(N,BLOCK_N))
    # This simplifies to cdiv(M,BLOCK_M) * cdiv(N,BLOCK_N) if group_id logic is correct.
    # Let's stick to original test's grid calculation.
    
    matmul_kernel[grid](
        a_tensor, b_tensor, out_tensor, M, N, K,
        a_tensor.stride(0), a_tensor.stride(1),
        b_tensor.stride(0), b_tensor.stride(1),
        out_tensor.stride(0), out_tensor.stride(1),
        dot_out_dtype=triton_dot_out_dtype, # Pass as constexpr
        # BLOCK_M=BLOCK_M_const, BLOCK_N=BLOCK_N_const,
        # BLOCK_K=BLOCK_K_const, GROUP_M=GROUP_M_const,
        # num_warps=num_warps_arg # This kernel is not autotuned, so num_warps in launch is a hint
                                 # to underlying compiler/scheduler but not a JIT param.
    )
    return out_tensor


# --- Define TFLOPS and GB/s calculators for this specific GEMM ---
def calculate_cast_matmul_tflops(params: dict, ms: float) -> float:
    M, N, K = params['M'], params['N'], params['K']
    # Standard GEMM FLOPs: 2 * M * N * K
    # Casting operations are usually not counted in high-level TFLOPS unless very significant.
    flops = 2 * M * N * K
    tflops = flops / (ms / 1000) / 1e12
    return tflops

def get_torch_dtype(dtype_str):
    if dtype_str == "float64": return torch.float64
    if dtype_str == "float32": return torch.float32
    return torch.float16 # Default for "float16" or others

def calculate_cast_matmul_gbps(params: dict, ms: float) -> float:
    M, N, K = params['M'], params['N'], params['K']
    # Dtypes from params
    a_dtype_str = params['a_dtype_str'] # x_dtype in original test
    b_dtype_str = params['b_dtype_str'] # w_dtype in original test
    c_dtype_str = params['c_dtype_str'] # out_dtype in original test

    element_size_a = torch.tensor([], dtype=get_torch_dtype(a_dtype_str)).element_size()
    element_size_b = torch.tensor([], dtype=get_torch_dtype(b_dtype_str)).element_size()
    element_size_c = torch.tensor([], dtype=get_torch_dtype(c_dtype_str)).element_size()

    bytes_a = M * K * element_size_a
    bytes_b = K * N * element_size_b
    bytes_c_write = M * N * element_size_c # Written in C's dtype

    total_bytes = bytes_a + bytes_b + bytes_c_write
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

# --- Name for the benchmark output JSON file ---
OP_NAME_FOR_BENCHMARK = "cast_matmul_triton_perf"

# --- Pytest parametrize for performance testing ---
# Shapes from original test
CAST_MATMUL_SHAPES_FOR_PERF = [(128, 128, 128), (1024, 768, 1280), (1280, 1024, 768)] # Swapped K,N for one case
                                                                                  # Original: (1280, 768, 1024) -> M,K,N

# Dtypes for performance testing (subset of original to keep it manageable)
# Focusing on cases where casting might impact performance.
# a_dtype (x_dtype), b_dtype (w_dtype), c_dtype (out_dtype), dot_acc_dtype (dot_out_dtype)
CAST_MATMUL_DTYPES_FOR_PERF = [
    # (a_str, b_str, c_str, dot_acc_tl_dtype_str)
    ("float32", "float16", "float16", "float32"), # A(f32), B(f16) -> C(f16), Acc(f32)
    ("float16", "float32", "float16", "float32"), # A(f16), B(f32) -> C(f16), Acc(f32)
    ("float16", "float16", "float32", "float32"), # A(f16), B(f16) -> C(f32), Acc(f32)
    ("float32", "float32", "float16", "float16"), # A(f32), B(f32) -> C(f16), Acc(f16) <- check if acc in f16 is intended/good
    ("float16", "float16", "float16", "float16"), # All f16
    ("float32", "float32", "float32", "float32"), # All f32
    # ("bfloat16", "bfloat16", "bfloat16", "float32"), # bf16 example
]

# Kernel block sizes are constexpr. Original test uses fixed BLOCK_M,N,K = 16,16,32 and GROUP_M=8.
# For performance, these should ideally be tuned or part of parametrization.
# For now, use the fixed ones from the original test.
FIXED_BLOCK_M = 16
FIXED_BLOCK_N = 16
FIXED_BLOCK_K = 32
FIXED_GROUP_M = 8


@pytest.mark.parametrize("M, K, N", CAST_MATMUL_SHAPES_FOR_PERF) # Note M,K,N order
@pytest.mark.parametrize("a_dtype_str, b_dtype_str, c_dtype_str, dot_acc_tl_dtype_str", CAST_MATMUL_DTYPES_FOR_PERF)
def test_performance(M, K, N, a_dtype_str, b_dtype_str, c_dtype_str, dot_acc_tl_dtype_str, request):
    set_seed()
    device = torch.cuda.current_device() # Use current device

    # Skip same input dtypes as per original test logic for functional part
    # For performance, we might want to test them, but let's follow original skip.
    if a_dtype_str == b_dtype_str:
        pytest.skip("Skipping same input dtypes for A and B for this performance test.")

    a_torch_dtype = get_torch_dtype(a_dtype_str)
    b_torch_dtype = get_torch_dtype(b_dtype_str)
    c_torch_dtype = get_torch_dtype(c_dtype_str)
    
    # Convert string representation of tl dtype to actual tl.dtype
    if dot_acc_tl_dtype_str == "float32": dot_triton_dtype = tl.float32
    elif dot_acc_tl_dtype_str == "float16": dot_triton_dtype = tl.float16
    # Add more if other tl dtypes are used for dot_out_dtype, e.g. tl.int32
    else: raise ValueError(f"Unsupported dot_out_dtype_str: {dot_acc_tl_dtype_str}")


    a = torch.randn((M, K), device=device, dtype=a_torch_dtype)
    b = torch.randn((K, N), device=device, dtype=b_torch_dtype) # B is (K,N) for A(M,K) @ B(K,N)
    
    # Output tensor for Triton kernel
    out_triton = torch.empty((M, N), device=device, dtype=c_torch_dtype)

    # --- Create op_lambda for benchmarking ---
    op_lambda = lambda: cast_matmul_triton_wrapper(
        a, b, out_triton, M, N, K,
        triton_dot_out_dtype=dot_triton_dtype, # Pass actual tl.dtype
        BLOCK_M_const=FIXED_BLOCK_M, BLOCK_N_const=FIXED_BLOCK_N,
        BLOCK_K_const=FIXED_BLOCK_K, GROUP_M_const=FIXED_GROUP_M,
        num_warps_arg=4 # Example, not used by JIT kernel params but by launch if autotuned for it
    )

    # --- Benchmarking ---
    bench_config = do_bench_config(warm_up=25, repetition=100)
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)

    current_params_for_logs_and_calc = {
        "M": M, "N": N, "K": K,
        "a_dtype_str": a_dtype_str, "b_dtype_str": b_dtype_str, "c_dtype_str": c_dtype_str,
        "dot_acc_dtype_str": dot_acc_tl_dtype_str, # Log the string version
        "BLOCK_M": FIXED_BLOCK_M, "BLOCK_N": FIXED_BLOCK_N,
        "BLOCK_K": FIXED_BLOCK_K, "GROUP_M": FIXED_GROUP_M
    }

    benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
                              gbps_calculator=calculate_cast_matmul_gbps,
                              tflops_calculator=calculate_cast_matmul_tflops)

######################################## HELPERS for Eval ########################################     
# --- Pytest hook to save the dictionary at the end of the session ---  
def test_save_results():  
    """  
    Called after whole test run finished, right before returning the exit status to the system.  
    """
    print('Inside session finish...')
    if "_CALL_SUCCESS_" not in result_gold:
        result_gold['_CALL_SUCCESS_'] = torch.tensor([[0.0]])
    OUTPUT_FILENAME = __file__.replace('.','_') + '.pt'
    print(f"\nSaving all y_triton results to {OUTPUT_FILENAME}...")  
    # Ensure the directory for the output file exists if it's in a subdirectory  
    output_dir = os.path.dirname(OUTPUT_FILENAME)  
    if output_dir and not os.path.exists(output_dir):  
        os.makedirs(output_dir, exist_ok=True)  
    torch.save(result_gold, OUTPUT_FILENAME)       
    print(f"Successfully saved {len(result_gold)} y_triton tensors to {OUTPUT_FILENAME}.")  


def test_save_performance_results():
    """
    Called after the test_performance function finishes.
    This is a separate hook to ensure performance results are saved.
    """
    print('\nPytest session finishing... Saving benchmark results...')

    output_directory = os.path.join(os.path.dirname(__file__), "perf")  # Save in a "perf" subdirectory next to the test file
    os.makedirs(output_directory, exist_ok=True)
    
    save_all_benchmark_results(output_directory)
    print(f"All benchmark results attempted to save to: {output_directory}")


######################################## HELPERS for Eval ########################################