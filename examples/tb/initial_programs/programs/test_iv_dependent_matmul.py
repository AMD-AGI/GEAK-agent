# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
######################################## Imports ######################################## 
import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton.language as tl
######################################## Imports ######################################## 

@triton.jit
def iv_dependent_matmul(a_ptr, b_ptr, c_ptr,  #
            M, N, K,  #
            stride_am, stride_ak,  #
            stride_bk, stride_bn,  #
            stride_cm, stride_cn,  #
            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
            type: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptr = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptr = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    a_ptrs = a_ptr
    b_ptrs = b_ptr
    if type == "post_load_two_iters":
        a_ptrs_next = a_ptr + BLOCK_SIZE_K * stride_ak
        b_ptrs_next = b_ptr + BLOCK_SIZE_K * stride_bk
    elif type == "post_load_three_iters":
        a_ptrs_next = a_ptr + BLOCK_SIZE_K * stride_ak
        b_ptrs_next = b_ptr + BLOCK_SIZE_K * stride_bk
        a_ptrs_next_next = a_ptr + 2 * BLOCK_SIZE_K * stride_ak
        b_ptrs_next_next = b_ptr + 2 * BLOCK_SIZE_K * stride_bk

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if type == "pre_load":
            a_ptrs = a_ptr + k * BLOCK_SIZE_K * stride_ak
            b_ptrs = b_ptr + k * BLOCK_SIZE_K * stride_bk
        elif type == "post_pre_mixed":
            a_ptrs = a_ptr + k * BLOCK_SIZE_K * stride_ak
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        if type == "post_load":
            a_ptrs = a_ptr + (k + 1) * BLOCK_SIZE_K * stride_ak
            b_ptrs = b_ptr + (k + 1) * BLOCK_SIZE_K * stride_bk
        elif type == "post_pre_mixed":
            b_ptrs = b_ptr + (k + 1) * BLOCK_SIZE_K * stride_bk
        elif type == "post_load_two_iters":
            a_ptrs = a_ptrs_next
            b_ptrs = b_ptrs_next
            a_ptrs_next = a_ptr + (k + 2) * BLOCK_SIZE_K * stride_ak
            b_ptrs_next = b_ptr + (k + 2) * BLOCK_SIZE_K * stride_bk
        elif type == "post_load_three_iters":
            a_ptrs = a_ptrs_next
            b_ptrs = b_ptrs_next
            a_ptrs_next = a_ptrs_next_next
            b_ptrs_next = b_ptrs_next_next
            a_ptrs_next_next = a_ptr + (k + 3) * BLOCK_SIZE_K * stride_ak
            b_ptrs_next_next = b_ptr + (k + 3) * BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

##################################################################################################################################################  

import numpy as np
import random
import torch 
import os
import pytest
from numpy.random import RandomState
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



@pytest.mark.parametrize("type",
                         ["pre_load", "post_load", "post_pre_mixed", "post_load_two_iters", "post_load_three_iters"])
def test_iv_dependent_matmul(type, request, device='cuda'):

    
    set_seed()

    M = 256
    K = 256
    N = 256
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_M = 32

    a = torch.rand((M, K), device=device)
    b = torch.rand((K, N), device=device)

    torch_output = torch.mm(a, b)
    triton_output = torch.empty_like(torch_output, device=torch_output.device)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    num_stages = 4 if type == "post_load_three_iters" else 3
    iv_dependent_matmul[grid](
        a, b, triton_output, M, N, K,  #
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),  #
        triton_output.stride(0), triton_output.stride(1),  #
        type=type,  #
        # BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, 
        # num_stages=num_stages
        )

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    ################### save triton_output in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = triton_output.clone().detach().cpu()
    ################################################################### 

    torch.testing.assert_close(torch_output, triton_output, rtol=1e-2, atol=1e-2)


# --- Python wrapper for the kernel for benchmarking ---
def iv_dependent_matmul_triton_wrapper(a_tensor, b_tensor, c_buffer,
                                       M_dim, N_dim, K_dim,
                                       block_m_const, block_n_const, block_k_const,
                                       kernel_type_str, num_stages_launch, num_warps_launch):
    grid = (triton.cdiv(M_dim, block_m_const) * triton.cdiv(N_dim, block_n_const), )
    
    iv_dependent_matmul[grid](
        a_tensor, b_tensor, c_buffer, M_dim, N_dim, K_dim,
        a_tensor.stride(0), a_tensor.stride(1),
        b_tensor.stride(0), b_tensor.stride(1),
        c_buffer.stride(0), c_buffer.stride(1),
        # BLOCK_SIZE_M=block_m_const, BLOCK_SIZE_N=block_n_const,
        # BLOCK_SIZE_K=block_k_const, 
        type=kernel_type_str,
        # num_stages=num_stages_launch,
        # num_warps=num_warps_launch
    )
    return c_buffer

# --- Define TFLOPS and GB/s calculators for GEMM ---
def calculate_gemm_tflops(params: dict, ms: float) -> float: # Standard GEMM
    M, N, K = params['M'], params['N'], params['K']
    flops = 2 * M * N * K 
    tflops = flops / (ms / 1000) / 1e12
    return tflops

def calculate_gemm_gbps(params: dict, ms: float) -> float: # Standard GEMM
    M, N, K = params['M'], params['N'], params['K']
    dtype_str = params.get('input_dtype_str', 'fp32') # Inputs are fp32 in this test
    
    current_dtype = torch.float32
    if dtype_str == 'fp16': current_dtype = torch.float16
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16
    element_size = torch.tensor([], dtype=current_dtype).element_size()

    # Output is fp16
    out_element_size = torch.tensor([], dtype=torch.float16).element_size()

    bytes_a = M * K * element_size
    bytes_b = K * N * element_size
    bytes_c_write = M * N * out_element_size # Output is fp16
    total_bytes = bytes_a + bytes_b + bytes_c_write
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

OP_NAME_FOR_BENCHMARK = "iv_dependent_matmul_perf"

# --- Pytest parametrize for performance testing ---
IV_MATMUL_SHAPES_FOR_PERF = [
    # M,   N,   K
    (256, 256, 256), (512, 512, 128), (1024, 1024, 64),
    (2048, 512, 128), (512, 2048, 64)
]
IV_MATMUL_BLOCK_CONFIGS_FOR_PERF = [
    # BM, BN, BK
    (32, 32, 32), (64, 64, 32), (64, 32, 64), (32, 64, 64),
    (128, 128, 32), (128, 64, 64), (64, 128, 64) 
    # Add (128,128,64) if K can be larger for BLOCK_K
]
IV_MATMUL_TYPES_FOR_PERF = ["pre_load", "post_load", "post_pre_mixed", "post_load_two_iters", "post_load_three_iters"]
IV_MATMUL_DTYPES_FOR_PERF = ['fp32', 'fp16'] # Input dtypes for A and B
# num_stages and num_warps are launch hints
NUM_STAGES_FOR_PERF = [2, 3, 4]
NUM_WARPS_FOR_PERF = [4, 8]


@pytest.mark.parametrize("m_n_k_shape", IV_MATMUL_SHAPES_FOR_PERF)
@pytest.mark.parametrize("block_config", IV_MATMUL_BLOCK_CONFIGS_FOR_PERF)
@pytest.mark.parametrize("kernel_type_str", IV_MATMUL_TYPES_FOR_PERF)
@pytest.mark.parametrize("input_dtype_str", IV_MATMUL_DTYPES_FOR_PERF)
@pytest.mark.parametrize("num_stages_launch", NUM_STAGES_FOR_PERF)
@pytest.mark.parametrize("num_warps_launch", NUM_WARPS_FOR_PERF)
def test_performance(m_n_k_shape, block_config, kernel_type_str, input_dtype_str, 
                                         num_stages_launch, num_warps_launch, request):
    set_seed()
    M, N, K = m_n_k_shape
    BLOCK_M, BLOCK_N, BLOCK_K = block_config

    # Skip if BLOCK_K is too large for K (kernel has K-loop)
    if BLOCK_K > K :
        pytest.skip(f"BLOCK_K ({BLOCK_K}) > K ({K}) not meaningful for tiled K-loop.")
    
    # Shared memory check (approx for one dot A(BM,BK) @ B(BK,BN) -> C(BM,BN))
    # Max shared mem usage is for A and B blocks in one dot.
    # Smem elements = BM*BK + BK*BN
    # This kernel has a K-loop, so BK is a tile size, not full K.
    if input_dtype_str == 'fp32': current_in_dtype = torch.float32; elem_size = 4
    elif input_dtype_str == 'bf16': current_in_dtype = torch.bfloat16; elem_size = 2
    else: current_in_dtype = torch.float16; elem_size = 2
    
    # Output is always fp16 by the kernel
    output_dtype = torch.float16

    smem_elements_needed = (BLOCK_M * BLOCK_K) + (BLOCK_K * BLOCK_N)
    # num_stages can increase shared memory usage for pipelining
    # A rough factor could be num_stages, or num_stages/2 + 1 etc.
    # Let's use a factor of num_stages for a conservative estimate.
    if smem_elements_needed * elem_size * (num_stages_launch if num_stages_launch > 1 else 1) > 65536:
        pytest.skip(f"Skipping M{M}N{N}K{K} Blocks({BLOCK_M},{BLOCK_N},{BLOCK_K}) "
                    f"dtype {input_dtype_str} stages {num_stages_launch} "
                    f"due to estimated shared memory.")

    a = torch.randn((M, K), device='cuda', dtype=current_in_dtype)
    b = torch.randn((K, N), device='cuda', dtype=current_in_dtype)
    # Kernel casts output to tl.float16
    triton_output_buffer = torch.empty((M, N), device='cuda', dtype=output_dtype)

    op_lambda = lambda: iv_dependent_matmul_triton_wrapper(
        a, b, triton_output_buffer, M, N, K,
        BLOCK_M, BLOCK_N, BLOCK_K,
        kernel_type_str, num_stages_launch, num_warps_launch
    )

    bench_config = do_bench_config(warm_up=25, repetition=100)
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)
    current_params_for_logs_and_calc = {
        "M": M, "N": N, "K": K,
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
        "type": kernel_type_str, "input_dtype_str": input_dtype_str, "output_dtype_str": "fp16",
        "num_stages": num_stages_launch, "num_warps": num_warps_launch
    }
    
    perf_result = benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
                                            gbps_calculator=calculate_gemm_gbps,
                                            tflops_calculator=calculate_gemm_tflops)


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