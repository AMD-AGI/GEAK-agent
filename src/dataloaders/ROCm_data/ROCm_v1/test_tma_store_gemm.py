# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.

# Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

######################################## Imports ######################################## 

import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl
######################################## Imports ######################################## 


@triton.jit
def matmul_tma_load_store(  #
        a_ptr, b_ptr, c_ptr,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        OUTPUT_F16: tl.constexpr  #
):
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(0, 0),
                                    block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), offsets=(0, 0),
                                    block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn), offsets=(0, 0),
                                    block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    a = tl.load(a_block_ptr)
    b = tl.load(b_block_ptr)

    c = tl.dot(a, b)
    if OUTPUT_F16:
        c = c.to(tl.float16)

    tl.store(c_block_ptr, c)

##################################################################################################################################################  


import pytest
import numpy as np
import random
import torch 
import os
import pytest
from torch.testing import assert_close
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



@pytest.mark.parametrize('M,N,K,NUM_CTAS,NUM_WARPS,TRANS_A,TRANS_B,OUTPUT_F16', [
    [64, 64, 16, 1, 4, False, True, False],
    [64, 64, 16, 1, 4, False, True, True],
    [128, 64, 32, 1, 4, False, True, False],
    [128, 64, 32, 1, 4, False, True, True],
    [64, 128, 32, 1, 4, False, True, False],
    [64, 128, 32, 1, 4, False, True, True],
    [128, 128, 64, 1, 4, False, True, False],
    [128, 128, 64, 1, 4, False, True, True],
])
def test_tma_load_store(M, N, K, NUM_CTAS, NUM_WARPS, TRANS_A, TRANS_B, OUTPUT_F16, request):
    set_seed()

    if (TRANS_A):
        a = torch.randn((K, M), device='cuda', dtype=torch.float16).T
    else:
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    if (TRANS_B):
        b = torch.randn((N, K), device='cuda', dtype=torch.float16).T
    else:
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    if OUTPUT_F16:
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    matmul_tma_load_store[(1, 1)](
        a_ptr=a, b_ptr=b, c_ptr=c,  #
        M=M, N=N, K=K,  #
        stride_am=a.stride(0), stride_ak=a.stride(1),  #
        stride_bk=b.stride(0), stride_bn=b.stride(1),  #
        stride_cm=c.stride(0), stride_cn=c.stride(1),  #
        BLOCK_M=M, BLOCK_N=N, BLOCK_K=K,  #
        num_warps=NUM_WARPS, num_ctas=NUM_CTAS,  #
        OUTPUT_F16=OUTPUT_F16)
    golden = torch.matmul(a, b)
    torch.set_printoptions(profile="full")

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = c.clone().detach().cpu()
    ###################################################################

    assert_close(c, golden, rtol=1e-2, atol=1e-3, check_dtype=False)


# --- Python wrapper for the kernel for benchmarking ---
def tma_gemm_triton_wrapper(a_tensor, b_tensor, c_buffer, 
                            M_dim, N_dim, K_dim, 
                            output_f16_flag, num_warps_launch): # num_ctas_launch removed as not used by grid
    grid = (1,) 
    matmul_tma_load_store[grid](
        a_ptr=a_tensor, b_ptr=b_tensor, c_ptr=c_buffer,
        M=M_dim, N=N_dim, K=K_dim,
        stride_am=a_tensor.stride(0), stride_ak=a_tensor.stride(1),
        stride_bk=b_tensor.stride(0), stride_bn=b_tensor.stride(1),
        stride_cm=c_buffer.stride(0), stride_cn=c_buffer.stride(1),
        BLOCK_M=M_dim, BLOCK_N=N_dim, BLOCK_K=K_dim, 
        OUTPUT_F16=output_f16_flag,
        num_warps=num_warps_launch
        # num_ctas=num_ctas_launch # Launch hint, not JIT param
    )
    return c_buffer

def calculate_tma_gemm_tflops(params: dict, ms: float) -> float:
    M, N, K = params['M'], params['N'], params['K']
    flops = 2 * M * N * K 
    tflops = flops / (ms / 1000) / 1e12
    return tflops

    
def calculate_tma_gemm_gbps(params: dict, ms: float) -> float:
    M, N, K = params['M'], params['N'], params['K']
    
    # Input dtype is fp16 as per test setup
    input_dtype_for_calc = torch.float16
    # Output C can be fp16 or fp32
    output_dtype_str = "fp16" if params['OUTPUT_F16'] else "fp32"
    
    output_torch_dtype_for_calc = torch.float16
    if output_dtype_str == 'fp32': output_torch_dtype_for_calc = torch.float32
    # No bf16 in this test's parametrize for output

    in_element_size = torch.tensor([], dtype=input_dtype_for_calc).element_size()
    out_element_size = torch.tensor([], dtype=output_torch_dtype_for_calc).element_size()
    
    bytes_a = M * K * in_element_size
    bytes_b = K * N * in_element_size 
    bytes_c_write = M * N * out_element_size
    total_bytes = bytes_a + bytes_b + bytes_c_write
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

OP_NAME_FOR_BENCHMARK = "tma_store_gemm_triton_perf"

TMA_GEMM_PERF_PARAMS = [ 
    [64, 64, 16, 1, 4, False, True, False], [64, 64, 16, 1, 4, False, True, True],
    [128, 64, 32, 1, 4, False, True, False], [128, 64, 32, 1, 4, False, True, True],
    [64, 128, 32, 1, 4, False, True, False], [64, 128, 32, 1, 4, False, True, True],
    [128, 128, 64, 1, 4, False, True, False], [128, 128, 64, 1, 4, False, True, True],
    [64, 64, 128, 1, 4, False, False, True], 
    [64, 64, 128, 1, 4, False, False, False],
    [32, 32, 256, 1, 4, False, False, True],
]

@pytest.mark.parametrize('M,N,K,NUM_CTAS_param,NUM_WARPS_param,TRANS_A,TRANS_B,OUTPUT_F16_param', TMA_GEMM_PERF_PARAMS)
# @pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason="Requires compute capability >= 9")
def test_performance(M, N, K, NUM_CTAS_param, NUM_WARPS_param, TRANS_A, TRANS_B, OUTPUT_F16_param, request):
    cap = torch.cuda.get_device_capability()
    if cap[0] < 9: 
        pytest.skip("Requires compute capability >= 9")

    # --- CORRECTED Shared memory skip logic ---
    input_torch_dtype_for_smem_check = torch.float16 # Inputs A and B are fp16 in this test
    element_size_bytes_for_smem_check = torch.tensor([], dtype=input_torch_dtype_for_smem_check).element_size()
    
    smem_elements_needed = (M * K) + (K * N) # For one dot A(M,K) @ B(K,N)
    if smem_elements_needed * element_size_bytes_for_smem_check > 65536: # 64KB limit
        pytest.skip(f"Skipping M{M}N{N}K{K} due to estimated shared memory for inputs: "
                    f"{smem_elements_needed * element_size_bytes_for_smem_check} > 65536")
    # --- End of corrected skip logic ---

    set_seed()
        
    # Input setup from original test, input dtype is fp16
    input_torch_dtype = torch.float16
    a_orig_shape = (K, M) if TRANS_A else (M, K)
    b_orig_shape = (N, K) if TRANS_B else (K, N) # If TRANS_B, B is (N,K), so B.T is (K,N) for matmul
    
    a = torch.randn(a_orig_shape, device='cuda', dtype=input_torch_dtype)
    if TRANS_A: a = a.T 
    
    b = torch.randn(b_orig_shape, device='cuda', dtype=input_torch_dtype)
    if TRANS_B: b = b.T 

    c_out_torch_dtype = torch.float16 if OUTPUT_F16_param else torch.float32
    c_buffer = torch.empty((M, N), device=a.device, dtype=c_out_torch_dtype)

    op_lambda = lambda: tma_gemm_triton_wrapper(
        a, b, c_buffer, M, N, K, 
        OUTPUT_F16_param, NUM_WARPS_param
    )

    bench_config = do_bench_config(warm_up=25, repetition=100)
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)
    current_params_for_logs_and_calc = {
        "M": M, "N": N, "K": K, 
        "NUM_CTAS": NUM_CTAS_param, "NUM_WARPS": NUM_WARPS_param,
        "TRANS_A": TRANS_A, "TRANS_B": TRANS_B, 
        "OUTPUT_F16": OUTPUT_F16_param, # This is for the calculator
        "input_dtype_str": "fp16" # For the calculator
    }
    
    perf_result = benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
                                            gbps_calculator=calculate_tma_gemm_gbps,
                                            tflops_calculator=calculate_tma_gemm_tflops)

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