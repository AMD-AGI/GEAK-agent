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
import itertools
import os
import re

import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl

######################################## Imports ######################################## 



@triton.jit
def matmul_no_scf_kernel(a_ptr, b_ptr, c_ptr,  #
                         M, N, K,  #
                         stride_am, stride_ak,  #
                         stride_bk, stride_bn,  #
                         stride_cm, stride_cn,  #
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
                         FLOAT16_OUTPUT: tl.constexpr, USE_TMA_EPILOGUE: tl.constexpr  #
                         ):
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, 0),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )
    a = tl.load(a_block_ptr)
    b = tl.load(b_block_ptr)

    c = tl.dot(a, b)

    if FLOAT16_OUTPUT:
        c = c.to(tl.float16)

    if USE_TMA_EPILOGUE:
        c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn), offsets=(0, 0),
                                        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
        tl.store(c_block_ptr, c)
    else:
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, c)

##################################################################################################################################################  

import numpy as np
import random
import torch 
import os
import pytest
from numpy.random import RandomState
import itertools
import re

from torch.testing import assert_close
from tb_eval.perf.ROCm.performance_utils_pytest import PytestBenchmarker, do_bench_config, save_all_benchmark_results
from typing import Dict
import triton
import triton.language as tl


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

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


######################################## HELPERS for Eval ######################################## 




@pytest.mark.parametrize(
    'M,N,K,NUM_CTAS,NUM_WARPS,TRANS_A,TRANS_B,OUTPUT_TYPE,USE_TMA_EPILOGUE',
    itertools.chain(*[[
        # numCTAs = 1, no TMA multicast:
        [64, 16, 16, 1, 4, False, True, "float16", USE_TMA_EPILOGUE],
        [64, 32, 16, 1, 4, False, True, "float16", USE_TMA_EPILOGUE],
        [64, 64, 16, 1, 4, False, True, "float16", USE_TMA_EPILOGUE],
        [64, 64, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
        [64, 64, 32, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
        [64, 64, 64, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
        [128, 128, 16, 1, 4, False, True, "float16", USE_TMA_EPILOGUE],
        [128, 128, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
        # static mask, cluster 4x1
        [256, 64, 16, 4, 4, False, True, "float16", USE_TMA_EPILOGUE],
        [256, 64, 16, 4, 4, False, True, "float32", USE_TMA_EPILOGUE],
        # dynamic mask, cluster 2x2
        [128, 128, 16, 4, 4, False, True, "float16", USE_TMA_EPILOGUE],
        [128, 128, 16, 4, 4, False, True, "float32", USE_TMA_EPILOGUE],
        # small M, N
        [16, 16, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
        [16, 32, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
        [32, 16, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
        [32, 32, 16, 1, 4, False, True, "float32", USE_TMA_EPILOGUE],
    ] for USE_TMA_EPILOGUE in [True, False]]))
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason="Requires compute capability >= 9")
def test_gemm_no_scf(M, N, K, NUM_CTAS, NUM_WARPS, TRANS_A, TRANS_B, OUTPUT_TYPE, USE_TMA_EPILOGUE, request):
    set_seed()
    
    if is_hip() and NUM_CTAS > 1:
        pytest.skip("NUM_CTAS > 1 is not supported in HIP backend")

    if (TRANS_A):
        a = torch.randn((K, M), device='cuda', dtype=torch.float16).T
    else:
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    if (TRANS_B):
        b = torch.randn((N, K), device='cuda', dtype=torch.float16).T
    else:
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)

    if OUTPUT_TYPE == "float16":
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    else:
        c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    matmul_no_scf_kernel[(1, 1)](
        a_ptr=a, b_ptr=b, c_ptr=c,  #
        M=M, N=N, K=K,  #
        stride_am=a.stride(0), stride_ak=a.stride(1),  #
        stride_bk=b.stride(0), stride_bn=b.stride(1),  #
        stride_cm=c.stride(0), stride_cn=c.stride(1),  #
        BLOCK_M=M, BLOCK_N=N, BLOCK_K=K,  #
        num_warps=NUM_WARPS,  #
        num_ctas=NUM_CTAS,  #
        FLOAT16_OUTPUT=(OUTPUT_TYPE == "float16"),  #
        USE_TMA_EPILOGUE=USE_TMA_EPILOGUE)
    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)
    golden = torch.matmul(a_f32, b_f32)
    torch.set_printoptions(profile="full")

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = c.clone().detach().cpu()
    ###################################################################


    assert_close(c, golden, rtol=1e-2, atol=1e-3, check_dtype=False)


def gemm_no_scf_triton_wrapper(a_tensor, b_tensor, c_buffer, 
                               M_dim, N_dim, K_dim, 
                               num_warps_launch, float16_out_flag, use_tma_flag):
    grid = (1,) 
    matmul_no_scf_kernel[grid](
        a_ptr=a_tensor, b_ptr=b_tensor, c_ptr=c_buffer,
        M=M_dim, N=N_dim, K=K_dim,
        stride_am=a_tensor.stride(0), stride_ak=a_tensor.stride(1),
        stride_bk=b_tensor.stride(0), stride_bn=b_tensor.stride(1),
        stride_cm=c_buffer.stride(0), stride_cn=c_buffer.stride(1),
        BLOCK_M=M_dim, BLOCK_N=N_dim, BLOCK_K=K_dim, 
        FLOAT16_OUTPUT=float16_out_flag,
        USE_TMA_EPILOGUE=use_tma_flag,
        num_warps=num_warps_launch
    )
    return c_buffer

def calculate_gemm_no_scf_tflops(params: dict, ms: float) -> float:
    M, N, K = params['M'], params['N'], params['K']
    flops = 2 * M * N * K 
    tflops = flops / (ms / 1000) / 1e12
    return tflops


def calculate_gemm_no_scf_gbps(params: dict, ms: float) -> float:
    M, N, K = params['M'], params['N'], params['K']
    
    input_dtype_str = params.get('input_dtype_str', 'fp16') 
    output_dtype_str = params['OUTPUT_TYPE'] 

    # Convert dtype strings to torch.dtype objects
    current_input_dtype = torch.float16 
    if input_dtype_str == 'fp32': current_input_dtype = torch.float32
    elif input_dtype_str == 'bf16': current_input_dtype = torch.bfloat16

    current_output_dtype = torch.float16 
    if output_dtype_str == 'fp32': current_output_dtype = torch.float32
    elif output_dtype_str == 'bf16': current_output_dtype = torch.bfloat16
    
    # Get element sizes by creating dummy tensors
    in_element_size = torch.tensor([], dtype=current_input_dtype).element_size()
    out_element_size = torch.tensor([], dtype=current_output_dtype).element_size()
    
    bytes_a = M * K * in_element_size
    bytes_b = K * N * in_element_size 
    bytes_c_write = M * N * out_element_size
    total_bytes = bytes_a + bytes_b + bytes_c_write
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

OP_NAME_FOR_BENCHMARK = "gemm_no_scf_triton_perf"

# --- NEW Performance Test Function using original test_gemm_no_scf's parametrization ---
@pytest.mark.parametrize(
    'M,N,K,NUM_CTAS,NUM_WARPS,TRANS_A,TRANS_B,OUTPUT_TYPE,USE_TMA_EPILOGUE',
    # Using a smaller, more targeted subset for performance to avoid excessive shared memory issues
    # and long runtimes. The original parametrize is very extensive.
    # Focus on K values that are likely to fit for a single-block kernel.
    itertools.chain(*[[
        # K=16
        [64, 64, 16, 1, 4, False, False, "float16", USE_TMA_EPILOGUE], # A(M,K), B(K,N)
        [64, 64, 16, 1, 4, False, False, "float32", USE_TMA_EPILOGUE],
        [128, 128, 16, 1, 4, False, False, "float16", USE_TMA_EPILOGUE],
        # K=32
        [64, 64, 32, 1, 4, False, False, "float16", USE_TMA_EPILOGUE],
        [64, 64, 32, 1, 4, False, False, "float32", USE_TMA_EPILOGUE],
        [128, 64, 32, 1, 4, False, False, "float16", USE_TMA_EPILOGUE], # M=128, N=64, K=32
        # K=64
        [32, 32, 64, 1, 4, False, False, "float16", USE_TMA_EPILOGUE], # M=32, N=32, K=64
        [64, 64, 64, 1, 4, False, False, "float16", USE_TMA_EPILOGUE],
        [64, 64, 64, 1, 4, False, False, "float32", USE_TMA_EPILOGUE],
        # K=128 - starts to get large for single block shared mem
        [32, 32, 128, 1, 4, False, False, "float16", USE_TMA_EPILOGUE],
        # [64, 64, 128, 1, 4, False, False, "float16", USE_TMA_EPILOGUE], # (32*128+128*32)*2 = 16384. Ok.
                                                                        # (64*128+128*64)*2 = 32768. Ok.
    ] for USE_TMA_EPILOGUE in [True, False]]))
# @pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason="Requires compute capability >= 9")
def test_performance(M, N, K, NUM_CTAS, NUM_WARPS, TRANS_A, TRANS_B, OUTPUT_TYPE, USE_TMA_EPILOGUE, request):
    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:
        pytest.skip("Requires compute capability >= 9 (as per original test)")
    if is_hip() and NUM_CTAS > 1: 
        pytest.skip("NUM_CTAS > 1 is not supported in HIP for this test (as per original)")

    # Shared memory check (for fp16 inputs)
    input_element_size = 2 # Assuming fp16 inputs
    smem_elements_needed = (M * K) + (K * N)
    if smem_elements_needed * input_element_size > 65536: # 64KB limit
        pytest.skip(f"Skipping M{M}N{N}K{K} due to estimated shared memory for inputs: "
                    f"{smem_elements_needed * input_element_size} > 65536")

    set_seed()
    
    input_torch_dtype = torch.float16 # Original test uses fp16 for a and b

    # Input setup: Ensure a is (M,K) and b is (K,N) before passing to wrapper
    a_shape_before_trans = (K, M) if TRANS_A else (M, K)
    b_shape_before_trans = (N, K) if TRANS_B else (K, N)
    
    a_host = torch.randn(a_shape_before_trans, device='cuda', dtype=input_torch_dtype)
    if TRANS_A: a_host = a_host.T 
    
    b_host = torch.randn(b_shape_before_trans, device='cuda', dtype=input_torch_dtype)
    if TRANS_B: b_host = b_host.T 

    c_out_torch_dtype = torch.float16 if OUTPUT_TYPE == "float16" else torch.float32
    c_buffer = torch.empty((M, N), device=a_host.device, dtype=c_out_torch_dtype)

    op_lambda = lambda: gemm_no_scf_triton_wrapper(
        a_host, b_host, c_buffer, M, N, K, 
        NUM_WARPS, (OUTPUT_TYPE == "float16"), USE_TMA_EPILOGUE
    )

    bench_config = do_bench_config(warm_up=25, repetition=100)
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)
    current_params_for_logs_and_calc = {
        "M": M, "N": N, "K": K, "NUM_CTAS": NUM_CTAS, "NUM_WARPS": NUM_WARPS,
        "TRANS_A": TRANS_A, "TRANS_B": TRANS_B, 
        "OUTPUT_TYPE": OUTPUT_TYPE, "USE_TMA_EPILOGUE": USE_TMA_EPILOGUE,
        "input_dtype_str": "fp16" # Hardcoded based on original test's a,b creation
    }
    
    perf_result = benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
                                            gbps_calculator=calculate_gemm_no_scf_gbps,
                                            tflops_calculator=calculate_gemm_no_scf_tflops)


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