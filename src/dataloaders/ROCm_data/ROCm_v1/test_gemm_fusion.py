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


######################################## Imports#######################################
import pytest
import torch

import triton
import triton.language as tl

######################################## Imports#######################################



@triton.jit
def gemm_fusion_kernel(A, B, C, E,  #
                       M, N, K,  #
                       stride_am, stride_ak, stride_bn, stride_bk, stride_cn, stride_ck, stride_em, stride_ek,  #
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)

    a_tile_ptr = tl.make_block_ptr(base=A, shape=(M, K), strides=(stride_am, stride_ak), offsets=(pid * BLOCK_M, 0),
                                   block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_tile_ptr = tl.make_block_ptr(base=B, shape=(N, K), strides=(stride_bn, stride_bk), offsets=(0, 0),
                                   block_shape=(BLOCK_N, BLOCK_K), order=(1, 0))
    c_tile_ptr = tl.make_block_ptr(base=C, shape=(N, K), strides=(stride_cn, stride_ck), offsets=(0, 0),
                                   block_shape=(BLOCK_N, BLOCK_K), order=(1, 0))
    e_tile_ptr = tl.make_block_ptr(base=E, shape=(M, K), strides=(stride_em, stride_ek), offsets=(pid * BLOCK_M, 0),
                                   block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))

    acc_e = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    a = tl.load(a_tile_ptr)
    for i in range(0, N, BLOCK_N):
        b = tl.load(b_tile_ptr)
        o_ab = tl.dot(a, tl.trans(b))
        c = tl.load(c_tile_ptr)
        o_ab = o_ab.to(tl.float16)
        acc_e += tl.dot(o_ab, c)
        b_tile_ptr = tl.advance(b_tile_ptr, [BLOCK_N, 0])
        c_tile_ptr = tl.advance(c_tile_ptr, [BLOCK_N, 0])

    acc_e = acc_e.to(tl.float16)
    tl.store(e_tile_ptr, acc_e)

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


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason="not passed on ampere")
def test_gemm_fusion(request):
    set_seed()


    M, N, K = 4096, 4096, 64
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    A = torch.empty((M, K), dtype=torch.float16, device='cuda').normal_(mean=0.1, std=0.2)
    B = torch.empty((N, K), dtype=torch.float16, device='cuda').normal_(mean=0.1, std=0.2)
    C = torch.empty((N, K), dtype=torch.float16, device='cuda').normal_(mean=0.1, std=0.2)
    E = torch.empty((M, K), dtype=torch.float16, device='cuda')
    ref_out = torch.matmul(torch.matmul(A, B.T), C)
    num_warps = 4
    grid = (triton.cdiv(M, BLOCK_M), 1)
    gemm_fusion_kernel[grid](
        A, B, C, E, M, N, K,  #
        A.stride(0), A.stride(1),  #
        B.stride(0), B.stride(1),  #
        C.stride(0), C.stride(1),  #
        E.stride(0), E.stride(1),  #
        BLOCK_M, BLOCK_N, BLOCK_K,  #
        num_warps=num_warps)

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_") 
    result_gold[sanitized_key_name] = E.clone().detach().cpu()
    ###################################################################

    torch.testing.assert_close(ref_out, E, atol=1e-2, rtol=1e-2)


# --- Python wrapper for the kernel for benchmarking ---
def gemm_fusion_triton_wrapper(A_in, B_in, C_in, E_buffer, M_dim, N_dim, K_dim, 
                               block_m_const, block_n_const, num_warps_launch):
    # K_dim is also block_k_const for this kernel
    block_k_const = K_dim
    
    grid = (triton.cdiv(M_dim, block_m_const), 1) # As per original test
    
    gemm_fusion_kernel[grid](
        A_in, B_in, C_in, E_buffer, M_dim, N_dim, K_dim,
        A_in.stride(0), A_in.stride(1),
        B_in.stride(0), B_in.stride(1),
        C_in.stride(0), C_in.stride(1),
        E_buffer.stride(0), E_buffer.stride(1),
        BLOCK_M=block_m_const, BLOCK_N=block_n_const, BLOCK_K=block_k_const,
        num_warps=num_warps_launch
    )
    return E_buffer

# --- Define TFLOPS and GB/s calculators ---
def calculate_gemm_fusion_tflops(params: dict, ms: float) -> float:
    M, N, K = params['M'], params['N'], params['K']
    # Op1: Intermediate(M,N) = A(M,K) @ B.T(K,N) -> 2 * M * N * K
    # Op2: Out(M,K) = Intermediate(M,N) @ C(N,K) -> 2 * M * K * N 
    flops = 2 * M * N * K + 2 * M * K * N 
    tflops = flops / (ms / 1000) / 1e12
    return tflops

def calculate_gemm_fusion_gbps(params: dict, ms: float) -> float:
    M, N, K = params['M'], params['N'], params['K']
    dtype_str = params.get('dtype_str', 'fp16')
    current_dtype = torch.float16
    if dtype_str == 'fp32': current_dtype = torch.float32
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16
    element_size = torch.tensor([], dtype=current_dtype).element_size()

    bytes_a = M * K * element_size
    bytes_b = N * K * element_size # B is (N,K)
    bytes_c_read = N * K * element_size # C is (N,K)
    bytes_e_write = M * K * element_size # E is (M,K)
    total_bytes = bytes_a + bytes_b + bytes_c_read + bytes_e_write
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

OP_NAME_FOR_BENCHMARK = "gemm_fusion_triton_perf"

# --- Pytest parametrize for performance testing ---
# K must be small enough for block_k = K to fit in shared memory for BOTH dots.
# Constraint approx (fp16): block_m*K + 2*K*block_n + block_m*block_n <= 32768
GEMM_FUSION_PERF_CONFIGS = []
# K = 32 (Very safe)
k_val = 32
for M_val in [128, 256, 512, 1024, 2048, 4096]:
    for N_val in [128, 256, 512, 1024, 2048, 4096]:
        for bm_val in [32, 64, 128]: # block_m
            if M_val % bm_val == 0:
                for bn_val in [32, 64, 128]: # block_n
                    if (bm_val*k_val + 2*k_val*bn_val + bm_val*bn_val) <= 32768:
                        GEMM_FUSION_PERF_CONFIGS.append({'M': M_val, 'N': N_val, 'K': k_val, 'bm': bm_val, 'bn': bn_val})
# K = 64
k_val = 64
for M_val in [128, 256, 512, 1024, 2048]: # Reduced M for larger K
    for N_val in [128, 256, 512, 1024]:   # Reduced N for larger K
        for bm_val in [32, 64, 128]:
            if M_val % bm_val == 0:
                for bn_val in [32, 64]: # Smaller block_n for K=64
                    if (bm_val*k_val + 2*k_val*bn_val + bm_val*bn_val) <= 32768:
                        GEMM_FUSION_PERF_CONFIGS.append({'M': M_val, 'N': N_val, 'K': k_val, 'bm': bm_val, 'bn': bn_val})
# K = 128 (Requires smaller block_m, block_n)
k_val = 128
for M_val in [64, 128, 256, 512]:
    for N_val in [64, 128, 256]:
        for bm_val in [16, 32, 64]:
            if M_val % bm_val == 0:
                for bn_val in [16, 32]:
                    if (bm_val*k_val + 2*k_val*bn_val + bm_val*bn_val) <= 32768:
                        GEMM_FUSION_PERF_CONFIGS.append({'M': M_val, 'N': N_val, 'K': k_val, 'bm': bm_val, 'bn': bn_val})

unique_configs = []
seen_configs_str = set()
for cfg in GEMM_FUSION_PERF_CONFIGS:
    cfg_str = f"M{cfg['M']}_N{cfg['N']}_K{cfg['K']}_bm{cfg['bm']}_bn{cfg['bn']}"
    if cfg_str not in seen_configs_str:
        unique_configs.append(cfg)
        seen_configs_str.add(cfg_str)
GEMM_FUSION_PERF_CONFIGS = unique_configs
print(f"Generated {len(GEMM_FUSION_PERF_CONFIGS)} unique performance test configurations for gemm_fusion.")

GEMM_FUSION_DTYPES_FOR_PERF = ['fp16'] # Original test uses fp16
# NUM_WARPS_FOR_PERF = [4, 8] # Original test used 4, can parametrize if needed

@pytest.mark.parametrize("test_params_dict", GEMM_FUSION_PERF_CONFIGS)
@pytest.mark.parametrize("dtype_str", GEMM_FUSION_DTYPES_FOR_PERF)
# @pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason="not passed on ampere") # Original skip
def test_performance(test_params_dict, dtype_str, request):
    # Apply capability skip if needed, e.g. for specific dtypes or features
    # if torch.cuda.get_device_capability()[0] < 8 and dtype_str == 'bf16':
    #     pytest.skip("bf16 requires Ampere+")

    set_seed()
    M = test_params_dict['M']
    N = test_params_dict['N']
    K = test_params_dict['K']
    block_m = test_params_dict['bm']
    block_n = test_params_dict['bn']
    
    block_k_const = K # Kernel requires BLOCK_K == K

    if dtype_str == 'fp32': current_dtype = torch.float32
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16
    else: current_dtype = torch.float16

    A = torch.randn((M, K), dtype=current_dtype, device='cuda')
    B = torch.randn((N, K), dtype=current_dtype, device='cuda')
    C_mat = torch.randn((N, K), dtype=current_dtype, device='cuda')
    E_buffer = torch.empty((M, K), dtype=current_dtype, device='cuda')
    
    num_warps_launch = 4 # As in original test_gemm_fusion

    op_lambda = lambda: gemm_fusion_triton_wrapper(
        A, B, C_mat, E_buffer, M, N, K,
        block_m, block_n, num_warps_launch
    )

    bench_config = do_bench_config(warm_up=25, repetition=100)
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)
    current_params_for_logs_and_calc = {
        "M": M, "N": N, "K": K,
        "block_m": block_m, "block_n": block_n, "block_k": block_k_const,
        "dtype_str": dtype_str, "num_warps": num_warps_launch
    }
    
    perf_result = benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
                                            gbps_calculator=calculate_gemm_fusion_gbps,
                                            tflops_calculator=calculate_gemm_fusion_tflops)



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