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
def chained_matmul_kernel(A,  # shape: (m, k)
                            B,  # shape: (n, k)
                            C,  # shape: (n, k)
                            out,  # shape: (m, k)
                            m, n, k: tl.constexpr,  #
                            block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr):

    tl.static_assert(block_k == k, f"expected block_k == k but got {block_k} != {k}")

    block_ix = tl.program_id(0)
    a_tile = (block_ix * block_m + tl.arange(0, block_m))[:, None] * block_k \
        + tl.arange(0, block_k)[None, :]

    a = tl.load(A + a_tile, mask=a_tile < m * k, other=0.0)

    acc = tl.zeros([block_m, block_k], dtype=tl.float32)

    for loop_block_start in range(0, n, block_n):
        bc_tile = (loop_block_start + tl.arange(0, block_n))[:, None] * block_k \
            + tl.arange(0, block_k)[None, :]
        b = tl.load(B + bc_tile, mask=bc_tile < n * k, other=0.0)

        intermediate = tl.dot(a, tl.trans(b))
        intermediate_mask = ((loop_block_start + tl.arange(0, block_n)) < n)[None, :] \
            * (tl.arange(0, block_m) < m)[:, None]

        intermediate = tl.where(intermediate_mask, intermediate, 0.0)

        c = tl.load(C + bc_tile, mask=bc_tile < n * k)

        acc += tl.dot(intermediate.to(A.dtype.element_ty), c)

    tl.store(out + a_tile, acc.to(A.dtype.element_ty), mask=a_tile < m * k)


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


def chained_matmul_reference(a, b, c):
    intermediate = torch.einsum('MK,NK->MN', a, b)
    return torch.einsum('MN,NK->MK', intermediate, c)


def test_chained_matmul(request, device='cuda'):
    # Regression test for issue #1601
    set_seed()


    m, n, k = 32, 64, 128
    block_m, block_n, block_k = 16, 32, k

    grid = (triton.cdiv(m, block_m), )
    a = torch.randint(low=0, high=2, size=(m, k), dtype=torch.float16, device=device)
    b = torch.randint(low=0, high=2, size=(n, k), dtype=torch.float16, device=device)
    c = torch.randint_like(b, low=0, high=2)
    triton_result = torch.zeros_like(a)

    torch_result = chained_matmul_reference(a, b, c)
    chained_matmul_kernel[grid](
        a, b, c, triton_result, m, n, k,  #
        block_m=block_m, block_n=block_n, block_k=block_k)

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = triton_result.clone().detach().cpu()
    ################################################################### 
    assert (torch_result == triton_result).all()

# --- Python wrapper for the kernel ---
def chained_matmul_triton_wrapper(A_in, B_in, C_in, out_buffer, block_m, block_n):
    m, k_a = A_in.shape
    n, k_b = B_in.shape
    _n_c, k_c = C_in.shape # C is also (n,k)
    assert k_a == k_b and k_a == k_c, "K dimensions must match for A, B, C"
    
    # Kernel's k is a constexpr fixed to k_a (full K dimension)
    # Kernel's block_k is also fixed to k_a
    
    grid = (triton.cdiv(m, block_m),) # Grid is 1D, iterating over M blocks
    
    chained_matmul_kernel[grid](
        A_in, B_in, C_in, out_buffer,
        m, n, k_a, # runtime m, n, k
        block_m=block_m, block_n=block_n, block_k=k_a # constexpr block_m, block_n, block_k=k
        # num_warps not in kernel signature, would be for autotune
    )
    return out_buffer

# --- Define TFLOPS and GB/s calculators ---
def calculate_chained_matmul_tflops(params: dict, ms: float) -> float:
    m, n, k = params['M'], params['N'], params['K']
    # Op1: Intermediate(M,N) = A(M,K) @ B.T(K,N) -> 2 * M * N * K FLOPs
    # Op2: Out(M,K) = Intermediate(M,N) @ C(N,K) -> 2 * M * K * N FLOPs (Note: C is (N,K))
    # Total FLOPs = 4 * M * N * K
    flops = 2 * m * n * k + 2 * m * k * n 
    tflops = flops / (ms / 1000) / 1e12
    return tflops

def calculate_chained_matmul_gbps(params: dict, ms: float) -> float:
    m, n, k = params['M'], params['N'], params['K']
    dtype_str = params.get('dtype_str', 'fp16')
    current_dtype = torch.float16
    if dtype_str == 'fp32': current_dtype = torch.float32
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16
    element_size = torch.tensor([], dtype=current_dtype).element_size()
    bytes_a = m * k * element_size
    bytes_b = n * k * element_size
    bytes_c_read = n * k * element_size
    bytes_out_write = m * k * element_size
    total_bytes = bytes_a + bytes_b + bytes_c_read + bytes_out_write
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

OP_NAME_FOR_BENCHMARK = "chained_matmul_triton_perf"

# --- REVISED Pytest parametrize for performance testing ---
# K must be small enough for block_k = K to fit in shared memory.
# Constraint: K * (block_m + block_n) <= 32768 (for fp16, 64KB shared mem limit)

CM_PERF_TEST_CONFIGS = []
# Shared memory constraint (fp16): block_m*K + 2*K*block_n + block_m*block_n <= 32768

# K = 64
k_val = 64
for M_val in [64, 128, 256]:
    for N_val in [64, 128, 256]: # N is the total dimension, looped over by block_n
        for bm_val in [16, 32, 64]:
            if M_val % bm_val == 0:
                for bn_val in [16, 32]: # Keep block_n small
                    if (bm_val*k_val + 2*k_val*bn_val + bm_val*bn_val) <= 32768:
                        CM_PERF_TEST_CONFIGS.append({'M': M_val, 'N': N_val, 'K': k_val, 'bm': bm_val, 'bn': bn_val})
                # Try bn_val = 64 if bm_val is small
                if bm_val <= 32:
                    bn_val = 64
                    if (bm_val*k_val + 2*k_val*bn_val + bm_val*bn_val) <= 32768:
                        CM_PERF_TEST_CONFIGS.append({'M': M_val, 'N': N_val, 'K': k_val, 'bm': bm_val, 'bn': bn_val})


# K = 128
k_val = 128
for M_val in [64, 128, 256]:
    for N_val in [32, 64, 128]: 
        for bm_val in [16, 32]: # Keep block_m smaller for larger K
            if M_val % bm_val == 0:
                for bn_val in [16, 32]: # Keep block_n small
                    if (bm_val*k_val + 2*k_val*bn_val + bm_val*bn_val) <= 32768:
                        CM_PERF_TEST_CONFIGS.append({'M': M_val, 'N': N_val, 'K': k_val, 'bm': bm_val, 'bn': bn_val})
        # Try bm_val = 64 if bn_val is very small
        if M_val % 64 == 0:
            bm_val = 64
            bn_val = 16 
            if (bm_val*k_val + 2*k_val*bn_val + bm_val*bn_val) <= 32768:
                 CM_PERF_TEST_CONFIGS.append({'M': M_val, 'N': N_val, 'K': k_val, 'bm': bm_val, 'bn': bn_val})


# K = 256 (Requires even smaller block_m, block_n)
k_val = 256
for M_val in [32, 64, 128]: 
    for N_val in [16, 32, 64]:  
        for bm_val in [16, 32]: 
            if M_val % bm_val == 0:
                for bn_val in [16]: # block_n must be very small
                    if (bm_val*k_val + 2*k_val*bn_val + bm_val*bn_val) <= 32768:
                        CM_PERF_TEST_CONFIGS.append({'M': M_val, 'N': N_val, 'K': k_val, 'bm': bm_val, 'bn': bn_val})
                # Try bn_val = 32 if bm_val is 16
                if bm_val == 16:
                    bn_val = 32
                    if (bm_val*k_val + 2*k_val*bn_val + bm_val*bn_val) <= 32768: # 16*256 + 2*256*32 + 16*32 = 4096 + 16384 + 512 = 20992 (OK)
                        CM_PERF_TEST_CONFIGS.append({'M': M_val, 'N': N_val, 'K': k_val, 'bm': bm_val, 'bn': bn_val})


# K = 512 (Extremely restrictive for this kernel design) - Likely only 16x16 blocks will work
k_val = 512
for M_val in [16, 32, 64]: 
    for N_val in [16, 32]: 
        for bm_val in [16]: 
            if M_val % bm_val == 0:
                for bn_val in [16]: 
                    # 16*512 + 2*512*16 + 16*16 = 8192 + 16384 + 256 = 24832 (OK)
                    if (bm_val*k_val + 2*k_val*bn_val + bm_val*bn_val) <= 32768:
                        CM_PERF_TEST_CONFIGS.append({'M': M_val, 'N': N_val, 'K': k_val, 'bm': bm_val, 'bn': bn_val})


# Remove duplicates
unique_configs = []
seen_configs_str = set()
for cfg in CM_PERF_TEST_CONFIGS:
    # Create a string representation for checking uniqueness easily
    cfg_str = f"M{cfg['M']}_N{cfg['N']}_K{cfg['K']}_bm{cfg['bm']}_bn{cfg['bn']}"
    if cfg_str not in seen_configs_str:
        unique_configs.append(cfg)
        seen_configs_str.add(cfg_str)
CM_PERF_TEST_CONFIGS = unique_configs

print(f"Generated {len(CM_PERF_TEST_CONFIGS)} unique performance test configurations for chained_matmul.")

# Only fp16 for now to reduce matrix and focus on shared mem issue
CM_DTYPES_FOR_PERF = ['fp16']
# CM_DTYPES_FOR_PERF = ['fp16', 'fp32'] # Add fp32 later if fp16 works


@pytest.mark.parametrize("test_params_dict", CM_PERF_TEST_CONFIGS)
@pytest.mark.parametrize("dtype_str", CM_DTYPES_FOR_PERF)
def test_performance(test_params_dict, dtype_str, request):
    # ... (rest of the test_chained_matmul_performance function remains the same as in the previous response)
    set_seed()
    m = test_params_dict['M']
    n = test_params_dict['N']
    k = test_params_dict['K']
    block_m = test_params_dict['bm']
    block_n = test_params_dict['bn']
    
    block_k_const = k 

    # This skip logic might be redundant if CM_PERF_TEST_CONFIGS is well-filtered
    # element_size_bytes = 4 if dtype_str == 'fp32' else 2
    # shared_mem_limit_elements = 65536 // element_size_bytes
    # estimated_elements_needed = block_m * k + 2 * k * block_n + block_m * block_n
    # if estimated_elements_needed > shared_mem_limit_elements:
    #     pytest.skip(f"Skipping M{m}N{n}K{k} bm{block_m}bn{block_n} dtype {dtype_str} due to estimated shared memory: {estimated_elements_needed*element_size_bytes} vs {65536}")


    if dtype_str == 'fp32': current_dtype = torch.float32
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16
    else: current_dtype = torch.float16

    a = torch.randn((m, k), dtype=current_dtype, device='cuda')
    b = torch.randn((n, k), dtype=current_dtype, device='cuda') 
    c_mat = torch.randn((n, k), dtype=current_dtype, device='cuda')
    triton_result_buffer = torch.empty((m, k), dtype=current_dtype, device='cuda')

    op_lambda = lambda: chained_matmul_triton_wrapper(
        a, b, c_mat, triton_result_buffer,
        block_m, block_n
    )

    bench_config = do_bench_config(warm_up=25, repetition=100)
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)

    current_params_for_logs_and_calc = {
        "M": m, "N": n, "K": k,
        "block_m": block_m, "block_n": block_n, "block_k": block_k_const,
        "dtype_str": dtype_str
    }

    benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
                              gbps_calculator=calculate_chained_matmul_gbps,
                              tflops_calculator=calculate_chained_matmul_tflops)
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