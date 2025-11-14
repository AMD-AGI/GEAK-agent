# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
######################################## Imports ######################################## 

import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl
######################################## Imports ######################################## 

dtype_mapping = {
    'float16': torch.float16,
    'float32': torch.float32,
}


@triton.jit
def load_reduce_kernel(
    x_ptr,
    y_ptr,
    stride_xm,
    stride_xn,
    stride_y,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    x_ptr = tl.make_block_ptr(base=x_ptr, shape=(BLOCK_M, BLOCK_N), strides=(stride_xm, stride_xn), offsets=(0, 0),
                              block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    x = tl.load(x_ptr)
    y = tl.max(x, axis=1)
    tl.store(y_ptr + tl.arange(0, BLOCK_M), y)


##################################################################################################################################################  

import numpy as np
import random
import torch 
import os
import pytest
from numpy.random import RandomState
from torch.testing import assert_close

import triton
import triton.language as tl

from geak_eval.perf.ROCm.performance_utils_pytest import PytestBenchmarker, do_bench_config, save_all_benchmark_results
from typing import Dict

result_gold = {}
dtype_mapping = {
    'float16': torch.float16,
    'float32': torch.float32,
}

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



@pytest.mark.parametrize('BLOCK_M,BLOCK_N,dtype_str', [(128, 64, dtype_str) for dtype_str in ['float16']])
def test_load_reduce(BLOCK_M, BLOCK_N, dtype_str, request):
    set_seed()

    dtype = dtype_mapping[dtype_str]
    x = torch.randn((BLOCK_M, BLOCK_N), device='cuda', dtype=dtype)
    y = torch.empty((BLOCK_M, ), device='cuda', dtype=dtype)

    load_reduce_kernel[(1, )](x, y, x.stride(0), x.stride(1), y.stride(0), BLOCK_M, BLOCK_N)

    golden = x.max(dim=1)[0]
    torch.set_printoptions(profile='full')

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = y.clone().detach().cpu()
    ################################################################### 

    assert_close(y, golden, rtol=1e-2, atol=1e-3, check_dtype=False)


# --- Python wrapper for the kernel for benchmarking ---
def load_reduce_triton_wrapper(x_tensor, y_buffer, block_m_const, block_n_const):
    # For this kernel, the full dimensions processed are defined by block_m_const, block_n_const
    # The grid is (1,) as the kernel does not use program_id for further tiling.
    grid = (1,)
    
    load_reduce_kernel[grid](
        x_tensor, y_buffer,
        x_tensor.stride(0), x_tensor.stride(1),
        y_buffer.stride(0), # Stride of y (typically 1 for contiguous 1D)
        BLOCK_M=block_m_const, 
        BLOCK_N=block_n_const
        # num_warps can be added as a launch hint if desired
    )
    return y_buffer

# --- Define TFLOPS and GB/s calculators ---
def calculate_load_reduce_tflops(params: dict, ms: float) -> float:
    # Operation: M rows, N elements per row. For each row, N-1 comparisons for max.
    M_dim = params['M_dim'] # This is BLOCK_M for the kernel
    N_dim = params['N_dim'] # This is BLOCK_N for the kernel
    
    flops = M_dim * (N_dim -1) if N_dim > 0 else 0 # Approx N comparisons per row
    tflops = flops / (ms / 1000) / 1e12
    return tflops

def calculate_load_reduce_gbps(params: dict, ms: float) -> float:
    M_dim = params['M_dim']
    N_dim = params['N_dim']
    dtype_str = params.get('dtype_str', 'fp16')

    current_dtype = torch.float16
    if dtype_str == 'fp32': current_dtype = torch.float32
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16
    element_size = torch.tensor([], dtype=current_dtype).element_size()

    # Read X (M,N), Write Y (M)
    bytes_x_read = M_dim * N_dim * element_size
    bytes_y_write = M_dim * element_size 
    total_bytes = bytes_x_read + bytes_y_write
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

OP_NAME_FOR_BENCHMARK = "load_reduce_triton_perf"

# --- Pytest parametrize for performance testing ---
# The kernel processes a single block of size BLOCK_M x BLOCK_N.
# So, M_dim and N_dim for the test will be BLOCK_M and BLOCK_N for the kernel.
LOAD_REDUCE_BLOCK_SHAPES_FOR_PERF = [
    # BLOCK_M, BLOCK_N (these are effectively M_dim, N_dim for the single block kernel)
    (128, 64), (128, 128), (128, 256), (128, 512), (128, 1024),
    (256, 64), (256, 128), (256, 256), (256, 512),
    (512, 64), (512, 128), (512, 256),
    (1024, 64), (1024, 128),
    # (4096, 64) # Larger M
]
LOAD_REDUCE_DTYPES_FOR_PERF = ['fp16', 'fp32', 'bf16'] 
# NUM_WARPS_FOR_PERF = [1, 2, 4, 8] # Can be added as a launch hint

@pytest.mark.parametrize("block_m_const, block_n_const", LOAD_REDUCE_BLOCK_SHAPES_FOR_PERF)
@pytest.mark.parametrize("dtype_str", LOAD_REDUCE_DTYPES_FOR_PERF)
# @pytest.mark.parametrize("num_warps_launch", NUM_WARPS_FOR_PERF) # Optional
def test_performance(block_m_const, block_n_const, dtype_str, request): # Added num_warps_launch if parametrized
    
    # num_warps_launch = 4 # Or from parametrize
    
    # Capability checks for dtypes
    if dtype_str == 'bf16':
        cap = torch.cuda.get_device_capability()
        if cap[0] < 8:
            pytest.skip("bf16 requires Ampere+ (arch 80+)")
    
    set_seed()
    
    if dtype_str == 'fp32': current_dtype = torch.float32
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16
    else: current_dtype = torch.float16

    # Input x has shape (block_m_const, block_n_const) as the kernel processes this single block
    x = torch.randn((block_m_const, block_n_const), device='cuda', dtype=current_dtype)
    # Output y has shape (block_m_const,)
    y_buffer = torch.empty((block_m_const,), device='cuda', dtype=current_dtype)
    
    op_lambda = lambda: load_reduce_triton_wrapper(
        x, y_buffer, block_m_const, block_n_const
        # , num_warps_launch # if num_warps is added to wrapper
    )

    bench_config = do_bench_config(warm_up=50, repetition=200) # Reduction can be fast
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)
    current_params_for_logs_and_calc = {
        "M_dim": block_m_const, "N_dim": block_n_const, # Use M_dim, N_dim for calculators
        "dtype_str": dtype_str,
        # "num_warps": num_warps_launch # if parametrized
    }
    
    perf_result = benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
                                            gbps_calculator=calculate_load_reduce_gbps,
                                            tflops_calculator=calculate_load_reduce_tflops)


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