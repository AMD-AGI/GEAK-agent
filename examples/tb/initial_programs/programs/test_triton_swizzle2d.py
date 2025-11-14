# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
######################################## Imports ########################################
import pytest
import torch
import triton
import triton.language as tl
# numpy is not strictly needed here if output is directly compared
######################################## Imports ########################################

@triton.jit
def swizzle2d_kernel(output, size_i, size_j, size_g):
    for i in tl.range(0, size_i, 1):
        for j in tl.range(0, size_j, 1):
            new_i, new_j = tl.swizzle2d(i, j, size_i, size_j, size_g)
            tl.store(output + new_i * size_j + new_j, i * size_j + j)

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





@pytest.mark.interpreter
@pytest.mark.parametrize("size_i, size_j, size_g", [[5, 7, 3]])
def test_swizzle2d(size_i, size_j, size_g, request, device='cuda'):
    # Output tensor to store results, initialized to a value like -1 to see what's written

    set_seed()
    
    output = torch.zeros(size_i, size_j).to(device)
    swizzle2d_kernel[(1, )](output, size_i, size_j, size_g)
    expected_order = torch.tensor([[0, 3, 6, 9, 12, 15, 18], [1, 4, 7, 10, 13, 16, 19], [2, 5, 8, 11, 14, 17, 20],
                                   [21, 23, 25, 27, 29, 31, 33], [22, 24, 26, 28, 30, 32, 34]]).to(device)
    
    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])

    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = output.clone().detach().cpu()
    ################################################################### 

    assert (output == expected_order).all(), (output, expected_order)


# --- Python wrapper for the kernel for benchmarking ---
def swizzle2d_triton_wrapper(output_buffer, size_i_k, size_j_k, size_g_k, num_warps_launch):
    grid = (1,) # Kernel is sequential, not tiled by program_id
    swizzle2d_kernel[grid](
        output_buffer, 
        size_i_k, size_j_k, size_g_k, # Runtime args to kernel
        # num_warps=num_warps_launch # Launch hint
    )
    return output_buffer

# --- Define TFLOPS and GB/s calculators ---
def calculate_swizzle2d_tflops(params: dict, ms: float) -> float:
    # tl.swizzle2d involves bitwise ops (shifts, xors, ands) and some arithmetic.
    # A rough estimate might be ~10-20 simple ops per (i,j) pair.
    # Number of pairs = size_i * size_j
    size_i = params['size_i_kernel']
    size_j = params['size_j_kernel']
    ops_per_swizzle = 15 # Rough estimate
    total_ops = size_i * size_j * ops_per_swizzle
    tflops = total_ops / (ms / 1000) / 1e12 # These are integer ops, not float ops
    return tflops # Reporting "integer TOPS" effectively

def calculate_swizzle2d_gbps(params: dict, ms: float) -> float:
    size_i = params['size_i_kernel']
    size_j = params['size_j_kernel']
    dtype_str = params.get('output_dtype_str', 'int32') # Kernel stores integers

    current_dtype = torch.int32 # Default
    if dtype_str == 'int64': current_dtype = torch.int64
    # Add other int types if parametrized
    element_size = torch.tensor([], dtype=current_dtype).element_size()

    # Writes size_i * size_j elements to output. No global memory reads for input data.
    elements_written = size_i * size_j
    total_bytes = elements_written * element_size
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

OP_NAME_FOR_BENCHMARK = "triton_swizzle2d_perf"

# --- Pytest parametrize for performance testing ---
# Kernel processes one tile of size_i x size_j sequentially.
SWIZZLE_SHAPES_FOR_PERF = [
    # size_i_kernel, size_j_kernel, size_g_kernel
    (32, 32, 4), (64, 64, 8), (128, 128, 16),
    (256, 256, 16), (256, 256, 32),
    (512, 512, 32), #(512, 512, 64) -> size_g should be < size_i, size_j for typical swizzle
    (512, 64, 8), (64, 512, 8)
]
SWIZZLE_DTYPES_FOR_PERF = ['int32', 'int64'] # Output tensor dtype
# NUM_WARPS_FOR_PERF = [1, 2, 4] # Kernel is sequential, num_warps might not have big impact

@pytest.mark.parametrize("size_i_k, size_j_k, size_g_k", SWIZZLE_SHAPES_FOR_PERF)
@pytest.mark.parametrize("output_dtype_str", SWIZZLE_DTYPES_FOR_PERF)
# @pytest.mark.parametrize("num_warps_launch", NUM_WARPS_FOR_PERF)
def test_performance(size_i_k, size_j_k, size_g_k, output_dtype_str, request, device='cuda'):
    # num_warps_launch = 4 # Or from parametrize
    set_seed()
    
    # Ensure size_g is valid for swizzle2d (typically power of 2, < size_i, size_j)
    # tl.swizzle2d doesn't strictly require power of 2 for size_g but it's common.
    # It does require size_g > 0.
    if not (size_g_k > 0 and size_g_k <= size_i_k and size_g_k <= size_j_k):
        pytest.skip(f"Invalid size_g={size_g_k} for tile ({size_i_k}, {size_j_k})")

    if output_dtype_str == 'int64': current_out_dtype = torch.int64
    else: current_out_dtype = torch.int32
        
    # Output buffer. Kernel stores original flattened indices (integers).
    output_buffer = torch.empty((size_i_k * size_j_k,), dtype=current_out_dtype, device=device)
    # The kernel writes to output_ptr + new_i * size_j + new_j.
    # If output_ptr is 1D, this is fine. If output_ptr is 2D, strides are needed.
    # The original test passed a 2D tensor to kernel. Let's match that.
    # Kernel expects output_ptr to be a flat pointer essentially, and calculates 2D offsets.
    # So, a 1D buffer for output_ptr is fine if kernel does `output_ptr + flat_offset`.
    # The kernel does `output_ptr + new_i * size_j + new_j`. This implies `output_ptr` is base of 2D array.
    # So, `output_buffer` should be 2D for the kernel call.
    output_buffer_2d = torch.empty((size_i_k, size_j_k), dtype=current_out_dtype, device=device)


    op_lambda = lambda: swizzle2d_triton_wrapper(
        output_buffer_2d, # Pass the 2D buffer
        size_i_k, size_j_k, size_g_k,
        num_warps_launch=4 
    )

    # This kernel is very fast for small sizes, might need many reps.
    bench_config = do_bench_config(warm_up=100, repetition=1000 if size_i_k*size_j_k < 256*256 else 200) 
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)
    current_params_for_logs_and_calc = {
        "size_i_kernel": size_i_k, 
        "size_j_kernel": size_j_k,
        "size_g_kernel": size_g_k,
        "output_dtype_str": output_dtype_str,
        "num_warps": 4 
    }
    
    perf_result = benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
                                            gbps_calculator=calculate_swizzle2d_gbps,
                                            tflops_calculator=calculate_swizzle2d_tflops)


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