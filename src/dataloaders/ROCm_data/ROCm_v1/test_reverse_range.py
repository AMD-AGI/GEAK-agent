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
def reverse_range(in_ptr, out_ptr):
    x0 = tl.arange(0, 512)
    tmp0 = tl.load(in_ptr + (512 - x0))
    tl.store(out_ptr + x0, tmp0)

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



def test_reverse_range(request, device='cuda'):
    set_seed()
    

    data = torch.randn((516, ), dtype=torch.float32, device=device)
    res = torch.empty((512, ), dtype=torch.float32, device=device)
    reverse_range[(1, )](data, res)
    ref = torch.flip(data[1:513], [0])

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = res.clone().detach().cpu()
    ################################################################### 


    assert (res == ref).all()


# --- Python wrapper for the kernel for benchmarking ---
def reverse_range_triton_wrapper(in_tensor_for_kernel, out_buffer, num_warps_launch):
    # The kernel is hardcoded to process 512 elements.
    # in_tensor_for_kernel should be the base pointer from which kernel reads data[1]..data[512]
    # out_buffer is where results are written.
    grid = (1,)
    reverse_range[grid](
        in_tensor_for_kernel, 
        out_buffer,
        num_warps=num_warps_launch # Launch hint
    )
    return out_buffer

# --- Define TFLOPS and GB/s calculators ---
KERNEL_FIXED_SIZE = 512

def calculate_reverse_range_tflops(params: dict, ms: float) -> float:
    # This is a memory copy operation, no significant arithmetic FLOPs.
    return 0.0 

def calculate_reverse_range_gbps(params: dict, ms: float) -> float:
    N_processed = KERNEL_FIXED_SIZE
    dtype_str = params.get('dtype_str', 'fp32') 

    current_dtype = torch.float32
    if dtype_str == 'fp16': current_dtype = torch.float16
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16
    element_size = torch.tensor([], dtype=current_dtype).element_size()

    # Reads N_processed elements, Writes N_processed elements
    bytes_read = N_processed * element_size
    bytes_write = N_processed * element_size 
    total_bytes = bytes_read + bytes_write
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

OP_NAME_FOR_BENCHMARK = "reverse_range_triton_perf"

# --- Pytest parametrize for performance testing ---
# Kernel size is fixed at 512. We can vary dtype and num_warps.
REVERSE_RANGE_DTYPES_FOR_PERF = ['fp32', 'fp16', 'bf16'] 
REVERSE_RANGE_NUM_WARPS_FOR_PERF = [1, 2, 4] 

@pytest.mark.parametrize("dtype_str", REVERSE_RANGE_DTYPES_FOR_PERF)
@pytest.mark.parametrize("num_warps_launch", REVERSE_RANGE_NUM_WARPS_FOR_PERF)
def test_performance(dtype_str, num_warps_launch, request, device='cuda'):
    set_seed()
    
    if dtype_str == 'bf16':
        cap = torch.cuda.get_device_capability()
        if cap[0] < 8:
            pytest.skip("bf16 requires Ampere+ (arch 80+)")
    
    if dtype_str == 'fp32': current_dtype = torch.float32
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16
    else: current_dtype = torch.float16

    # Input tensor `data_perf` needs to be large enough for kernel's access pattern.
    # Kernel reads from in_ptr[1] to in_ptr[512].
    # So, `data_perf` needs at least 513 elements if passed directly.
    # Original test used size 516 for `data`.
    data_perf = torch.randn((KERNEL_FIXED_SIZE + 4, ), dtype=current_dtype, device=device) # e.g. 516 elements
    # Output buffer `res_perf` is size 512.
    res_perf_buffer = torch.empty((KERNEL_FIXED_SIZE, ), dtype=current_dtype, device=device)
    
    op_lambda = lambda: reverse_range_triton_wrapper(
        data_perf, res_perf_buffer, num_warps_launch
    )

    bench_config = do_bench_config(warm_up=100, repetition=1000) # Simple kernel, more reps
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)
    current_params_for_logs_and_calc = {
        "N_processed": KERNEL_FIXED_SIZE, 
        "dtype_str": dtype_str,
        "num_warps": num_warps_launch
    }
    
    perf_result = benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
                                            gbps_calculator=calculate_reverse_range_gbps,
                                            tflops_calculator=calculate_reverse_range_tflops)

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