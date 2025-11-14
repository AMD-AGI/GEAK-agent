# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
######################################## Imports ######################################## 
import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl

dtype_mapping = {
    'float16': torch.float16,
    'float32': torch.float32,
}
######################################## Imports ######################################## 



@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_block_ptr = tl.make_block_ptr(base=x_ptr, shape=(n_elements, ), strides=(1, ), offsets=(pid * BLOCK_SIZE, ),
                                    block_shape=(BLOCK_SIZE, ), order=(0, ))
    x = tl.load(x_block_ptr, boundary_check=(0, ), padding_option='zero')

    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

##################################################################################################################################################  

import numpy as np
import random
import torch 
import os
from numpy.random import RandomState
import pytest
from torch.testing import assert_close
from geak_eval.perf.ROCm.performance_utils_pytest import PytestBenchmarker, do_bench_config, save_all_benchmark_results
from typing import Dict

import triton
import triton.language as tl

dtype_mapping = {
    'float16': torch.float16,
    'float32': torch.float32,
}

result_gold = {}

######################################## HELPERS for Eval ######################################## 
# Helper function to define GB/s for add_kernel
def calculate_add_gbps(params: Dict, ms: float) -> float:
    # params will contain 'SIZE', 'dtype_str'
    size = params['SIZE']
    dtype = dtype_mapping[params['dtype_str']]
    # For add: read x, read y, write output
    # If x, y, output are torch.Tensor objects passed to this calculator:
    # total_bytes = (x.numel() * x.element_size() +
    #                y.numel() * y.element_size() +
    #                output.numel() * output.element_size())
    # If only params are available:
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    total_bytes = 3 * size * bytes_per_element # 2 reads, 1 write
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

# Helper function to define TFLOPS for add_kernel
def calculate_add_tflops(params: Dict, ms: float) -> float:
    size = params['SIZE']
    # For add: N operations (N additions)
    flops = size
    tflops = flops / (ms / 1000) / 1e12
    return tflops

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




@pytest.mark.parametrize('SIZE,BLOCK_SIZE,dtype_str',
                         [(98432, 1024, dtype_str) for dtype_str in ['float16', 'float32']])
def test_add(SIZE, BLOCK_SIZE, dtype_str, request):
    set_seed()

    dtype = dtype_mapping[dtype_str]
    output = torch.empty(SIZE, device='cuda', dtype=dtype)
    x = torch.randn(SIZE, device='cuda', dtype=dtype)
    y = torch.randn(SIZE, device='cuda', dtype=dtype)

    def grid(meta):
        return (triton.cdiv(SIZE, meta['BLOCK_SIZE']), )

    add_kernel[grid](x, y, output, SIZE, BLOCK_SIZE=BLOCK_SIZE)

    output_torch = x + y
    torch.set_printoptions(profile='full')

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = output.clone().detach().cpu()
    ################################################################### 

    assert_close(output, output_torch, rtol=1e-2, atol=1e-3, check_dtype=False)


OP_NAME_FOR_BENCHMARK = "add_kernel_perf"

@pytest.mark.parametrize('SIZE,BLOCK_SIZE_ARG,dtype_str', # BLOCK_SIZE_ARG is the pytest param name
                         [(98432, 1024, dtype_str) for dtype_str in ['float16', 'float32']] +
                         [(1048576, 2048, dtype_str) for dtype_str in ['float16', 'float32']]
                        )
def test_performance(SIZE, BLOCK_SIZE_ARG, dtype_str, request): # Function accepts BLOCK_SIZE_ARG
    set_seed()
    dtype = dtype_mapping[dtype_str]
    x = torch.randn(SIZE, device='cuda', dtype=dtype)
    y = torch.randn(SIZE, device='cuda', dtype=dtype)
    output = torch.empty(SIZE, device='cuda', dtype=dtype)

    # Kernel launch grid
    # The 'meta' dict passed to the grid lambda by Triton contains the constexpr arguments
    # that were passed to the kernel launch.
    # When we call `add_kernel[grid](..., BLOCK_SIZE=BLOCK_SIZE_ARG)`,
    # the `meta` dict will have a key 'BLOCK_SIZE' (the name of the constexpr in the kernel signature)
    # and its value will be the runtime `BLOCK_SIZE_ARG`.
    grid = lambda meta: (triton.cdiv(SIZE, meta['BLOCK_SIZE']),) # ***** CORRECTED HERE *****

    kernel_args = [x, y, output, SIZE]
    
    # The op_lambda passes BLOCK_SIZE_ARG (runtime value) as the kernel's `BLOCK_SIZE` (constexpr name)
    op_lambda = lambda: add_kernel[grid](*kernel_args, BLOCK_SIZE=BLOCK_SIZE_ARG)

    bench_config = do_bench_config(warm_up=25, repetition=100) # Smaller for faster debug
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)

    # The dictionary passed to calculators should use consistent keys
    current_params_for_calculators = {"SIZE": SIZE, "BLOCK_SIZE_RUNTIME": BLOCK_SIZE_ARG, "dtype_str": dtype_str}
    # Note: I used "BLOCK_SIZE_RUNTIME" here to be explicit that it's the value from parametrize,
    # not necessarily the same as the constexpr name if they differed.
    # If your calculators expect 'BLOCK_SIZE', then use that:
    # current_params_for_calculators = {"SIZE": SIZE, "BLOCK_SIZE": BLOCK_SIZE_ARG, "dtype_str": dtype_str}


    benchmarker.run_benchmark(current_params_dict=current_params_for_calculators,
                              gbps_calculator=calculate_add_gbps,
                              tflops_calculator=calculate_add_tflops)
    
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