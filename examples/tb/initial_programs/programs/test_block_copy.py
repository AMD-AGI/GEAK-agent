# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
######################################## Imports ######################################## 
import pytest
import torch

import triton
import triton.language as tl
import os
######################################## Imports ######################################## 

@triton.jit
def block_copy_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr, padding_option: tl.constexpr):
    pid = tl.program_id(0)
    # We only copy half of the data to see if the padding works
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(N // 2, ), strides=(1, ), offsets=(pid * BLOCK_SIZE, ),
                                    block_shape=(BLOCK_SIZE, ), order=(0, ))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(N, ), strides=(1, ), offsets=(pid * BLOCK_SIZE, ),
                                    block_shape=(BLOCK_SIZE, ), order=(0, ))
    if padding_option is None:
        a = tl.load(a_block_ptr, boundary_check=(0, ))
    else:
        a = tl.load(a_block_ptr, boundary_check=(0, ), padding_option=padding_option)
    tl.store(b_block_ptr, a, boundary_check=(0, ))

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
def is_interpreter():
    return os.environ.get('TRITON_INTERPRET', '0') == '1'

def check_type_supported(dtype, device='cuda'):
    '''
    skip test if dtype is not supported on the current device
    '''
    if device in ['cuda']:
        cc = torch.cuda.get_device_capability()
        if cc[0] < 8 and (dtype is tl.bfloat16 or dtype == "bfloat16" or dtype is torch.bfloat16):
            pytest.skip("bfloat16 is only supported on NVGPU with cc >= 80")
        if cc[0] < 9 and dtype in {tl.float8e4nv, "float8e4nv", "float8_e4m3fn"}:
            pytest.skip("float8e4nv is only supported on NVGPU with cc >= 90")
    if is_interpreter():
        if dtype in [tl.bfloat16, "bfloat16", torch.bfloat16]:
            pytest.skip("bfloat16 is not supported in the interpreter")


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
@pytest.mark.parametrize("dtypes_str, n, padding_option", [  #
    (dtypes_str, n, padding)
    for dtypes_str in (("bool", "bool"), ("int16", "int16"), ("int32", "int32"), ("float16", "float16"),
                       ("float32", "float32"), ("bfloat16", "bfloat16"))
    for n in (64, 128, 256, 512, 1024)
    for padding in (None, "zero", "nan")  #
])
def test_block_copy(dtypes_str, n, padding_option, request, device='cuda'):
    src_dtype_str = dtypes_str[0]
    dst_dtype_str = dtypes_str[1]
    src_dtype = getattr(torch, src_dtype_str)
    dst_dtype = getattr(torch, dst_dtype_str)
    check_type_supported(src_dtype, device)
    check_type_supported(dst_dtype, device)
    if src_dtype_str in ("bool", "int16", "int32"):
        if padding_option == "nan":
            pytest.skip("Padding with NaN is not supported for integer types")
        a = torch.randint(0, 2, (n, ), device=device, dtype=src_dtype)
    else:
        a = torch.randn((n, ), device=device, dtype=src_dtype)
    b = torch.zeros((n, ), device=device, dtype=dst_dtype)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), )
    # block_copy_kernel[grid](a_ptr=a, b_ptr=b, N=n, BLOCK_SIZE=64, padding_option=padding_option)
    block_copy_kernel[grid](a_ptr=a, b_ptr=b, N=n, padding_option=padding_option)
    a.to(dst_dtype)
    
    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    assert torch.all(a[0:n // 2] == b[0:n // 2])
    if padding_option == "zero":
        assert torch.all(b[n // 2:n] == 0)
    elif padding_option == "nan":
        assert torch.all(torch.isnan(b[n // 2:n]))

    
    ################### save True in result_gold (indicates it passed block copy tests, implies the gen kernel worked) ###################
    c = torch.tensor([[1.0]])
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = c.clone().detach().cpu()
    ################################################################### 

# --- Define GB/s calculator for Block Copy ---
# TFLOPS is not relevant for a copy operation.
def calculate_block_copy_gbps(params: dict, ms: float) -> float:
    N = params['N']
    # Kernel reads N/2 elements from A and writes N/2 elements to B (plus padding if any).
    # For bandwidth, consider the actual data moved.
    # If padding is "zero" or "nan", these are effectively writes.
    # Let's count N/2 read and N/2 write of actual data from A.
    # The padding writes up to BLOCK_SIZE elements per block, but only up to N for b_block_ptr.
    # For simplicity, assume effective N/2 elements read, N/2 elements written from A's content.
    # If padding fills the rest of B up to N, then N elements written in total.
    # The kernel copies a_block_ptr (derived from N//2 elements) to b_block_ptr.
    # The load is from a conceptual N//2 elements. The store is into N elements.
    # Let's assume N/2 elements are read from 'a' and N/2 elements (from 'a') are written to 'b'.
    # The padding part affects what's written to b[N//2:].
    
    # Data moved: read N/2 elements, write N/2 elements (from 'a')
    # + potentially N/2 elements written as padding.
    # Let's consider useful data copied: N/2 read, N/2 written.
    elements_moved = (N // 2) + (N // 2) # Read from A, Write to B (meaningful part)
    
    dtype_str = params['src_dtype_str'] # Use source dtype for element size
    if dtype_str == 'bool': element_size = 1 # Approx
    elif dtype_str in ('int16', 'float16', 'bfloat16'): element_size = 2
    elif dtype_str in ('int32', 'float32'): element_size = 4
    else: element_size = 4 # Default

    total_bytes = elements_moved * element_size
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

# --- Name for the benchmark output JSON file ---
OP_NAME_FOR_BENCHMARK = "block_copy_triton_perf"

# --- Pytest test_block_copy function MODIFIED for performance benchmarking ---
# Original parametrization is kept.
# BLOCK_SIZE is fixed at 64 in the original kernel call.
# If we want to test different BLOCK_SIZEs, it needs to be part of parametrize
# and passed to the kernel. For now, stick to fixed BLOCK_SIZE=64.
FIXED_BLOCK_SIZE_FOR_PERF = 64

@pytest.mark.parametrize("dtypes_str_tuple, n, padding_option", [
    (dtypes_tuple, n_val, padding_val)
    for dtypes_tuple in (("float16", "float16"), ("float32", "float32"), ("bfloat16", "bfloat16"), ("int32", "int32")) # Focus on common perf types
    for n_val in (1024, 8192, 65536, 262144, 1048576) # Larger N for perf
    for padding_val in (None, "zero") # "nan" might be slow or less common for perf focus
])
def test_performance(dtypes_str_tuple, n, padding_option, request, device='cuda'): # Renamed
    src_dtype_str, dst_dtype_str = dtypes_str_tuple
    # For performance, usually src_dtype == dst_dtype for simple copy.
    # If they differ, it's a copy + cast. Let's assume they are same for perf test.
    if src_dtype_str != dst_dtype_str:
        pytest.skip("Skipping perf test where src_dtype != dst_dtype for block_copy.")
    
    current_dtype = getattr(torch, src_dtype_str)
    check_type_supported(current_dtype, device) # Check for bfloat16/float8 if added

    if src_dtype_str in ("bool", "int16", "int32") and padding_option == "nan":
        pytest.skip("Padding with NaN is not supported for integer types")

    set_seed()
    if src_dtype_str in ("bool", "int16", "int32"):
        a = torch.randint(0, 2, (n,), device=device, dtype=current_dtype)
    else:
        a = torch.randn((n,), device=device, dtype=current_dtype)
    
    # b is the output tensor. For benchmarking, its initial content doesn't matter as it's overwritten.
    b = torch.empty((n,), device=device, dtype=current_dtype) # Use empty for perf

    # Kernel launch grid
    # The kernel processes N//2 elements of A. Grid should be based on this.
    # If BLOCK_SIZE is 64, grid is cdiv(N//2, 64)
    # However, the original test_block_copy uses cdiv(n, meta["BLOCK_SIZE"])
    # This seems to imply the kernel is launched as if processing N elements of A,
    # but internally a_block_ptr shape is N//2.
    # Let's follow the original grid logic for the launch.
    # The kernel's pid will iterate based on this grid.
    # Inside kernel: a_block_ptr offsets are pid * BLOCK_SIZE, shape N//2.
    # This means if grid is cdiv(N, BLOCK_SIZE), pid can go up to N/BLOCK_SIZE -1.
    # pid * BLOCK_SIZE can exceed N//2. This is handled by boundary_check in tl.load.
    # This seems correct for testing boundary checks.
    
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), ) # BLOCK_SIZE is a key for autotune if used
                                                              # Here, it's a direct constexpr.

    # --- Create op_lambda for benchmarking ---
    # BLOCK_SIZE and padding_option are constexpr in the kernel
    op_lambda = lambda: block_copy_kernel[grid](
        a_ptr=a, b_ptr=b, N=n,
        # BLOCK_SIZE=FIXED_BLOCK_SIZE_FOR_PERF, # Pass as constexpr
        padding_option=padding_option         # Pass as constexpr
    )

    # --- Benchmarking ---
    bench_config = do_bench_config(warm_up=50, repetition=200) # Copy is fast
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)

    current_params_for_logs_and_calc = {
        "N": n, "src_dtype_str": src_dtype_str, "dst_dtype_str": dst_dtype_str,
        "padding_option": padding_option,
        "BLOCK_SIZE": FIXED_BLOCK_SIZE_FOR_PERF # Log the fixed block size
    }

    benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
                              gbps_calculator=calculate_block_copy_gbps,
                              tflops_calculator=None) # TFLOPS not relevant

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