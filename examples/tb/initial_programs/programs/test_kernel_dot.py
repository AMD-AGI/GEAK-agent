# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
######################################## Imports ########################################
import multiprocessing
import shutil
import tempfile
import os
import pytest

import triton
import triton.language as tl
from triton.backends.compiler import AttrsDescriptor
from triton.compiler import ASTSource
######################################## Imports ########################################

@triton.jit
def kernel_dot(Z):
    offs = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
    z = tl.load(Z + offs)
    z = tl.dot(z, z)
    tl.store(Z + offs, z)


##################################################################################################################################################  

import numpy as np
import random
import torch 
import os
import pytest
from numpy.random import RandomState
import multiprocessing
import shutil
import tempfile
import os
import pytest
from tb_eval.perf.ROCm.performance_utils_pytest import PytestBenchmarker, do_bench_config, save_all_benchmark_results
from typing import Dict
import triton
import triton.language as tl
from triton.backends.compiler import AttrsDescriptor
from triton.compiler import ASTSource

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



# --- Fixture Definition ---
@pytest.fixture
def fresh_triton_cache(monkeypatch):
    temp_dir = tempfile.mkdtemp()
    cache_dir = os.path.join(temp_dir, ".triton")
    os.makedirs(cache_dir, exist_ok=True)
    monkeypatch.setenv("TRITON_CACHE_DIR", cache_dir)
    yield cache_dir
    shutil.rmtree(temp_dir)
# --- End Fixture Definition ---

# Check if a target is available. Skip tests if not.
try:
    target = triton.runtime.driver.active.get_current_target()
    TARGET_AVAILABLE = True
except Exception:
    TARGET_AVAILABLE = False
    target = None

# Decorator to skip tests if target is not available
skip_if_no_target = pytest.mark.skipif(not TARGET_AVAILABLE, reason="Triton target not available (e.g., no GPU or CUDA/ROCm setup)")

@skip_if_no_target
def compile_kernel_dot_for_test(attrs): # Renamed to be specific
    # kernel_dot is defined globally above
    src = ASTSource(fn=kernel_dot, signature={'Z': "*fp32"}, attrs=attrs, constants={})
    triton.compile(src=src, target=target)

@skip_if_no_target
def test_compile_kernel_dot_in_forked_subproc(fresh_triton_cache, request) -> None: # Test name updated for clarity
    config = AttrsDescriptor.from_hints({0: 16})
    current_start_method = multiprocessing.get_start_method(allow_none=True)
    if current_start_method is None:
        try:
            multiprocessing.set_start_method('fork', force=True)
        except RuntimeError:
            print("Warning: Could not force 'fork' start method. Using default.")
    if multiprocessing.get_start_method(allow_none=True) != 'fork':
        pytest.skip("Test requires 'fork' multiprocessing start method.")

    proc = multiprocessing.Process(target=compile_kernel_dot_for_test, args=(config, ))
    proc.start()
    proc.join(timeout=60)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        pytest.fail("Process timed out")

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])

    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = torch.tensor([[0.0]]).clone().detach().cpu()
    ################################################################### 

    assert proc.exitcode == 0

    result_gold[sanitized_key_name] = torch.tensor([[1.0]]).clone().detach().cpu()



# --- Python wrapper for launching kernel_dot for benchmarking ---
def kernel_dot_triton_wrapper(Z_tensor, num_warps_launch):
    # Kernel operates on a 16x16 tile.
    # Grid is (1,) because the kernel itself doesn't use program_id for tiling.
    grid = (1,)
    kernel_dot[grid](Z_tensor) #, num_warps=num_warps_launch) # Z_tensor is modified in-place
    return Z_tensor # Return for consistency, though modified in-place

# --- Define TFLOPS and GB/s calculators for the 16x16 dot product ---
FIXED_DIM_FOR_KERNEL_DOT = 16

def calculate_kernel_dot_tflops(params: dict, ms: float) -> float:
    M = N = K = FIXED_DIM_FOR_KERNEL_DOT
    # Operation: Z_out = Z_in @ Z_in
    flops = 2 * M * N * K 
    tflops = flops / (ms / 1000) / 1e12
    return tflops

def calculate_kernel_dot_gbps(params: dict, ms: float) -> float:
    M = N = K = FIXED_DIM_FOR_KERNEL_DOT
    dtype_str = params.get('dtype_str', 'fp32') # Original signature was *fp32

    current_dtype = torch.float32
    if dtype_str == 'fp16': current_dtype = torch.float16
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16
    element_size = torch.tensor([], dtype=current_dtype).element_size()

    # Read Z (16,16) twice (as A and B), Write Z (16,16)
    bytes_read_z1 = M * K * element_size
    bytes_read_z2 = K * N * element_size 
    bytes_write_z = M * N * element_size
    total_bytes = bytes_read_z1 + bytes_read_z2 + bytes_write_z
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

OP_NAME_FOR_BENCHMARK = "kernel_dot_triton_perf"

# --- Pytest parametrize for performance testing ---
# Kernel is fixed to 16x16. We can vary dtype and num_warps.
KERNEL_DOT_DTYPES_FOR_PERF = ['fp32', 'fp16'] # bf16 if supported and desired
KERNEL_DOT_NUM_WARPS_FOR_PERF = [1, 2, 4] 

@pytest.mark.parametrize("dtype_str", KERNEL_DOT_DTYPES_FOR_PERF)
@pytest.mark.parametrize("num_warps_launch", KERNEL_DOT_NUM_WARPS_FOR_PERF)
@skip_if_no_target # Use the skip if target not available
def test_performance(dtype_str, num_warps_launch, request):
    set_seed()
    
    if dtype_str == 'fp32': current_dtype = torch.float32
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16
    else: current_dtype = torch.float16

    # Input tensor Z (16x16)
    # The kernel signature in ASTSource used *fp32. For perf, we might test other dtypes
    # if the kernel JIT itself is flexible or if we compile variants.
    # For now, assume the JIT `kernel_dot` can handle the dtype of the passed tensor.
    Z_tensor = torch.randn((FIXED_DIM_FOR_KERNEL_DOT, FIXED_DIM_FOR_KERNEL_DOT), 
                           device='cuda', dtype=current_dtype)
    
    # --- Create op_lambda for benchmarking ---
    # The kernel modifies Z_tensor in-place.
    op_lambda = lambda: kernel_dot_triton_wrapper(Z_tensor, num_warps_launch)

    # --- Benchmarking ---
    bench_config = do_bench_config(warm_up=100, repetition=1000) # Kernel is tiny, need more reps
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)

    current_params_for_logs_and_calc = {
        "DIM": FIXED_DIM_FOR_KERNEL_DOT, # M=N=K=16
        "dtype_str": dtype_str,
        "num_warps": num_warps_launch
    }
    
    perf_result = benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
                                            gbps_calculator=calculate_kernel_dot_gbps,
                                            tflops_calculator=calculate_kernel_dot_tflops)

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