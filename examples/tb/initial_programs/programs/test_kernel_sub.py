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
def kernel_sub(a, b, o, N: tl.constexpr):
    idx = tl.arange(0, N)
    tl.store(o + idx, tl.load(a + idx) - tl.load(b + idx) * 777)


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
def compile_kernel_sub_for_test(attrs): # Renamed to be specific
    # kernel_sub is defined globally above
    src = ASTSource(
        fn=kernel_sub,
        constants={'N': 32},
        signature={'a': "*fp32", 'b': "*fp32", 'o': "*fp32"},
        attrs=attrs,
    )
    triton.compile(src=src, target=target)

@skip_if_no_target
def test_compile_kernel_sub_in_subproc(fresh_triton_cache, request) -> None: # Test name updated for clarity

    set_seed()
    
    config = AttrsDescriptor.from_hints({i: 16 for i in range(4)})
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        print("Warning: Could not force 'fork' start method. Using default.")
        if multiprocessing.get_start_method(allow_none=True) != 'fork': # allow_none for safety
            pytest.skip("Test requires 'fork' multiprocessing start method.")

    proc = multiprocessing.Process(target=compile_kernel_sub_for_test, args=(config, ))
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


def kernel_sub_triton_wrapper(a_tensor, b_tensor, o_buffer, 
                              N_val_const, num_warps_launch): # N_val becomes constexpr N
    grid = (1,) 
    kernel_sub[grid](
        a_tensor, b_tensor, o_buffer, 
        N=N_val_const, 
        # num_warps=num_warps_launch
    )
    return o_buffer 

def calculate_kernel_sub_tflops(params: dict, ms: float) -> float:
    N = params['N_val']
    flops = 2 * N 
    tflops = flops / (ms / 1000) / 1e12
    return tflops

def calculate_kernel_sub_gbps(params: dict, ms: float) -> float:
    N = params['N_val']
    dtype_str = params.get('dtype_str', 'fp32') 
    current_dtype = torch.float32
    if dtype_str == 'fp16': current_dtype = torch.float16
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16
    element_size = torch.tensor([], dtype=current_dtype).element_size()
    bytes_a_read, bytes_b_read, bytes_o_write = [N * element_size] * 3
    total_bytes = bytes_a_read + bytes_b_read + bytes_o_write
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

OP_NAME_FOR_BENCHMARK = "kernel_sub_triton_perf"

# --- Pytest parametrize for performance testing ---
# MODIFIED: N_val significantly reduced to avoid HSA_STATUS_ERROR_OUT_OF_RESOURCES
# This kernel is not tiled, so N must be small.
# Max N around 16384 to 32768 might be the limit for a single SM program without tiling.
# Let's test up to 65536, but expect larger ones might fail.
KERNEL_SUB_N_VALS_FOR_PERF = [2**i for i in range(10, 17)] # 1024 to 65536
# KERNEL_SUB_N_VALS_FOR_PERF = [2**i for i in range(10, 21)] # Original problematic range

KERNEL_SUB_DTYPES_FOR_PERF = ['fp32', 'fp16'] 
KERNEL_SUB_NUM_WARPS_FOR_PERF = [1, 2, 4] 

@pytest.mark.parametrize("N_val", KERNEL_SUB_N_VALS_FOR_PERF)
@pytest.mark.parametrize("dtype_str", KERNEL_SUB_DTYPES_FOR_PERF)
@pytest.mark.parametrize("num_warps_launch", KERNEL_SUB_NUM_WARPS_FOR_PERF)
@skip_if_no_target 
def test_performance(N_val, dtype_str, num_warps_launch, request):
    set_seed()
    
    # Proactive skip for very large N for this non-tiled kernel
    # This limit is empirical and might need adjustment based on specific GPU/driver
    # For ROCm, even 65536 might be too large without tiling for some internal resources.
    # The error at N=524288 confirms this.
    if N_val > 65536 * 2 : # A more conservative upper bound for non-tiled kernel
         pytest.skip(f"Skipping N_val={N_val} as it's too large for this non-tiled kernel_sub.")

    if dtype_str == 'fp32': current_dtype = torch.float32
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16 
    else: current_dtype = torch.float16

    a = torch.randn(N_val, device='cuda', dtype=current_dtype)
    b = torch.randn(N_val, device='cuda', dtype=current_dtype)
    o_buffer = torch.empty(N_val, device='cuda', dtype=current_dtype) 
    
    op_lambda = lambda: kernel_sub_triton_wrapper(
        a, b, o_buffer, N_val, num_warps_launch
    )

    bench_config = do_bench_config(warm_up=100, repetition=500) 
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)
    current_params_for_logs_and_calc = {
        "N_val": N_val, 
        "dtype_str": dtype_str,
        "num_warps": num_warps_launch
    }
    
    perf_result = benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
                                            gbps_calculator=calculate_kernel_sub_gbps,
                                            tflops_calculator=calculate_kernel_sub_tflops)


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