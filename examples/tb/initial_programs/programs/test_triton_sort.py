# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
######################################## Imports ########################################
import pytest
import torch
import triton
import triton.language as tl
import numpy as np
######################################## Imports ########################################

@triton.jit
def sort_kernel(X, Z, N: tl.constexpr, M: tl.constexpr, descending: tl.constexpr):
    offx = tl.arange(0, M)
    offy = tl.arange(0, N) * M
    off2d = offx[None, :] + offy[:, None]
    x = tl.load(X + off2d)
    x = tl.sort(x, descending=descending)
    tl.store(Z + off2d, x)

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




# Common helper code
def np_dtype_from_str(dtype_str):
    if dtype_str == "bfloat16": return np.float32
    return getattr(np, dtype_str)

def gen_numpy_array_for_torch_conversion(shape, dtype_str, low=0, high=100):
    np_dtype = np_dtype_from_str(dtype_str)
    actual_low = int(low)
    actual_high = int(high)

    if dtype_str == 'bfloat16':
        return np.random.uniform(actual_low, actual_high, shape).astype(np.float32)
    if 'float' in dtype_str:
        return np.random.uniform(actual_low, actual_high, shape).astype(np_dtype)
    elif 'int' in dtype_str:
        if actual_high <= actual_low: actual_high = actual_low + 1
        if np.issubdtype(np_dtype, np.integer):
            iinfo = np.iinfo(np_dtype)
            actual_low = max(iinfo.min, actual_low)
            # For randint, high is exclusive. Cap at iinfo.max, so use iinfo.max + 1 for np.random.randint
            # but ensure it does not overflow if iinfo.max is already the true max value.
            effective_high = min(iinfo.max, actual_high)
            if effective_high <= actual_low: effective_high = actual_low +1 
            if effective_high > iinfo.max : effective_high = iinfo.max # Ensure it doesn't exceed max for dtype for randint high bound
            
            # np.random.randint's high parameter is exclusive
            return np.random.randint(actual_low, effective_high + 1 if effective_high < iinfo.max else effective_high, shape, dtype=np_dtype)


    raise ValueError(f"Unsupported dtype_str for gen_numpy_array_for_torch_conversion: {dtype_str}")

def torch_dtype_from_str(dtype_str): # Also needed for torch.from_numpy().to(torch_dtype)
    if dtype_str == "bfloat16": return torch.bfloat16
    if dtype_str == "float16": return torch.float16
    if dtype_str == "float32": return torch.float32
    # No float64 in original parametrize for sort, but good to have
    if dtype_str == "int32": return torch.int32
    raise ValueError(f"Unsupported dtype_str for torch: {dtype_str}")


@pytest.mark.interpreter
@pytest.mark.parametrize("M_cols, N_rows", [[512, 1], [64, 8], [16, 256], [8, 512]])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("dtype_str", ['int32', 'float16', 'float32', 'bfloat16'])
def test_sort(N_rows, M_cols, descending, dtype_str, request, device='cuda'):
    set_seed()

    x_np = gen_numpy_array_for_torch_conversion((N_rows, M_cols), dtype_str=dtype_str)
    
    torch_dtype = torch_dtype_from_str(dtype_str)
    x = torch.from_numpy(x_np).to(dtype=torch_dtype, device=device) # Ensure correct torch dtype

    y_ref = torch.sort(x, dim=1, descending=descending)[0]
    z_triton = torch.empty_like(x)

    # Grid is (N_rows,) since each program sorts one row
    grid = (N_rows,)
    sort_kernel[grid](x, z_triton, N_rows, M_cols, descending) # num_warps removed, let triton decide or set default

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = z_triton.clone().detach().cpu()
    ################################################################### 

    assert torch.allclose(y_ref.float(), z_triton.float(), atol=1e-2, rtol=1e-2), \
        f"Sort mismatch for dtype {dtype_str}, shape ({N_rows}, {M_cols}), descending={descending}.\nReference: {y_ref}\nTriton: {z_triton}"


# --- Python wrapper for the kernel for benchmarking ---
def sort_triton_wrapper(x_tensor, z_buffer, N_const, M_const, descending_const, num_warps_launch):
    # For benchmarking, we assume the kernel processes one N_const x M_const tile.
    # The grid will be (1,) as the kernel is not tiled with program_id for larger inputs.
    grid = (1,) 
    sort_kernel[grid](
        x_tensor, z_buffer, 
        N=N_const, M=M_const, # Pass as constexpr
        descending=descending_const,
        # num_warps=num_warps_launch # Launch hint
    )
    return z_buffer

# --- Define TFLOPS and GB/s calculators ---
def calculate_sort_tflops(params: dict, ms: float) -> float:
    # Sorting N_rows * (M_cols * log(M_cols)) comparisons approx.
    # Each comparison is 1 FLOP. This is a rough lower bound.
    N_rows = params['N_tile_rows'] 
    M_cols = params['M_tile_cols']
    if M_cols <= 1: return 0.0 # No comparisons if 0 or 1 element per row
    
    # For tl.sort on each row of M_cols elements
    flops_per_row = M_cols * np.log2(M_cols) if M_cols > 0 else 0 # O(M log M) comparisons
    total_flops = N_rows * flops_per_row
    tflops = total_flops / (ms / 1000) / 1e12
    return tflops

def calculate_sort_gbps(params: dict, ms: float) -> float:
    N_rows = params['N_tile_rows'] 
    M_cols = params['M_tile_cols']
    dtype_str = params.get('dtype_str', 'fp32') 

    current_torch_dtype = torch_dtype_from_str(dtype_str)
    element_size = torch.tensor([], dtype=current_torch_dtype).element_size()

    elements_processed = N_rows * M_cols
    bytes_read = elements_processed * element_size
    bytes_write = elements_processed * element_size 
    total_bytes = bytes_read + bytes_write
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

OP_NAME_FOR_BENCHMARK = "triton_sort_perf"

# --- Pytest parametrize for performance testing ---
# Kernel's N and M are constexpr, defining the tile size it processes.
SORT_TILE_SHAPES_FOR_PERF = [
    # N_tile_rows, M_tile_cols (for the single tile processed by kernel)
    (1, 1024), (1, 4096), (1, 8192), (1, 16384), # Sorting long single rows
    (8, 512), (8, 1024), (8, 2048),             # Sorting a few medium rows
    (64, 128), (64, 256),            # Sorting more, shorter rows
    (256, 64), (256, 128), (256, 256)
]
SORT_DTYPES_FOR_PERF = ['int32', 'float16', 'float32', 'bfloat16'] 
SORT_DESCENDING_FOR_PERF = [False, True]
# NUM_WARPS_FOR_PERF = [1, 2, 4, 8]

@pytest.mark.parametrize("N_tile_rows, M_tile_cols", SORT_TILE_SHAPES_FOR_PERF)
@pytest.mark.parametrize("descending_val", SORT_DESCENDING_FOR_PERF)
@pytest.mark.parametrize("dtype_str", SORT_DTYPES_FOR_PERF)
# @pytest.mark.parametrize("num_warps_launch", NUM_WARPS_FOR_PERF) # Optional
def test_performance(N_tile_rows, M_tile_cols, descending_val, dtype_str, request, device='cuda'):
    # num_warps_launch = 4 # Or from parametrize
    
    if dtype_str == 'bfloat16':
        cap = torch.cuda.get_device_capability()
        if cap[0] < 8:
            pytest.skip("bfloat16 requires Ampere+ (arch 80+)")
    
    set_seed()
    current_torch_dtype = torch_dtype_from_str(dtype_str)

    # Input tensor `x_perf` has shape (N_tile_rows, M_tile_cols)
    # as the kernel is designed to process one such tile.
    x_np_perf = gen_numpy_array_for_torch_conversion((N_tile_rows, M_tile_cols), dtype_str)
    x_perf_tensor = torch.from_numpy(x_np_perf).to(dtype=current_torch_dtype, device=device)
    
    z_perf_buffer = torch.empty_like(x_perf_tensor)
    
    op_lambda = lambda: sort_triton_wrapper(
        x_perf_tensor, z_perf_buffer, 
        N_tile_rows, M_tile_cols, descending_val,
        num_warps_launch=4 # Example, can be parametrized
    )

    bench_config = do_bench_config(warm_up=50, repetition=200) # Sorting can vary
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)
    current_params_for_logs_and_calc = {
        "N_tile_rows": N_tile_rows, 
        "M_tile_cols": M_tile_cols,
        "descending": descending_val,
        "dtype_str": dtype_str,
        "num_warps": 4 # Log the fixed num_warps
    }
    
    perf_result = benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
                                            gbps_calculator=calculate_sort_gbps,
                                            tflops_calculator=calculate_sort_tflops)



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