# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
import argparse  
import sys  
import pytest  
  
import torch  
import triton  
import triton.language as tl  
import os  
import json  
import math  
from itertools import product  
  
######################################## HELPERS utils ######################################## 
def is_cuda():  
    return triton.runtime.driver.active.get_current_target().backend == "cuda"  
  
  
def is_hip():  
    return triton.runtime.driver.active.get_current_target().backend == "hip"  
  
  
def get_cuda_autotune_config():  
    return [  
        triton.Config({}, num_warps=4, num_stages=1),  
        triton.Config({}, num_warps=8, num_stages=1),  
        triton.Config({}, num_warps=16, num_stages=1),  
    ]  
  
  
def get_hip_autotune_config():  
    return [  
        triton.Config({'waves_per_eu': we}, num_warps=wa, num_stages=1) for we, wa in product([1, 2, 4], [4, 8, 16])  
    ]  
  
  
def get_autotune_config():  
    if is_cuda():  
        return get_cuda_autotune_config()  
    else:  
        return get_hip_autotune_config()  
  
######################################## HELPERS utils ######################################## 


@triton.autotune(configs=get_autotune_config(), key=['n_rows', 'n_cols'], use_cuda_graph=True)  
@triton.jit  
def layernorm_kernel(x_ptr, y_ptr, w_ptr, b_ptr, mean_ptr, rstd_ptr, x_row_stride, y_row_stride, n_rows, n_cols, eps,  
                     BLOCK_SIZE: tl.constexpr):  
  
    #program id  
    row = tl.program_id(0)  
    x_ptr_start = x_ptr + (row * x_row_stride)  
    y_ptr_start = y_ptr + (row * y_row_stride)  
  
    loop_num = tl.cdiv(n_cols, BLOCK_SIZE) - 1  
  
    #calculate mean  
    mean = 0  
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)  
    loop_num_l = loop_num  
    for b in range(0, loop_num_l):  
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)  #Unmasked loads  
        _mean += x_block  
  
    #For last iteration, do masked load  
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  
    x_block = tl.load(x_ptr_start + col_offsets, mask=col_offsets < n_cols, other=0.).to(tl.float32)  
    _mean += x_block  
    mean = tl.sum(_mean, axis=0) / n_cols  
  
    #variance  
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)  
    loop_num_l = loop_num  
    for b in range(0, loop_num_l):  
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)  #Unmasked loads  
        x_block = x_block - mean  
        _var += x_block * x_block  
  
    #For last iteration, do masked load  
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  
    x_block = tl.load(x_ptr_start + col_offsets, mask=col_offsets < n_cols, other=0.).to(tl.float32)  
    x_block = tl.where(col_offsets < n_cols, x_block - mean, 0.)  
    _var += x_block * x_block  
  
    var = tl.sum(_var, axis=0) / n_cols  
    rstd = tl.rsqrt(var + eps)  
  
    # Write mean / rstd  
    tl.store(mean_ptr + row, mean)  
    tl.store(rstd_ptr + row, rstd)  
  
    #Normalize and store  
    loop_num_l = loop_num  
    for b in range(0, loop_num_l):  
        col_offsets = b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  
        w_block = tl.load(w_ptr + col_offsets)  
        b_block = tl.load(b_ptr + col_offsets)  
        x_block = tl.load(x_ptr_start + col_offsets).to(tl.float32)  
        y_block = (x_block - mean) * rstd  
        y_block = y_block * w_block + b_block  
        tl.store(y_ptr_start + col_offsets, y_block)  
  
    #For last iteration, do masked load and store  
    col_offsets = loop_num_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  
    mask = col_offsets < n_cols  
    w_block = tl.load(w_ptr + col_offsets, mask=mask, other=0.0)  
    b_block = tl.load(b_ptr + col_offsets, mask=mask, other=0.0)  
    x_block = tl.load(x_ptr_start + col_offsets, mask=mask, other=0.0).to(tl.float32)  
    y_block = (x_block - mean) * rstd  
    y_block = y_block * w_block + b_block  
    tl.store(y_ptr_start + col_offsets, y_block, mask=mask)  
  
  
##################################################################################################################################################  

import numpy as np
import random
import torch 
import os
from geak_eval.perf.ROCm.performance_utils_pytest import PytestBenchmarker, do_bench_config, save_all_benchmark_results
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
  

@triton.jit  
def _layer_norm_bwd_dx_fused(DX,  # pointer to the input gradient  
                             DY,  # pointer to the output gradient  
                             DW,  # pointer to the partial sum of weights gradient  
                             DB,  # pointer to the partial sum of biases gradient  
                             X,  # pointer to the input  
                             W,  # pointer to the weights  
                             Mean,  # pointer to the mean  
                             Rstd,  # pointer to the 1/std  
                             stride,  # how much to increase the pointer when moving by 1 row  
                             N,  # number of columns in X  
                             NUM_ROWS: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):  
    # Map the program id to the elements of X, DX, and DY it should compute.  
    pid = tl.program_id(0)  
    pid_n = tl.program_id(1)  
    tile_num = tl.num_programs(0)  
    rows_per_tile = NUM_ROWS // tile_num  
    if pid < NUM_ROWS % tile_num:  
        rows_per_tile += 1  
  
    cols = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  
    mask = cols < N  
    row = pid  
    for _ in range(0, rows_per_tile):  
        x_ptrs = X + row * stride  
        dy_ptrs = DY + row * stride  
        dx_ptrs = DX + row * stride  
        dw_ptrs = DW + pid * N + cols  
        db_ptrs = DB + pid * N + cols  
        # Load data to SRAM  
        x = tl.load(x_ptrs + cols, mask=mask, other=0).to(tl.float32)  
        dy = tl.load(dy_ptrs + cols, mask=mask, other=0).to(tl.float32)  
        w = tl.load(W + cols, mask=mask).to(tl.float32)  
        mean = tl.load(Mean + row)  
        rstd = tl.load(Rstd + row)  
        # Compute dx  
        xhat = (x - mean) * rstd  
        wdy = w * dy  
        xhat = tl.where(mask, xhat, 0.)  
        wdy = tl.where(mask, wdy, 0.)  
        c1 = tl.sum(xhat * wdy, axis=0) / N  
        c2 = tl.sum(wdy, axis=0) / N  
        dx = (wdy - (xhat * c1 + c2)) * rstd  
        # Write dx  
        tl.store(dx_ptrs + cols, dx, mask=mask)  
        # Accumulate partial sums for dw/db  
        partial_dw = (dy * xhat).to(w.dtype)  
        partial_db = (dy).to(w.dtype)  
        partial_dw += tl.load(dw_ptrs, mask=mask)  
        partial_db += tl.load(db_ptrs, mask=mask)  
        tl.store(dw_ptrs, partial_dw, mask=mask)  
        tl.store(db_ptrs, partial_db, mask=mask)  
        row += tile_num  
  
  
@triton.jit  
def _layer_norm_bwd_dwdb(DW,  # pointer to the partial sum of weights gradient  
                         DB,  # pointer to the partial sum of biases gradient  
                         FINAL_DW,  # pointer to the weights gradient  
                         FINAL_DB,  # pointer to the biases gradient  
                         M,  # GROUP_SIZE_M  
                         N,  # number of columns  
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):  
    # Map the program id to the elements of DW and DB it should compute.  
    pid = tl.program_id(0)  
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  
    # Iterate through the rows of DW and DB to sum the partial sums.  
    for i in range(0, M, BLOCK_SIZE_M):  
        rows = i + tl.arange(0, BLOCK_SIZE_M)  
        mask = (rows[:, None] < M) & (cols[None, :] < N)  
        offs = rows[:, None] * N + cols[None, :]  
        dw += tl.load(DW + offs, mask=mask, other=0.)  
        db += tl.load(DB + offs, mask=mask, other=0.)  
    # Write the final sum to the output.  
    sum_dw = tl.sum(dw, axis=0)  
    sum_db = tl.sum(db, axis=0)  
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)  
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)  



class LayerNorm(torch.autograd.Function):  
  
    @staticmethod  
    def forward(ctx, x, normalized_shape, weight, bias, eps=1e-5):  
        y = torch.empty_like(x)  
        M, N = x.shape  
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)  
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)  
        # Less than 64KB per feature: enqueue fused kernel  
        MAX_FUSED_SIZE = 65536 // x.element_size()  
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))  
        # heuristics for number of warps  
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)  
        layernorm_kernel[(M, )](x, y, weight, bias, mean, rstd, x.stride(0), y.stride(0), M, N, eps, BLOCK_SIZE)  
        ctx.save_for_backward(x, weight, bias, mean, rstd)  
        ctx.BLOCK_SIZE = BLOCK_SIZE  
        ctx.num_warps = num_warps  
        ctx.eps = eps  
  
        return y  
  
    @staticmethod  
    def backward(ctx, dy):  
        x, w, b, m, v = ctx.saved_tensors  
        N = w.shape[0]  
        x_arg = x.reshape(-1, x.shape[-1])  
        M = x_arg.shape[0]  
        tile_num = max(min(256, M // 4), 1)  
        # allocate output  
        _dw = torch.zeros((tile_num, N), dtype=x.dtype, device=w.device)  
        _db = torch.zeros((tile_num, N), dtype=x.dtype, device=w.device)  
        dw = torch.empty((N, ), dtype=w.dtype, device=w.device)  
        db = torch.empty((N, ), dtype=w.dtype, device=w.device)  
        dx = torch.empty_like(dy)  
  
        # enqueue kernel using forward pass heuristics  
        # also compute partial sums for DW and DB  
        M, N = x_arg.shape  
        grid_bwd = lambda meta: (tile_num, triton.cdiv(N, meta['BLOCK_SIZE_N']))  
        _layer_norm_bwd_dx_fused[grid_bwd](  #  
            dx, dy, _dw, _db, x, w, m, v,  #  
            x_arg.stride(0), N,  #  
            NUM_ROWS=M,  #  
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,  #  
            num_warps=ctx.num_warps)  
        grid_reduce = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]  
        # accumulate partial sums in separate kernel  
        _layer_norm_bwd_dwdb[grid_reduce](  
            _dw, _db, dw, db, min(tile_num, M), N,  #  
            BLOCK_SIZE_M=32,  #  
            BLOCK_SIZE_N=128)  
  
        return dx, None, dw, db, None  
  
  
layernorm = LayerNorm.apply  
  
  
def torch_layernorm(x, w_shape, w, b):  
    M, N = x.shape  
    y_torch = torch.nn.functional.layer_norm(x, w_shape, w, b, eps=1e-5)  
    return y_torch  
  
  
def run_layernorm(M, N):  
    print(f"Running Layernorm on shape ({M},{N})")  
    set_seed()  
    x = torch.randn(M, N, device='cuda')  
    w_shape = (N, )  
    w = torch.rand(w_shape, device='cuda')  
    b = torch.rand(w_shape, device='cuda')  
    y_triton = layernorm(x, w_shape, w, b)  
  
    return y_triton  
  
  
#pytest  
@pytest.mark.parametrize('M, N', [(1823, 781), (2, 128), (1, 4), (128, 2), (1, 128), (8192, 8192), (4096, 8192),  
                                  (359, 1), (1, 359), (1, 131072), (1, 89999)])  
def test_layernorm(M, N, request, eps=1e-5):  
    set_seed()  
    x = torch.randn(M, N, device='cuda')  
    w_shape = (N, )  
    w = torch.rand(w_shape, device='cuda', requires_grad=True)  
    b = torch.rand(w_shape, device='cuda', requires_grad=True)  
  
    dy = 0.1 * torch.randn_like(x)  
    x.requires_grad_(True)  
  
    # forward pass  
    y_triton = layernorm(x, w_shape, w, b, eps)  
    y_ref = torch.nn.functional.layer_norm(x, w_shape, w, b, eps)  

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_") + "_fwd"
    result_gold[sanitized_key_name] = y_triton.clone().detach().cpu()
    ###################################################################
  
    # backward pass (triton)  
    y_triton.backward(dy, retain_graph=True)  
    dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, w, b]]  
    x.grad, w.grad, b.grad = None, None, None  
  
    #backward pass (torch)  
    y_ref.backward(dy, retain_graph=True)


# --- Define TFLOPS and GB/s calculators for LayerNorm Forward ---
def calculate_layernorm_gbps(params: dict, ms: float) -> float:
    M = params['M']
    N = params['N']
    # Assuming params contains dtype_str for x, w, b or we infer.
    # For layernorm:
    # Read x (M*N), w (N), b (N)
    # Write y (M*N), mean (M), rstd (M)
    try:
        # Assuming x, y, w, b, mean, rstd are of similar precision for byte counting
        # A more precise calculation would use actual dtypes if they vary significantly.
        dtype = torch.float16 # Default assumption if not in params
        if 'dtype_str' in params:
            if params['dtype_str'] == 'fp32': dtype = torch.float32
            elif params['dtype_str'] == 'bf16': dtype = torch.bfloat16
        element_size = torch.tensor([], dtype=dtype).element_size()
    except KeyError:
        element_size = 2 # Default to 2 bytes (fp16)

    bytes_read_x = M * N * element_size
    bytes_read_w = N * element_size
    bytes_read_b = N * element_size
    bytes_write_y = M * N * element_size
    bytes_write_mean = M * 4 # Mean/rstd often float32
    bytes_write_rstd = M * 4

    total_bytes = bytes_read_x + bytes_read_w + bytes_read_b + \
                  bytes_write_y + bytes_write_mean + bytes_write_rstd
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

def calculate_layernorm_tflops(params: dict, ms: float) -> float:
    M = params['M']
    N = params['N']
    # FLOPs for LayerNorm forward:
    # 1. Mean: N additions + 1 division per row  (N ops)
    # 2. Variance: N subtractions, N squares, N additions, 1 division per row (3N+1 ops, approx 3N)
    # 3. rsqrt: (approx ~5-10 ops, let's say 5)
    # 4. Normalize: N subtractions, N multiplications per row (2N ops)
    # 5. Scale/shift: N multiplications, N additions per row (2N ops)
    # Total per row: N + 3N + 5 + 2N + 2N = 8N + 5 ops
    # Total FLOPs: M * (8*N + 5)
    flops = M * (8 * N + 5) # Simplified, actual can vary by rsqrt implementation etc.
    tflops = flops / (ms / 1000) / 1e12
    return tflops

# --- Name for the benchmark output JSON file ---
OP_NAME_FOR_BENCHMARK = "layernorm_triton_fwd_perf"

# --- Pytest parametrize for performance testing ---
# Using the same parameters as the original test_layernorm
LAYERNORM_SHAPES_FOR_PERF = [
    (1823, 781), (2048, 2048), (8192, 8192), # Some medium to large
    (4096, 10240), # LLM typical
    (1, 131072), (1, 89999), # Long sequence, batch 1
    (128, 2048), (512, 4096)
]
# Dtypes to test for performance
DTYPES_FOR_PERF = ['fp16', 'bf16', 'fp32']


@pytest.mark.parametrize('M, N', LAYERNORM_SHAPES_FOR_PERF)
@pytest.mark.parametrize('dtype_str', DTYPES_FOR_PERF)
def test_performance(M, N, dtype_str, request):
    set_seed()
    eps = 1e-5

    # Determine torch dtype
    if dtype_str == 'fp32': current_dtype = torch.float32
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16
    else: current_dtype = torch.float16 # Default to fp16

    # Input tensors
    x = torch.randn(M, N, device='cuda', dtype=current_dtype)
    normalized_shape_arg = (N,) # For torch.nn.functional.layer_norm and our LayerNorm.apply
    w = torch.rand(N, device='cuda', dtype=current_dtype)
    b = torch.rand(N, device='cuda', dtype=current_dtype)

    # --- Create op_lambda for benchmarking the forward pass ---
    op_lambda = lambda: layernorm(x, normalized_shape_arg, w, b, eps)

    # --- Benchmarking ---
    # Autotuned kernels might benefit from fewer reps if tuning takes time,
    # but do_bench needs enough reps for stable measurement AFTER tuning.
    bench_config = do_bench_config(warm_up=25, repetition=100)
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)

    current_params_for_logs_and_calc = {
        "M": M, "N": N, "eps": eps, "dtype_str": dtype_str
    }

    benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
                              gbps_calculator=calculate_layernorm_gbps,
                              tflops_calculator=calculate_layernorm_tflops)
    
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