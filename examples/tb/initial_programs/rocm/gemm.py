# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import sys
import argparse
import pytest
import re

# This is a Triton kernel for matrix multiplication (GEMM) with support for various data types and scaling modes.

#################### Helper utils functions ####################
# Activation function.  
@triton.jit  
def leaky_relu(x):  
    x = x + 1  
    return tl.where(x >= 0, x, 0.01 * x)  
#################### Helper utils functions ####################



@triton.autotune(  
    configs=[  
        triton.Config(  
            {  
                'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,  
                'kpack': 2, 'matrix_instr_nonkdim': 16  
            }, num_warps=4, num_stages=2),  
        triton.Config(  
            {  
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,  
                'kpack': 2, 'matrix_instr_nonkdim': 16  
            }, num_warps=8, num_stages=2),  
        triton.Config(  
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4, 'waves_per_eu': 0},  
            num_warps=8, num_stages=2),  
        triton.Config(  
            {  
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,  
                'kpack': 1, 'matrix_instr_nonkdim': 16  
            }, num_warps=8, num_stages=2),  
        triton.Config(  
            {  
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 0,  
                'kpack': 1  
            }, num_warps=8, num_stages=2),  
        triton.Config(  
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'waves_per_eu': 0},  
            num_warps=8, num_stages=2),  
        triton.Config(  
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},  
            num_warps=8, num_stages=2),  
    ],  
    key=['M', 'N', 'K'],  
    use_cuda_graph=True,  
)  
@triton.heuristics({  
    'EVEN_K':  
    lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0, 'GRID_MN':  
    lambda args: triton.cdiv(args['M'], args['BLOCK_SIZE_M']) * triton.cdiv(args['N'], args['BLOCK_SIZE_N'])  
})  
@triton.jit  
def matmul_kernel(  
    a_ptr,  
    b_ptr,  
    c_ptr,  
    M,  
    N,  
    K,  
    stride_am,  
    stride_ak,  
    stride_bk,  
    stride_bn,  
    stride_cm,  
    stride_cn,  
    a_scale_ptr,  
    b_scale_ptr,  
    stride_ascale_m,  
    stride_ascale_k,  
    stride_bscale_k,  
    stride_bscale_n,  
    # Meta-parameters  
    GROUP_K: tl.constexpr,  
    GROUP_N: tl.constexpr,  
    BLOCK_SIZE_M: tl.constexpr,  
    BLOCK_SIZE_N: tl.constexpr,  
    BLOCK_SIZE_K: tl.constexpr,  
    EVEN_K: tl.constexpr,  
    GROUP_SIZE_M: tl.constexpr,  
    APPLY_SCALE: tl.constexpr,  
    ACTIVATION: tl.constexpr,  
    GRID_MN: tl.constexpr,  
):  
    """Kernel for computing the matmul C = A x B.  
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)  
    """  
  
    NUM_XCDS: tl.constexpr = 8  
  
    tl.static_assert(((APPLY_SCALE is None) or (APPLY_SCALE == 'tensor')) or (APPLY_SCALE == 'block'),  
                     f"Scaling mode {APPLY_SCALE} is not supported!!!")  
  
    tl.assume(stride_am > 0)  
    tl.assume(stride_ak > 0)  
    tl.assume(stride_bk > 0)  
    tl.assume(stride_bn > 0)  
    tl.assume(stride_cm > 0)  
    tl.assume(stride_cn > 0)  
    tl.assume(stride_ascale_m > 0)  
    tl.assume(stride_ascale_k > 0)  
    tl.assume(stride_bscale_k > 0)  
    tl.assume(stride_bscale_n > 0)  
  
    # -----------------------------------------------------------  
    # Map program ids `pid` to the block of C it should compute.  
    # This is done in a grouped ordering to promote L2 data reuse.  
    # TODO(vgokhale): Add XCD remapping.  
    pid = tl.program_id(axis=0)  
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)  
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  
  
    ## pid remapping on xcds  
    # Number of pids per XCD in the new arrangement  
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS  
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have  
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.  
    # We calculate the number of xcds that have pids_per_xcd pids as  
    # tall_xcds  
    tall_xcds = GRID_MN % NUM_XCDS  
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds  
    # Compute current XCD and local pid within the XCD  
    xcd = pid % NUM_XCDS  
    local_pid = pid // NUM_XCDS  
    # Calculate new pid based on the new grouping  
    # Note that we need to consider the following two cases:  
    # 1. the current pid is on a tall xcd  
    # 2. the current pid is on a short xcd  
    if xcd < tall_xcds:  
        pid = xcd * pids_per_xcd + local_pid  
    else:  
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid  
  
    if GROUP_SIZE_M == 1:  
        pid_m = pid // num_pid_n  
        pid_n = pid % num_pid_n  
    else:  
        num_pid_in_group = GROUP_SIZE_M * num_pid_n  
        group_id = pid // num_pid_in_group  
        first_pid_m = group_id * GROUP_SIZE_M  
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)  
        pid_m = first_pid_m + (pid % group_size_m)  
        pid_n = (pid % num_pid_in_group) // group_size_m  
  
    tl.assume(pid_m > 0)  
    tl.assume(pid_n > 0)  
  
    # Create pointers for first block of A and B input matrices  
    offs_k = tl.arange(0, BLOCK_SIZE_K)  
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M  
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N  
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)  
    if APPLY_SCALE == 'tensor':  
        a_scale = tl.load(a_scale_ptr) if a_scale_ptr else 1.0  
        b_scale = tl.load(b_scale_ptr)  
    elif APPLY_SCALE == 'block':  
        k_start = 0  
        offs_ks = k_start // GROUP_K  
        a_scale_ptrs = None if a_scale_ptr is None else (a_scale_ptr + offs_am * stride_ascale_m +  
                                                         offs_ks * stride_ascale_k)  
        offs_bsn = offs_bn // GROUP_N  
        b_scale_ptrs = b_scale_ptr + offs_bsn * stride_bscale_n + offs_ks * stride_bscale_k  
  
    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32  
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)  
  
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):  
        # Load the next block of A and B, generate a mask by checking the K dimension.  
        # If it is out of bounds, set it to 0.  
        if EVEN_K:  
            a = tl.load(a_ptrs)  
            b = tl.load(b_ptrs)  
        else:  
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)  
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)  
  
        if APPLY_SCALE == 'block':  
            b_scale = tl.load(b_scale_ptrs)  
            if a_scale_ptrs is not None:  
                a_scale = tl.load(a_scale_ptrs)  
  
        # Type conversion to support mixed precision GEMMs where b is lower precision than a  
        b = b.to(a_ptr.type.element_ty)  
  
        if APPLY_SCALE == 'block':  
            if a_scale_ptrs is not None:  
                accumulator += tl.dot(a, b, input_precision="ieee") * a_scale[:, None] * b_scale[None, :]  
            else:  
                accumulator += tl.dot(a, b, input_precision="ieee") * b_scale[None, :]  
        else:  
            accumulator += tl.dot(a, b, input_precision="ieee")  
  
        # Advance the ptrs to the next K block.  
        a_ptrs += BLOCK_SIZE_K * stride_ak  
        b_ptrs += BLOCK_SIZE_K * stride_bk  
  
        if APPLY_SCALE == 'block':  
            k_cur = k * BLOCK_SIZE_K // GROUP_K  
            k_nxt = (k + 1) * BLOCK_SIZE_K // GROUP_K  
            offs_ks = k_nxt - k_cur  
            b_scale_ptrs += offs_ks * stride_bscale_k  
            if a_scale_ptrs is not None:  
                a_scale_ptrs += offs_ks * stride_ascale_k  
  
    # Apply scale to recover dynamic range reduced due to lower precision inputs.  
    if APPLY_SCALE == 'tensor':  
        accumulator = accumulator * a_scale * b_scale  
    # Apply activation function, if specified.  
    # TODO(vgokhale): Add different types of activations.  
    if ACTIVATION == "leaky_relu":  
        accumulator = leaky_relu(accumulator)  
    c = accumulator.to(c_ptr.type.element_ty)  
  
    # Write back the block of the output matrix C with masks.  
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]  
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)  
    tl.store(c_ptrs, c, mask=c_mask)  
  
  



##################################################################################################################################################

# This is a Triton kernel for matrix multiplication (GEMM) with support for various data types and scaling modes.
# Usage Instruction: python3 -m pytest gemm.py  
  
import torch  
import triton  
import triton.language as tl  
import sys  
import argparse  
import pytest  
import re  
from geak_eval.perf.ROCm.performance_utils_pytest import PytestBenchmarker, do_bench_config, save_all_benchmark_results
from typing import Dict  
######################################## HELPERS for Eval ######################################## 
import numpy as np
import random
import torch 
import os

result_gold = {}
CONFIG = {
  "llama3": {
    "8B": {
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "hidden_size": 4096,
      "intermediate_size": 14336,
      "vocab_size": 128256
    },
    "70B": {
      "num_attention_heads": 64,
      "num_key_value_heads": 8,
      "hidden_size": 8192,
      "intermediate_size": 28672,
      "vocab_size": 128256
    },
    "405B": {
      "num_attention_heads": 128,
      "num_key_value_heads": 8,
      "hidden_size": 16384,
      "intermediate_size": 53248,
      "vocab_size": 128256
    }
  },
  "mistral": {
    "7B": {
      "hidden_size": 4096,
      "intermediate_size": 14336,
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "vocab_size": 32000
    },
    "22B": {
      "hidden_size": 6144,
      "intermediate_size": 16384,
      "num_attention_heads": 48,
      "num_key_value_heads": 8,
      "vocab_size": 32000
    }

  }
}

  
def get_model_configs(config_path='model_configs.json', model_families=["llama3"], model="all"):  
    """  
    Load model names from the configuration file.  
  
    Args:  
        config_path (str): User-provided path to the configuration JSON file.  
        model_families (list): List of model family names to retrieve.  
  
    Returns:  
        dict: A dictionary of available models and their configurations for the specified families.  
    """  
    configs = CONFIG.copy()
  
    # Extract models and their configurations for the specified families  
    filtered_configs = {}  
  
    for family in model_families:  
        if family in configs:  
            # Check if model filtering is required  
            if model == "all":  
                # Include all models in the family  
                for model_size, model_configs in configs[family].items():  
                    filtered_configs[f"{family}-{model_size}"] = model_configs  
            else:  
                # Parse the model string (e.g., llama3_8B or llama3-8B)  
                delimiter = "_" if "_" in model else "-"  
                model_parts = model.split(delimiter)  
  
                # Check if the family and size match  
                if len(model_parts) == 2 and model_parts[0] == family:  
                    model_size = model_parts[1]  
                    if model_size in configs[family]:  
                        filtered_configs[f"{family}-{model_size}"] = configs[family][model_size]  
  
    if not filtered_configs:  
        print(f"Warning: No models selected for families: {model_families} with filter: '{model}'")  
  
    return filtered_configs  
  
  
def get_available_models(config_file='model_configs.json', model_families=["llama3"]):  
    """  
    Load model names from the configuration file.  
  
    Args:  
        config_file (str): Path to the configuration JSON file.  
        model_families (list): List of model family names to retrieve.  
  
    Returns:  
        list: A list of available models for the specified families.  
    """  
    configs = CONFIG.copy()
  
    models = [f"{family}-{model}" for family in model_families if family in configs for model in configs[family]]  
  
    return models  


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

# --- Define TFLOPS and GB/s calculators for GEMM ---
def calculate_gemm_tflops(params: dict, ms: float) -> float:
    M = params['M']
    N = params['N']
    K = params['K']
    # For GEMM: 2 * M * N * K FLOPs
    flops = 2 * M * N * K
    tflops = flops / (ms / 1000) / 1e12
    return tflops

def calculate_gemm_gbps(params: dict, ms: float) -> float:
    M = params['M']
    N = params['N']
    K = params['K']
    # Dtypes are needed for element size
    # Assuming params will contain 'in_dtype_a_str' and 'in_dtype_b_str'
    # or we can infer from the created tensors if passed differently.
    # For simplicity, let's assume fp16/bf16 for now if not specified.
    # A more robust way is to pass tensor objects or precise dtype info.
    try:
        dtype_a = name_to_torch_types[params['in_dtype_a_str']]
        dtype_b = name_to_torch_types[params['in_dtype_b_str']]
        # dtype_c = name_to_torch_types[params['out_dtype_str']] # If C is different
    except KeyError:
        print("Warning: Dtype strings not found in params for GB/s calc, assuming float16.")
        dtype_a = torch.float16
        dtype_b = torch.float16
        # dtype_c = torch.float16

    bytes_a = M * K * torch.tensor([], dtype=dtype_a).element_size()
    bytes_b = K * N * torch.tensor([], dtype=dtype_b).element_size()
    bytes_c = M * N * torch.tensor([], dtype=dtype_a).element_size() # Assuming C is same type as A for this calc

    # Read A, Read B, Write C
    total_bytes = bytes_a + bytes_b + bytes_c
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

######################################## HELPERS for Eval ######################################## 
   
SCALE_BLOCK_SIZE = 128  
  
  
# Wrapper for gemm kernel.  
def matmul(a, b, c, a_scale, b_scale, scale_a8_b8=None, activation=""):  
    # Check constraints.  
    assert a.shape[1] == b.shape[0], "Incompatible dimensions!!!"  
    assert (a.element_size()  
            >= b.element_size()), "Mixed dtype GEMMs are only supported when data type of a is bigger than b!!!"  
    assert (a.is_floating_point() == b.is_floating_point()  
            ), "GEMMs between float and integer type tensors are not supported!!!"  
    assert (scale_a8_b8 in [None, 'tensor', 'block']), f"Scaling mode {scale_a8_b8} is not supported!!!"  
    M, K = a.shape  
    K, N = b.shape  
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )  
    matmul_kernel[grid](  
        a,  
        b,  
        c,  
        M,  
        N,  
        K,  
        a.stride(0),  
        a.stride(1),  
        b.stride(0),  
        b.stride(1),  
        c.stride(0),  
        c.stride(1),  
        a_scale,  
        b_scale,  
        a_scale.stride(0) if (a_scale is not None) and a_scale.ndim else 0,  
        a_scale.stride(1) if (a_scale is not None) and a_scale.ndim else 0,  
        b_scale.stride(0) if (b_scale is not None) and b_scale.ndim else 0,  
        b_scale.stride(1) if (b_scale is not None) and b_scale.ndim else 0,  
        GROUP_K=SCALE_BLOCK_SIZE,  
        GROUP_N=SCALE_BLOCK_SIZE,  
        APPLY_SCALE=scale_a8_b8,  
        ACTIVATION=activation,  
    )  
  
  
def is_cdna4():  
    return triton.runtime.driver.active.get_current_target().arch == 'gfx950'  
  
  
e5m2_type = torch.float8_e5m2 if is_cdna4() else torch.float8_e5m2fnuz  
e4m3_type = torch.float8_e4m3fn if is_cdna4() else torch.float8_e4m3fnuz  
  
name_to_torch_types = {  
    'int8': torch.int8,  
    'int32': torch.int32,  
    'fp16': torch.float16,  
    'fp32': torch.float32,  
    'bf16': torch.bfloat16,  
    'fp8e5': e5m2_type,  
    'fp8e4': e4m3_type,  
}  
  
dtype_max = {  
    dtype: (torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)).max  
    for dtype in [  
        e5m2_type,  
        e4m3_type,  
        torch.int8,  
    ]  
}  
  
  
def dtype_is_8_bit(dtype):
    return (
        dtype is e5m2_type
        or dtype is e4m3_type
        or dtype is torch.int8
    )
  
  
def gen_input(M, N, dtype, needTrans, seed=0, fp8_scaling_mode='tensor', device='cuda'):  
    set_seed()  
  
    if needTrans:  
        raw_data = torch.randn((N, M), dtype=torch.float32, device='cuda').T  
    else:  
        raw_data = torch.randn((M, N), dtype=torch.float32, device='cuda')  
    scale = None  
    if dtype_is_8_bit(dtype):  
        if fp8_scaling_mode == 'token':  
            assert raw_data.size(1) % SCALE_BLOCK_SIZE == 0  
            raw_data = raw_data.view(M, -1, SCALE_BLOCK_SIZE)  
            max_val = raw_data.abs().float().amax(dim=2).view(M, -1).clamp(1e-4)  
            scale = max_val.unsqueeze(2) / dtype_max[dtype]  
            raw_data = (raw_data / scale).view(M, N)  
            scale = scale.view(M, -1)  
            scale = scale.T.contiguous().T  
        elif fp8_scaling_mode == 'block':  
            x_padded = torch.zeros((triton.cdiv(M, SCALE_BLOCK_SIZE) * SCALE_BLOCK_SIZE,  
                                    triton.cdiv(N, SCALE_BLOCK_SIZE) * SCALE_BLOCK_SIZE), dtype=raw_data.dtype,  
                                   device=raw_data.device)  
            x_padded[:M, :N] = raw_data  
            x_view = x_padded.view(-1, SCALE_BLOCK_SIZE, x_padded.size(1) // SCALE_BLOCK_SIZE, SCALE_BLOCK_SIZE)  
            x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)  
            x_scaled = x_view * (dtype_max[dtype] / x_amax)  
            raw_data = x_scaled.view_as(x_padded)[:M, :N].T.contiguous().T  
            scale = (x_amax / dtype_max[dtype]).view(x_view.size(0), x_view.size(2))  
        elif fp8_scaling_mode == 'tensor':  
            max_val = torch.max(torch.abs(raw_data))  
            scale = max_val / dtype_max[dtype]  
            raw_data = raw_data / scale  
  
    input = raw_data.to(dtype)  
    input_f32 = input.to(torch.float32)  
  
    return input, input_f32, scale  
  
  
def get_x_vals():  
    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]  
  
    x_vals += [(4864, 4096, 8192), (9728, 8192, 65536), (4864, 8192, 4160)]  
  
    return x_vals  
  
  
# Unit tests  
#TODO(vgokhale): Test activation.  
# yapf: disable  
@pytest.mark.parametrize(  
    "M, N, K, in_dtype_a, in_dtype_b, out_dtype, col_a, col_b",  
    [(*shape, in_dtype_a, in_dtype_b, out_dtype, col_a, col_b)  
     for shape in get_x_vals()  
     for in_dtype_a, in_dtype_b, out_dtype in [  
        ('fp16', 'fp16', 'fp16'),   ('bf16', 'bf16', 'bf16'),   ('fp32', 'fp32', 'fp32'),  
        ('fp8e4', 'fp8e4', 'fp16'), ('fp8e5', 'fp8e5', 'fp16'), ('fp16', 'fp8e4', 'fp16'),  
        ('fp16', 'fp8e5', 'fp16'),  ('bf16', 'fp8e4', 'bf16'),  ('bf16', 'fp8e5', 'bf16'),  
        ('int8', 'int8', 'int8'),   ('int8', 'int8', 'int32')]  
     # Defines if a matrix is row or column major.  
     for col_a in [True, False]  
     for col_b in [True, False]])  
# yapf: enable  
def test_correctness(M, N, K, col_a, col_b, in_dtype_a, in_dtype_b, out_dtype, request):  
    set_seed()
    torch_in_dtype_a = name_to_torch_types[in_dtype_a]  
    torch_in_dtype_b = name_to_torch_types[in_dtype_b]  
    a, a_fp32, a_scale = gen_input(M, K, torch_in_dtype_a, col_a, seed=1, device='cuda')  
    b, b_fp32, b_scale = gen_input(K, N, torch_in_dtype_b, col_b, seed=2, device='cuda')  
    torch_out_dtype = name_to_torch_types[out_dtype]  
    c = torch.empty((M, N), device=a.device, dtype=torch_out_dtype)  
    # For 8-bit, we have scaled to the dynamic range of the data type.  
    # This requires us to compute in fp32 because for e5m2, the range is same as fp16 (e5m10).  
    # If we use fp16 it is possible to return infs from the torch.matmul call.  
    if dtype_is_8_bit(torch_in_dtype_a) or dtype_is_8_bit(torch_in_dtype_b):  
        matmul(a, b, c, a_scale, b_scale, scale_a8_b8='tensor', activation="")  
        torch_output = torch.matmul(a_fp32, b_fp32)  
        # Set a_scale to 1.0 if it is not set  
        torch_output = torch_output * (a_scale or 1.0) * b_scale  
    # For other dtypes, use the same torch matmul as the dtype.  
    else:  
        matmul(a, b, c, a_scale=None, b_scale=None, scale_a8_b8=None, activation="")  
        torch_output = torch.matmul(a.to(torch_in_dtype_a), b.to(torch_in_dtype_b))  
    
    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])

    if out_dtype == 'int8':  
        ################### save c in result_gold ###################
        test_case_name = request.node.name
        sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
        result_gold[sanitized_key_name] = c.to(torch.float32).clone().detach().cpu()
        ###################################################################
        # torch.testing.assert_close(c.to(torch.float32),  
        #                            torch_output.to(torch.int8).to(torch.float32), atol=1e-3, rtol=1e-2)  
    else:  
        ################### save c in result_gold ###################
        test_case_name = request.node.name
        sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
        result_gold[sanitized_key_name] = c.clone().detach().cpu()
        ###################################################################
        # torch.testing.assert_close(c, torch_output.to(torch_out_dtype), atol=5e-3, rtol=1e-2)  
  
OP_NAME_FOR_BENCHMARK = "gemm_triton_perf"

@pytest.mark.parametrize(
    "M, N, K, in_dtype_a_str, in_dtype_b_str, out_dtype_str, col_a, col_b",
    [(*shape, ida, idb, od, ca, cb)
     for shape in get_x_vals() # Uses the reduced get_x_vals for faster testing
     for ida, idb, od in [
        ('fp16', 'fp16', 'fp16'), ('bf16', 'bf16', 'bf16'), ('fp32', 'fp32', 'fp32'),
        # ('fp8e4', 'fp8e4', 'fp16'), ('fp8e5', 'fp8e5', 'fp16'), # FP8 needs careful scale handling
        # ('int8', 'int8', 'int32') # Int8 also needs care
        ]
     for ca in [False] # Simplified: only row-major A for now
     for cb in [False]] # Simplified: only row-major B for now
)
def test_performance(M, N, K, col_a, col_b, in_dtype_a_str, in_dtype_b_str, out_dtype_str, request):
    set_seed() # Consistent seed for input data generation
    torch_in_dtype_a = name_to_torch_types[in_dtype_a_str]
    torch_in_dtype_b = name_to_torch_types[in_dtype_b_str]
    torch_out_dtype = name_to_torch_types[out_dtype_str]

    # --- Input Generation (from original test_correctness) ---
    # Determine fp8_scaling_mode based on dtypes, or make it a parameter
    fp8_mode_a = 'tensor' if dtype_is_8_bit(torch_in_dtype_a) else None
    fp8_mode_b = 'tensor' if dtype_is_8_bit(torch_in_dtype_b) else None

    a, a_fp32_ref, a_scale = gen_input(M, K, torch_in_dtype_a, col_a, seed=1, fp8_scaling_mode=fp8_mode_a or 'tensor', device='cuda')
    b, b_fp32_ref, b_scale = gen_input(K, N, torch_in_dtype_b, col_b, seed=2, fp8_scaling_mode=fp8_mode_b or 'tensor', device='cuda')
    c = torch.empty((M, N), device=a.device, dtype=torch_out_dtype)

    # Determine scale_a8_b8 and activation for matmul wrapper
    current_scale_a8_b8 = None
    current_activation = "" # No activation for pure GEMM perf
    if dtype_is_8_bit(torch_in_dtype_a) or dtype_is_8_bit(torch_in_dtype_b):
        current_scale_a8_b8 = 'tensor' # Or make this part of parametrize if testing block/token scaling perf

    # --- Create op_lambda for benchmarking ---
    # This lambda will call the matmul wrapper
    op_lambda = lambda: matmul(a, b, c, a_scale, b_scale,
                               scale_a8_b8=current_scale_a8_b8,
                               activation=current_activation)

    # --- Benchmarking ---
    bench_config = do_bench_config(warm_up=10, repetition=100) # Adjust for GEMM complexity
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)

    current_params_for_logs_and_calc = {
        "M": M, "N": N, "K": K,
        "in_dtype_a_str": in_dtype_a_str, "in_dtype_b_str": in_dtype_b_str, "out_dtype_str": out_dtype_str,
        "col_a": col_a, "col_b": col_b,
        "fp8_scaling_a": fp8_mode_a, "fp8_scaling_b": fp8_mode_b,
        "triton_scale_mode": current_scale_a8_b8, "activation": current_activation
    }

    benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
                              gbps_calculator=calculate_gemm_gbps,
                              tflops_calculator=calculate_gemm_tflops)
  
def get_type(provider):  
    res = re.findall(r'\(.*?\)', provider)  
    return res[0][1:-1].split('/', 1)  
  
  
@triton.testing.perf_report(  
    triton.testing.Benchmark(  
        x_names=['M', 'N', 'K'],  
        x_vals=get_x_vals(),  
        line_arg='provider',  
        line_vals=[  
            'hipblaslt(fp16/fp16)', 'hipblaslt(bf16/bf16)', 'triton(fp16/fp16)', 'triton(bf16/bf16)',  
            'triton(int8/int8)', 'triton(fp8e4/fp8e4)', 'triton(fp8e5/fp8e5)', 'triton(fp16/fp8e4)',  
            'triton(fp16/fp8e5)'  
        ],  
        line_names=[  
            "rocBLAS.Fp16", "rocBLAS.Bf16", "Triton.Fp16", "Triton.Bf16", "Triton.Int8", "Triton.Fp8E4", "Triton.Fp8E5",  
            "Triton.Fp16.Fp8E4", "Triton.Fp16.Fp8E5"  
        ],  
        ylabel="TFLOPS",  
        plot_name="matmul-performance",  
        args={},  
    ))  
def benchmark(M, N, K, provider, model=None, args=None):  
    in_dtype_a, in_dtype_b = [name_to_torch_types[x] for x in get_type(provider)]  
    out_dtype = in_dtype_a  
  
    quantiles = [0.5, 0.2, 0.8]  
    layout_tn = args.layout == 'tn'  
  
    if args.fp8_scaling_mode == 'tensor' or in_dtype_b == torch.int8:  
        a, _, a_scale = gen_input(M, K, in_dtype_a, False, seed=1, device='cuda')  
        b, _, b_scale = gen_input(K, N, in_dtype_b, layout_tn, seed=2, device='cuda')  
    else:  
        a, _, a_scale = gen_input(M, K, in_dtype_a, False, seed=1, fp8_scaling_mode='token', device='cuda')  
        b, _, b_scale = gen_input(K, N, in_dtype_b, layout_tn, seed=2, fp8_scaling_mode='block', device='cuda')  
  
    if 'hipblaslt' in provider:  
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)  
    else:  # triton, different data types  
        assert "triton" in provider  
        # Allocates output.  
        c = torch.empty((M, N), device=a.device, dtype=out_dtype)  
  
        # If data type is 8 bit  
        #   Default to tensor scaling if scaling mode is tensor or dtype is int8  
        #   Use block scaling otherwise  
        scale_a8_b8 = None  
        if dtype_is_8_bit(in_dtype_a) or dtype_is_8_bit(in_dtype_b):  
            scale_a8_b8 = 'tensor' if in_dtype_b == torch.int8 else args.fp8_scaling_mode  
  
        ms, min_ms, max_ms = triton.testing.do_bench(  
            lambda: matmul(a, b, c, a_scale, b_scale, scale_a8_b8=scale_a8_b8, activation=""), quantiles=quantiles)  
        if args.v:  
            print(f'Best tuning config for M={M}, N={N}, K={K}, '  
                  f'dtype={in_dtype_a} / {in_dtype_b} / {out_dtype}: \n({matmul_kernel.best_config})\n')  
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)  
    return perf(ms), perf(max_ms), perf(min_ms)  
  

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
    print(f"\nSaving all c_triton_dot results to {OUTPUT_FILENAME}...")  
    # Ensure the directory for the output file exists if it's in a subdirectory  
    output_dir = os.path.dirname(OUTPUT_FILENAME)  
    if output_dir and not os.path.exists(output_dir):  
        os.makedirs(output_dir, exist_ok=True)  
    torch.save(result_gold, OUTPUT_FILENAME)       
    print(f"Successfully saved {len(result_gold)} c_triton_dot tensors to {OUTPUT_FILENAME}.")  

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
  
def parse_args():  
    parser = argparse.ArgumentParser(  
        prog="AMD Triton GEMM kernel",  
        allow_abbrev=False,  
    )  
  
    parser.add_argument('-model_configs', type=str, default="model_configs.json", help="Model config json file.")  
  
    available_models = get_available_models(model_families=["llama3"])  # Dynamically load model names  
    model_help = (  
        "Model name to benchmark. Select from: [" + ", ".join(available_models) +  
        "]. Use 'all' to benchmark all models. Not providing runs the default benchmark script with custom configs.")  
    parser.add_argument('-model', type=str, default=None, help=model_help)  
    parser.add_argument("-v", action='store_true', default=False, help="Print out the best tuning config")  
    parser.add_argument("-M", type=int, default=0)  
    parser.add_argument("-N", type=int, default=0)  
    parser.add_argument("-K", type=int, default=0)  
    parser.add_argument("-layout", type=str, default='tn')  
    parser.add_argument("-dtype", type=str, default=None, help="Data type of inputs and outputs")  
    parser.add_argument("-b_dtype", type=str, default=None,  
                        help="Data type of B operand, if specified (else same as dtype)")  
    parser.add_argument("-fp8_scaling_mode", type=str, default='tensor', choices=['tensor', 'block'],  
                        help="Type of scaling to apply when either or both inputs are fp8")  
  
    args = parser.parse_args()  
  
    return args  
  
  
def get_line_vals_names(a_dtype=None, b_dtype=None):  
    line_vals = [  
        'hipblaslt(fp16/fp16)', 'hipblaslt(bf16/bf16)', 'triton(fp16/fp16)', 'triton(bf16/bf16)', 'triton(int8/int8)',  
        'triton(fp8e4/fp8e4)', 'triton(fp8e5/fp8e5)', 'triton(fp16/fp8e4)', 'triton(fp16/fp8e5)'  
    ]  
    line_names = [  
        "rocBLAS.Fp16", "rocBLAS.Bf16", "Triton.Fp16", "Triton.Bf16", "Triton.Int8", "Triton.Fp8E4", "Triton.Fp8E5",  
        "Triton.Fp16.Fp8E4", "Triton.Fp16.Fp8E5"  
    ]  
    assert not ((a_dtype is None) ^ (b_dtype is None))  
    if a_dtype is not None:  
        line_vals_suffix_str = '(' + a_dtype + '/' + b_dtype + ')'  
        line_names_suffix_str = '.' + a_dtype + '.' + b_dtype  
        line_vals = ['triton' + line_vals_suffix_str]  
        line_names = ['Triton' + line_names_suffix_str]  
        if (not dtype_is_8_bit(name_to_torch_types[a_dtype])
            and not dtype_is_8_bit(name_to_torch_types[b_dtype])):
            line_vals += ['hipblaslt' + line_vals_suffix_str]
            line_names += ['hipblaslt' + line_names_suffix_str]
  
    return line_vals, line_names  
  
  
def main():  
    args = parse_args()  
  
    if args.model:  
        config_file = args.model_configs  
        configs = get_model_configs(config_path=config_file, model_families=["llama3"], model=args.model)  
        mnk_list = []  
  
        for model_name, config in configs.items():  
            M, N, K = args.M or 8192, config["hidden_size"], config["intermediate_size"]  
            mnk_list.append((model_name, M, N, K))  
  
        benchmark.benchmarks.x_names = ['model', 'M', 'N', 'K']  
        benchmark.benchmarks.x_vals = mnk_list  
  
    a_dtype = args.dtype  
    b_dtype = args.b_dtype or args.dtype  
    assert a_dtype is None or a_dtype in name_to_torch_types, f"Unsupported dtype {a_dtype}"  
    assert b_dtype is None or b_dtype in name_to_torch_types, f"Unsupported dtype {b_dtype}"  
    benchmark.benchmarks.line_vals, benchmark.benchmarks.line_names = get_line_vals_names(a_dtype, b_dtype)  
    if args.N or args.K:  
        assert args.model is None, "Providing both -model and N/K is not compatible! -model already fixes N/K."  
  
    if args.M and args.N and args.K:  
        x_vals = [(args.M, args.N, args.K)]  
        benchmark.benchmarks.x_vals = x_vals  
  
    benchmark.run(show_plots=True, print_data=True, args=args)  
  
  
if __name__ == '__main__':  
    sys.exit(main())  
