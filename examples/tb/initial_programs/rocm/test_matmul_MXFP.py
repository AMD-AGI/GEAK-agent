# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
######################################## Imports ######################################## 
import pytest
import torch
import triton
import triton.language as tl
import triton.tools.experimental_descriptor
######################################## Imports ######################################## 



@triton.jit
def matmul_kernel(  #
        a_ptr, scale_ptr, b_ptr, output_ptr,  #
        M, N, K_MXFP,  # K_MXFP is the number of mxfp vectors in a row of a. Otherwise it's just K
        stride_am, stride_ak,  #
        stride_sm, stride_sk,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
        NUM_STAGES: tl.constexpr, a_type: tl.constexpr, b_type: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    IS_SCALED: tl.constexpr = a_type is not None and b_type is not None
    DIV_FACTOR: tl.constexpr = 2 if IS_SCALED and a_type == "e2m1" else 1
    # We pass K_MXFP to make explicit that KB is multiple of 32 and KA is multiple of 16 or 32
    # for the pipeliner divisibility condition
    KA = K_MXFP if not IS_SCALED else K_MXFP * (32 // DIV_FACTOR)
    KB = K_MXFP if not IS_SCALED else K_MXFP * 32
    BLOCK_AK: tl.constexpr = BLOCK_K // DIV_FACTOR
    offs_k = tl.arange(0, BLOCK_K)
    offs_ak = tl.arange(0, BLOCK_AK)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    if IS_SCALED:
        BLOCK_SK: tl.constexpr = BLOCK_K // 32
        offs_sk = tl.arange(0, BLOCK_SK)
        scale_ptrs = scale_ptr + (offs_am[:, None] * stride_sm + offs_sk[None, :] * stride_sk)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(KB, BLOCK_K), num_stages=NUM_STAGES):
        mask_a = (offs_am[:, None] < M) & (offs_ak[None, :] + k * BLOCK_AK < KA)
        mask_b = ((offs_k[:, None] + k * BLOCK_K) < KB) & (offs_bn[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0)
        b = tl.load(b_ptrs, mask=mask_b, other=0)
        if IS_SCALED:
            # Adapted scale indexing and dot_scaled operation
            mask_scale = (offs_am[:, None] < M) & (offs_sk[None, :] + k * BLOCK_SK < K_MXFP)
            a_scale = tl.load(scale_ptrs, mask=mask_scale, other=0)
            accumulator = tl.dot_scaled(a, a_scale, a_type, b, None, b_type, acc=accumulator)
        else:
            accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_AK * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        if IS_SCALED:
            scale_ptrs += BLOCK_SK * stride_sk
    OUT_DTYPE = tl.bfloat16 if IS_SCALED else tl.float16
    accumulator = accumulator.to(OUT_DTYPE)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(output_ptrs, accumulator, mask=mask_c)


@triton.jit
def mxfp_to_bf16_kernel(
    x_ptr,
    scale_ptr,
    mxfp_ptr,
    N,
    e_bits: tl.constexpr,
    m_bits: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # x.shape ==     (N, 32) for fp8 or (N, 16) for fp4
    # scale.shape == (N,)
    # out.shape   == (N, 32)
    is_fp8: tl.constexpr = e_bits + m_bits == 7
    # fp8: BLOCK_SIZE -> BLOCK_SIZE // 32, 32
    # fp4: BLOCK_SIZE // 2 -> BLOCK_SIZE // 32 , 16
    PARALLEL_DIM: tl.constexpr = BLOCK_SIZE // 32
    LAST_DIM: tl.constexpr = 32 if is_fp8 else 16
    LOAD_SIZE: tl.constexpr = LAST_DIM * PARALLEL_DIM

    offsets = (tl.program_id(0) * LOAD_SIZE + tl.arange(0, PARALLEL_DIM)[:, None] * LAST_DIM +
               tl.arange(0, LAST_DIM)[None, :])
    x = tl.load(x_ptr + offsets, mask=offsets < N * LAST_DIM)

    offsets = tl.program_id(0) * PARALLEL_DIM + tl.arange(0, PARALLEL_DIM)[:, None]
    scale = tl.load(scale_ptr + offsets, mask=offsets < N)
    tl.static_assert(scale.dtype == tl.uint8)
    tl.static_assert(x.dtype == tl.uint8)

    scale_bf16 = (scale.to(tl.uint16) << 7).to(tl.bfloat16, bitcast=True)
    if is_fp8:
        if e_bits == 5 and m_bits == 2:
            x_f8 = x.to(tl.float8e5, bitcast=True)
            x_bf16 = x_f8.to(tl.bfloat16)
            # Preserve infs and nans. FIXME Fp8E5M2_to_Bf16 doesn't preserve them!
            non_finite_mask: tl.constexpr = ((1 << e_bits) - 1) << m_bits
            non_finite_mask_bf16: tl.constexpr = ((1 << 8) - 1) << 7
            x_bf16 = tl.where(
                x & non_finite_mask == non_finite_mask,
                (x_bf16.to(tl.uint16, bitcast=True) | non_finite_mask_bf16).to(tl.bfloat16, bitcast=True),
                x_bf16,
            )
        else:
            tl.static_assert(e_bits == 4 and m_bits == 3)
            x_f8 = x.to(tl.float8e4nv, bitcast=True)
            x_bf16 = x_f8.to(tl.bfloat16)
    else:
        # e2m1
        em0 = x & 0x70
        em1 = x & 0x7
        x0 = (em0.to(tl.uint16) << 2) | ((x & 0x80).to(tl.uint16) << 8)
        x1 = (em1.to(tl.uint16) << (2 + 4)) | ((x & 0x8).to(tl.uint16) << (8 + 4))
        # Three cases:
        # 1) x is normal and non-zero: Correct bias
        x0 = tl.where((em0 & 0x60) != 0, x0 + ((127 - 1) << 7), x0)
        x1 = tl.where((em1 & 0x6) != 0, x1 + ((127 - 1) << 7), x1)
        # 2) x is subnormal (x == 0bs001 where s is the sign): Map to +-0.5 in bf16
        x0 = tl.where(em0 == 0x10, 16128 | (x0 & 0x8000), x0)
        x1 = tl.where(em1 == 0x1, 16128 | (x1 & 0x8000), x1)
        # 3) x is zero, do nothing
        x_bf16 = tl.interleave(x0, x1).to(tl.bfloat16, bitcast=True)
    # Multiplication preserves infs and NaNs in x_bf16
    mxfp = x_bf16 * scale_bf16
    # If scale is NaN, we encode it as an bf16 inf, so we need to correct for that
    mxfp = tl.where(scale == 0xFF, float("nan"), mxfp)

    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(mxfp_ptr + offsets, tl.ravel(mxfp), mask=offsets < N * 32)


##################################################################################################################################################  

import numpy as np
import random
import torch 
import os
import pytest
from numpy.random import RandomState

result_gold = {}
from geak_eval.perf.ROCm.performance_utils_pytest import PytestBenchmarker, do_bench_config, save_all_benchmark_results
from typing import Dict

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


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'


def check_capabilities():
    if is_cuda():
        cc = torch.cuda.get_device_capability()
        if cc[0] < 8:
            pytest.skip("CUDA 8.0+ required")


def dot_scale_ref(x, scale, y, type_x, type_y):
    e_bits, m_bits = {"e2m1": (2, 1), "e4m3": (4, 3), "e5m2": (5, 2)}[type_x]
    type_fp8_y = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[type_y]

    comp_dtype = torch.float32
    out_dtype = torch.bfloat16

    x = x.contiguous()
    x_upcast = x.new_empty(scale.shape[:-1] + (32 * scale.shape[-1], ), dtype=comp_dtype)

    N = x_upcast.numel()
    BLOCK_SIZE = 512
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    mxfp_to_bf16_kernel[grid](x, scale, x_upcast, scale.numel(), e_bits, m_bits, BLOCK_SIZE, num_warps=4)
    y_upcast = y.view(type_fp8_y)

    class AccumulateInFp32:

        def __enter__(self):
            self.prev_value = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        def __exit__(self, exc_type, exc_val, exc_tb):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = self.prev_value

    with AccumulateInFp32():
        return torch.matmul(x_upcast.to(out_dtype), y_upcast.to(out_dtype))


@pytest.mark.parametrize("scale", [True, False])
def test_pipeline_matmul(scale, request, device='cuda'):
    check_capabilities()
    set_seed()
    if scale and not is_cuda():
        pytest.skip("NYI: scale_dot just implemented in CUDA")
    M, N, K = 512, 512, 128
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    NUM_STAGES = 4

    if scale:
        # TODO Use e5m2 for Ampere, as it does not support fp_to_fp conversions for fp8e4m3
        BLOCK_K = 64  # 32 NYI
        K = BLOCK_K * NUM_STAGES
        a_type = "e2m1"
        DIV_FACTOR = 2 if a_type == "e2m1" else 1
        a = torch.randint(256, (M, K // DIV_FACTOR), device=device, dtype=torch.uint8)
        # Sample small-ish scales to avoid overflow
        scale_a = torch.randint(74, (M, K // 32), device=device, dtype=torch.uint8)
        # Ampere does not support fp8e4m3
        b_type = "e4m3" if is_hopper() else "e5m2"
        b = torch.randint(256, (K, N), device=device, dtype=torch.uint8)
        # e5m2 has too many non-finite values when sampled uniformly (1 / 32) and
        # Fp8E5M2_to_Bf16 doesn't preserve NaNs (fixme)
        if b_type == "e5m2":
            finite = torch.arange(K * N, device=device, dtype=torch.uint8).reshape(K, N) % 0x7C
            b = torch.where(b & 0x7C == 0x7C, finite | (0x80 & b), b)
        output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    else:
        a = torch.randn(M, K, device=device, dtype=torch.float16)
        b = torch.randn(K, N, device=device, dtype=torch.float16)
        scale_a = None
        a_type, b_type = None, None
        output = torch.empty((M, N), dtype=torch.float16, device=device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)


    # Pass K_MXFP to make explicit that KB is multiple of 32 and KA is multiple of 16 or 32º
    if scale:
        K = scale_a.shape[-1]
    stride_sm, stride_sk = scale_a.stride() if scale else (0, 0)
    handler = matmul_kernel[grid](a, scale_a, b, output, M, N, K, a.stride(0), a.stride(1), stride_sm, stride_sk,
                                    b.stride(0), b.stride(1), output.stride(0), output.stride(1), BLOCK_M, BLOCK_N,
                                    BLOCK_K, NUM_STAGES=NUM_STAGES, a_type=a_type, b_type=b_type)
    if scale:
        ref_out = dot_scale_ref(a, scale_a, b, a_type, b_type)
    else:
        ref_out = torch.matmul(a, b)
    # Bigger tolerance for AMD MI200 devices.
    # MI200 devices use reduced precision fp16 and bf16 and flush input and
    # output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    atol = 1e-2 if is_hip_mi200() or scale else None
    rtol = 1e-2 if is_hip_mi200() or scale else None

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = output.clone().detach().cpu()
    ################################################################### 


    torch.testing.assert_close(ref_out, output, atol=atol, rtol=rtol, equal_nan=scale)


# Define these globally so they are accessible by test_matmul_mxfp_performance
FIXED_BLOCK_M_perf = 64
FIXED_BLOCK_N_perf = 64
FIXED_BLOCK_K_tile_perf = 32 # This is BLOCK_K in kernel, for tiling K_eff
FIXED_NUM_STAGES_perf = 2    
FIXED_NUM_WARPS_perf = 4     

# --- Python wrapper for the kernel for benchmarking (UNCHANGED from previous corrected version) ---
def matmul_mxfp_triton_wrapper(a_tensor, scale_a_tensor, b_tensor, output_buffer,
                               M_dim, N_dim, K_runtime_dim, 
                               block_m_const, block_n_const, block_k_const_tile, 
                               num_stages_const, a_type_str, b_type_str,
                               num_warps_launch): 
    grid = (triton.cdiv(M_dim, block_m_const) * triton.cdiv(N_dim, block_n_const), 1)
    stride_sm, stride_sk = (scale_a_tensor.stride(0), scale_a_tensor.stride(1)) \
                           if scale_a_tensor is not None else (0,0)
    matmul_kernel[grid](
        a_tensor, scale_a_tensor, b_tensor, output_buffer,
        M_dim, N_dim, K_runtime_dim, 
        a_tensor.stride(0), a_tensor.stride(1),
        stride_sm, stride_sk,
        b_tensor.stride(0), b_tensor.stride(1),
        output_buffer.stride(0), output_buffer.stride(1),
        BLOCK_M=block_m_const, BLOCK_N=block_n_const, BLOCK_K=block_k_const_tile, 
        NUM_STAGES=num_stages_const, a_type=a_type_str, b_type=b_type_str,
        num_warps=num_warps_launch
    )
    return output_buffer

# --- CORRECTED TFLOPS and GB/s Calculators ---
def calculate_mxfp_matmul_tflops(params: dict, ms: float) -> float:
    M, N = params['M'], params['N']
    K_for_flops = params['K_for_flops'] # Use the K dim relevant for FLOPs
    flops = 2 * M * N * K_for_flops
    tflops = flops / (ms / 1000) / 1e12
    return tflops

def get_torch_dtype_from_str(dtype_str: str, default_dtype=torch.float16):
    if dtype_str == 'fp32': return torch.float32
    if dtype_str == 'bf16': return torch.bfloat16
    if dtype_str == 'fp16': return torch.float16
    if dtype_str == 'uint8': return torch.uint8
    return default_dtype

def calculate_mxfp_matmul_gbps(params: dict, ms: float) -> float:
    M, N = params['M'], params['N']
    K_for_A_storage = params['K_for_A_storage'] 
    K_for_B_storage = params['K_for_B_storage'] 
    
    is_scaled = params['is_scaled_mode']
    
    if is_scaled:
        elem_size_a_data = torch.tensor([], dtype=torch.uint8).element_size()
        elem_size_scale_a = torch.tensor([], dtype=torch.uint8).element_size()
        elem_size_b_data = torch.tensor([], dtype=torch.uint8).element_size()
        elem_size_out = torch.tensor([], dtype=torch.bfloat16).element_size()
        K_mxfp_vectors_for_scale = params['K_MXFP_runtime'] 
    else: 
        input_dtype_str_calc = params.get('input_dtype_str', 'fp16')
        output_dtype_str_calc = params.get('output_dtype_str_actual', 'fp16')

        torch_input_dtype = get_torch_dtype_from_str(input_dtype_str_calc)
        torch_output_dtype = get_torch_dtype_from_str(output_dtype_str_calc)
        
        elem_size_a_data = torch.tensor([], dtype=torch_input_dtype).element_size()
        elem_size_b_data = torch.tensor([], dtype=torch_input_dtype).element_size()
        elem_size_out = torch.tensor([], dtype=torch_output_dtype).element_size()
        elem_size_scale_a = 0 
        K_mxfp_vectors_for_scale = 0

    bytes_a = M * K_for_A_storage * elem_size_a_data
    bytes_b = K_for_B_storage * N * elem_size_b_data 
    bytes_scale_a = M * K_mxfp_vectors_for_scale * elem_size_scale_a if is_scaled and elem_size_scale_a > 0 else 0
    bytes_out_write = M * N * elem_size_out
    
    total_bytes = bytes_a + bytes_b + bytes_scale_a + bytes_out_write
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

OP_NAME_FOR_BENCHMARK = "matmul_mxfp_triton_perf"

MXFP_PERF_CONFIGS = []
shapes_perf = [(512, 512, 128), (1024, 1024, 64), (2048, 1024, 32)] 

for M_val, N_val, K_actual_val in shapes_perf:
    for input_dtype_s in ['fp16', 'fp32']: 
        MXFP_PERF_CONFIGS.append({
            'M':M_val, 'N':N_val, 'K_param':K_actual_val, 'is_scaled':False, 
            'a_type':None, 'b_type':None, 'input_dtype':input_dtype_s
        })

if is_cuda() and torch.cuda.get_device_capability()[0] >=9 : 
    for M_val, N_val, K_mxfp_val in shapes_perf:
        if K_mxfp_val == 0 : continue # K_mxfp must be > 0 for scaled mode
        if K_mxfp_val % (FIXED_BLOCK_K_tile_perf // (32 // 2)) != 0 and K_mxfp_val % (FIXED_BLOCK_K_tile_perf // (32 // 1)) != 0 :
             # K_MXFP must be such that KA_eff and KB_eff are reasonable for K_tile
             # This check is complex, for now assume K_mxfp_val is okay.
             # A simpler check: K_mxfp_val should be multiple of BLOCK_SK_eff (BLOCK_K_tile/32)
             if FIXED_BLOCK_K_tile_perf % 32 == 0 and K_mxfp_val % (FIXED_BLOCK_K_tile_perf // 32) != 0:
                 # print(f"Skipping K_mxfp={K_mxfp_val} with BLOCK_K_tile={FIXED_BLOCK_K_tile_perf} due to scale alignment")
                 continue

        MXFP_PERF_CONFIGS.append({ 
            'M':M_val, 'N':N_val, 'K_param':K_mxfp_val, 'is_scaled':True, 
            'a_type':'e2m1', 'b_type':'e4m3', 'input_dtype':'uint8'
        })
        if is_hopper(): 
            MXFP_PERF_CONFIGS.append({ 
                'M':M_val, 'N':N_val, 'K_param':K_mxfp_val, 'is_scaled':True, 
                'a_type':'e2m1', 'b_type':'e5m2', 'input_dtype':'uint8'
            })

@pytest.mark.parametrize("test_cfg_dict", MXFP_PERF_CONFIGS)
def test_performance(test_cfg_dict, request):
    check_capabilities() 
    set_seed()
    device = 'cuda'

    M = test_cfg_dict['M']
    N = test_cfg_dict['N']
    K_param = test_cfg_dict['K_param'] 
    is_scaled_mode = test_cfg_dict['is_scaled']
    a_type_kernel_str = test_cfg_dict['a_type'] 
    b_type_kernel_str = test_cfg_dict['b_type'] 
    
    K_MXFP_runtime = 0
    K_eff_A_storage = 0 
    K_eff_B_storage = 0 
    K_for_flops_calc = 0 
    scale_a_tensor = None 
    input_dtype_for_calc = test_cfg_dict['input_dtype'] 

    if is_scaled_mode:
        if not is_cuda(): pytest.skip("Scaled MXFP test part currently CUDA-specific")
        if a_type_kernel_str is None or b_type_kernel_str is None :
             pytest.skip("a_type and b_type must be specified for scaled mode")

        K_MXFP_runtime = K_param 
        DIV_FACTOR = 2 if a_type_kernel_str == "e2m1" else 1
        
        K_eff_A_storage = K_MXFP_runtime * (32 // DIV_FACTOR)
        K_eff_B_storage = K_MXFP_runtime * 32            
        
        # For dot(A(M,K1), B(K1,N)), K_for_flops is K1.
        # Here a is (M, KA_eff) and b is (KB_eff, N) for the kernel's view.
        # The dot_scaled implies A's K dim must match B's K dim logically.
        # KA_eff = K_MXFP * (32/DIV_FACTOR), KB_eff = K_MXFP * 32.
        # These are generally different unless DIV_FACTOR=1.
        # tl.dot_scaled is flexible: `dot_scaled(a, scale_a, "eXmY", b, scale_b, "eZmQ", acc)`
        # It handles the internal expansion. The logical K for FLOPs is related to K_MXFP * 32.
        K_for_flops_calc = K_eff_B_storage # Use the larger effective K from B for FLOPs

        if K_eff_A_storage == 0 or K_eff_B_storage == 0 : pytest.skip("Effective K is zero for scaled mode.")

        a_tensor = torch.randint(256, (M, K_eff_A_storage), device=device, dtype=torch.uint8)
        scale_a_tensor = torch.randint(74, (M, K_MXFP_runtime), device=device, dtype=torch.uint8)
        b_tensor = torch.randint(256, (K_eff_B_storage, N), device=device, dtype=torch.uint8)
        output_buffer = torch.empty((M, N), dtype=torch.bfloat16, device=device)
        actual_output_dtype_str = "bf16"
    else: 
        K_MXFP_runtime = K_param 
        K_eff_A_storage = K_param
        K_eff_B_storage = K_param
        K_for_flops_calc = K_param
        
        current_input_torch_dtype = get_torch_dtype_from_str(input_dtype_for_calc)
        
        a_tensor = torch.randn(M, K_param, device=device, dtype=current_input_torch_dtype)
        b_tensor = torch.randn(K_param, N, device=device, dtype=current_input_torch_dtype)
        output_buffer = torch.empty((M, N), dtype=torch.float16, device=device)
        actual_output_dtype_str = "fp16"

    # Use globally defined fixed block sizes for performance test
    block_m_const = FIXED_BLOCK_M_perf
    block_n_const = FIXED_BLOCK_N_perf
    block_k_const_tile = FIXED_BLOCK_K_tile_perf 
    num_stages_const = FIXED_NUM_STAGES_perf
    num_warps_launch = FIXED_NUM_WARPS_perf

    op_lambda = lambda: matmul_mxfp_triton_wrapper(
        a_tensor, scale_a_tensor, b_tensor, output_buffer,
        M, N, K_MXFP_runtime, 
        block_m_const, block_n_const, block_k_const_tile,
        num_stages_const, a_type_kernel_str, b_type_kernel_str,
        num_warps_launch
    )

    bench_config = do_bench_config(warm_up=10, repetition=50)
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)
    current_params_for_logs_and_calc = {
        "M": M, "N": N, 
        "K_param_input": K_param, 
        "K_MXFP_runtime": K_MXFP_runtime,
        "K_for_flops": K_for_flops_calc, 
        "K_for_A_storage": K_eff_A_storage, 
        "K_for_B_storage": K_eff_B_storage, 
        "is_scaled_mode": is_scaled_mode, 
        "a_type_str": a_type_kernel_str, "b_type_str": b_type_kernel_str,
        "input_dtype_str": input_dtype_for_calc, 
        "output_dtype_str_actual": actual_output_dtype_str, 
        "BLOCK_M": block_m_const, "BLOCK_N": block_n_const, "BLOCK_K_tile": block_k_const_tile,
        "NUM_STAGES": num_stages_const, "num_warps": num_warps_launch
    }
    
    perf_result = benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
                                            gbps_calculator=calculate_mxfp_matmul_gbps,
                                            tflops_calculator=calculate_mxfp_matmul_tflops)




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