# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
"""
Testing the (FP8) case of a dot op that consumes the output (MFMA) of
another dot op as an input.

"""
######################################## Imports#######################################
import math
import pytest
import torch

import triton
import triton.language as tl
######################################## Imports#######################################

########################## HELPER utils ##########################
TORCH_HAS_FP8E4 = hasattr(torch, 'float8_e4m3fnuz')
float8: tl.constexpr = None if not TORCH_HAS_FP8E4 else torch.float8_e4m3fnuz
########################## HELPER utils ##########################


@triton.jit
def _chained_dot(
    Q,
    K,
    V,
    Out,
    q_desc,
    k_desc,
    v_desc,
    s_sc,
    s_desc,
    o_sc,
    stride_qz,
    stride_qm,
    stride_qd,
    stride_kz,
    stride_kn,
    stride_kd,
    stride_vz,
    stride_vd,
    stride_vn,
    stride_oz,
    stride_om,
    stride_od,
    Z,
    M,
    N,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_FP8: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_z = tl.program_id(1)
    qkv_offset = off_z * stride_qz
    Q_block_ptr = tl.make_block_ptr(base=Q + qkv_offset, shape=(N, BLOCK_D), strides=(stride_qm, stride_qd),
                                    offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_D), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + qkv_offset, shape=(BLOCK_D, N), strides=(stride_kd, stride_kn),
                                    offsets=(0, 0), block_shape=(BLOCK_D, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(base=V + qkv_offset, shape=(N, BLOCK_D), strides=(stride_vn, stride_vd),
                                    offsets=(0, 0), block_shape=(BLOCK_N, BLOCK_D), order=(0, 1))

    s_scale = q_desc * k_desc * s_sc
    acc_scale = s_desc * v_desc * o_sc

    q = tl.load(Q_block_ptr)

    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    lo, hi = 0, N
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = tl.load(K_block_ptr)
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, k)

        if USE_FP8:
            s *= s_scale

        v = tl.load(V_block_ptr)
        acc += tl.dot(s.to(v.dtype), v)

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    if USE_FP8:
        acc *= acc_scale

    O_block_ptr = tl.make_block_ptr(base=Out + qkv_offset, shape=(N, BLOCK_D), strides=(stride_om, stride_od),
                                    offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_D), order=(1, 0))
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

##################################################################################################################################################  

import numpy as np
import random
import torch 
import os
from tb_eval.perf.ROCm.performance_utils_pytest import PytestBenchmarker, do_bench_config, save_all_benchmark_results
from typing import Dict
########################## HELPER utils ##########################
TORCH_HAS_FP8E4 = hasattr(torch, 'float8_e4m3fnuz')
float8: tl.constexpr = None if not TORCH_HAS_FP8E4 else torch.float8_e4m3fnuz
########################## HELPER utils ##########################


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
class chained_dot_fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, msize=32, q_desc=1.0, k_desc=1.0, v_desc=1.0, s_sc=1.0, s_desc=1.0, o_sc=1.0):
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-2]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        assert msize in {16, 32}
        o = torch.empty_like(q, dtype=v.dtype)

        BLOCK_M = 128 if q.dtype == float8 else 256
        if BLOCK_M > q.shape[1]:
            BLOCK_M = int(math.pow(2, math.floor(math.log2(q.shape[1]))))
        BLOCK_N = 32
        if BLOCK_N > k.shape[1]:
            BLOCK_N = int(math.pow(2, math.floor(math.log2(k.shape[1]))))
        waves_per_eu = 2
        num_warps = 4 if q.dtype == float8 else 8
        num_stages = 1

        grid = (triton.cdiv(q.shape[1], BLOCK_M), q.shape[0], 1)

        _chained_dot[grid](q, k, v, o, q_desc,
                           k_desc, v_desc, s_sc, s_desc, o_sc, q.stride(0), q.stride(1), q.stride(2), k.stride(0),
                           k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2), o.stride(0), o.stride(1),
                           o.stride(2), Z=q.shape[0], M=q.shape[1], N=k.shape[1], 
                        #    BLOCK_D=Lk, BLOCK_M=BLOCK_M,
                        #    BLOCK_N=BLOCK_N, USE_FP8=(q.dtype == float8), waves_per_eu=waves_per_eu, num_warps=num_warps,
                        #    num_stages=num_stages, matrix_instr_nonkdim=msize
                           )

        return o


chained_dot = chained_dot_fn.apply


def to_float8(x, dtype=float8, margin: float = 1.0):
    finfo = torch.finfo(dtype)
    scale = finfo.max / x.abs().max().clamp(min=1e-12)
    scale = math.pow(2, math.floor(math.log2(scale.float().item())) - margin)
    x_scaled = (x.float() * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scaled.to(dtype), scale, 1.0 / scale


@pytest.mark.parametrize('M, N, D, dtype, msize', [(*shape, dtype, msize)
                                                   for shape in [(128, 64, 32), (256, 128, 128)]
                                                   for dtype in ['fp8']
                                                   for msize in [16, 32]])
def test_chained_dot(M, N, D, dtype, msize,request):
    set_seed()
    if dtype == 'fp8':
        assert float8 is not None

    BATCH = 1
    q = torch.empty((BATCH, M, D), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    k = torch.empty((BATCH, N, D), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    v = torch.empty((BATCH, D, N), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)

    if dtype == 'fp8':
        q_f8, _, q_desc = to_float8(q)
        k_f8, _, k_desc = to_float8(k)
        v_f8, _, v_desc = to_float8(v)

        s = torch._scaled_mm(q_f8[0], k_f8[0].transpose(0, 1), out_dtype=torch.float32,
                             scale_a=torch.tensor(q_desc, dtype=torch.float32, device="cuda"),
                             scale_b=torch.tensor(k_desc, dtype=torch.float32, device="cuda"))
        s_f8, s_sc, s_desc = to_float8(s)
        ref = torch._scaled_mm(s_f8, v_f8[0].transpose(0, 1), out_dtype=torch.float32,
                               scale_a=torch.tensor(s_desc, dtype=torch.float32, device="cuda"),
                               scale_b=torch.tensor(v_desc, dtype=torch.float32, device="cuda"))
        ref_f8, ref_sc, _ = to_float8(ref)

        tri_out = chained_dot(q_f8, k_f8, v_f8, msize, q_desc, k_desc, v_desc, s_sc, s_desc, ref_sc)
        result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
        ################### save tri_out in result_gold ###################
        test_case_name = request.node.name
        sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_") + "_fwd"
        result_gold[sanitized_key_name] = tri_out[0].float().clone().detach().cpu()
        ###################################################################


        assert tri_out.isnan().sum() == 0
        torch.testing.assert_close(tri_out[0].float(), ref_f8.float(), atol=1e-2, rtol=0)

    else:
        s = torch.matmul(q, k.transpose(1, 2))
        ref = torch.matmul(s, v.transpose(1, 2))

        tri_out = chained_dot(q, k, v, msize)
        result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
        ################### save tri_out in result_gold ###################
        test_case_name = request.node.name
        sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
        result_gold[sanitized_key_name] = tri_out.clone().detach().cpu()
        ###################################################################

        torch.testing.assert_close(tri_out, ref, atol=1e-2, rtol=0)

# --- Define TFLOPS and GB/s calculators ---
def calculate_chained_dot_tflops(params: dict, ms: float) -> float:
    BATCH, M, N_kv, D = params['BATCH'], params['M_seqlen'], params['N_kv_seqlen'], params['D_head']
    # S = Q(M,D) @ K.T(D,N_kv) -> 2*M*N_kv*D
    # O = S(M,N_kv) @ V.T(N_kv,D) -> 2*M*N_kv*D (assuming V value dim is D)
    flops = BATCH * ( (2 * M * N_kv * D) + (2 * M * N_kv * D) )
    tflops = flops / (ms / 1000) / 1e12
    return tflops

def calculate_chained_dot_gbps(params: dict, ms: float) -> float:
    BATCH, M, N_kv, D = params['BATCH'], params['M_seqlen'], params['N_kv_seqlen'], params['D_head']
    dtype_str = params.get('dtype_str', 'fp16')
    
    element_size = 1 if dtype_str == 'fp8' and TORCH_HAS_FP8E4 and float8 is not None else \
                   (4 if dtype_str == 'fp32' else 2) # Default to 2 bytes for fp16/bf16

    bytes_q = BATCH * M * D * element_size
    bytes_k = BATCH * N_kv * D * element_size
    # V is passed as (B,D,N_kv) to kernel, effectively same number of elements as K
    bytes_v = BATCH * D * N_kv * element_size 
    bytes_o = BATCH * M * D * element_size # Output O is (B,M,D)
    total_bytes = bytes_q + bytes_k + bytes_v + bytes_o
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

OP_NAME_FOR_BENCHMARK = "chained_dot_fp8_perf"

# --- Pytest parametrize for performance testing ---
CHAINED_DOT_PERF_CONFIGS = []
# M, N_kv, D_head
base_shapes_perf = [(128, 64, 32), (256, 128, 64), (512, 128, 128), (1024, 256, 128), (2048, 512, 64)]
dtypes_perf_list = ['fp16']
if TORCH_HAS_FP8E4 and float8 is not None: # Check if the global float8 (torch.dtype or None) is set
    dtypes_perf_list.append('fp8')
msize_args_perf_list = [16, 32] # msize from original test

for shape_cfg in base_shapes_perf:
    m_val, n_kv_val, d_val = shape_cfg
    for dtype_s_val in dtypes_perf_list:
        for msize_val_val in msize_args_perf_list:
            CHAINED_DOT_PERF_CONFIGS.append({
                'M':m_val, 'N_kv':n_kv_val, 'D':d_val, 
                'dtype_str':dtype_s_val, 'msize':msize_val_val
            })

@pytest.mark.parametrize("test_config", CHAINED_DOT_PERF_CONFIGS)
def test_performance(test_config, request):
    set_seed()
    M_seqlen = test_config['M']
    N_kv_seqlen = test_config['N_kv']
    D_head = test_config['D']
    dtype_str = test_config['dtype_str']
    msize_arg = test_config['msize']

    BATCH = 2 # Example batch size for performance test

    # Prepare inputs (always start with fp16 for data generation simplicity before potential cast)
    q_host = torch.empty((BATCH, M_seqlen, D_head), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    k_host = torch.empty((BATCH, N_kv_seqlen, D_head), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)
    # V is passed as (B, D, N_kv) to chained_dot call in original test
    v_host_for_call = torch.empty((BATCH, D_head, N_kv_seqlen), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5)

    q_for_kernel, k_for_kernel, v_for_kernel_call = q_host, k_host, v_host_for_call
    # Python floats for scales, matching chained_dot_fn.forward signature
    q_desc_py, k_desc_py, v_desc_py = 1.0, 1.0, 1.0
    s_sc_py, s_desc_py, o_sc_py = 1.0, 1.0, 1.0

    if dtype_str == 'fp8':
        if not (TORCH_HAS_FP8E4 and float8 is not None): # Check global float8
            pytest.skip("FP8 (e4m3fnuz) not available or global float8 is None.")
        
        # to_float8's default dtype is the global `float8` (torch.dtype or None)
        q_f8, _, q_desc_float = to_float8(q_host) # Returns Python float for scales
        k_f8, _, k_desc_float = to_float8(k_host)
        v_f8, _, v_desc_float = to_float8(v_host_for_call)
        
        q_for_kernel, k_for_kernel, v_for_kernel_call = q_f8, k_f8, v_f8
        q_desc_py, k_desc_py, v_desc_py = q_desc_float, k_desc_float, v_desc_float

        # Derive s_sc_py, s_desc_py, o_sc_py using the reference path from original test
        # Use BATCH=0 for these calculations as original test did
        s_ref_fp32 = torch._scaled_mm(
            q_f8[0], k_f8[0].transpose(0, 1), 
            out_dtype=torch.float32,
            scale_a=torch.tensor(q_desc_py, dtype=torch.float32, device="cuda"), # _scaled_mm needs tensor scales
            scale_b=torch.tensor(k_desc_py, dtype=torch.float32, device="cuda")
        )
        _s_f8_ref, s_sc_float, s_desc_float = to_float8(s_ref_fp32)
        
        # v_f8[0] is (D, N_kv). Transpose to (N_kv, D) for S(M,N_kv) @ V.T(N_kv,D)
        o_ref_fp32 = torch._scaled_mm(
            _s_f8_ref, v_f8[0].transpose(0, 1), 
            out_dtype=torch.float32, 
            scale_a=torch.tensor(s_desc_float, dtype=torch.float32, device="cuda"), 
            scale_b=torch.tensor(v_desc_float, dtype=torch.float32, device="cuda")
        )
        _o_f8_ref, o_sc_float, _ = to_float8(o_ref_fp32)

        s_sc_py, s_desc_py, o_sc_py = s_sc_float, s_desc_float, o_sc_float
    
    # --- Create op_lambda for benchmarking ---
    op_lambda = lambda: chained_dot( # This is chained_dot_fn.apply
        q_for_kernel, k_for_kernel, v_for_kernel_call, msize_arg,
        q_desc_py, k_desc_py, v_desc_py,
        s_sc_py, s_desc_py, o_sc_py
    )

    # --- Benchmarking ---
    bench_config = do_bench_config(warm_up=10, repetition=50) # Adjust reps as needed
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)

    current_params_for_logs_and_calc = {
        "BATCH": BATCH, "M_seqlen": M_seqlen, "N_kv_seqlen": N_kv_seqlen, "D_head": D_head,
        "dtype_str": dtype_str, "msize_arg": msize_arg
    }
    benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
                              gbps_calculator=calculate_chained_dot_gbps,
                              tflops_calculator=calculate_chained_dot_tflops)
    
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