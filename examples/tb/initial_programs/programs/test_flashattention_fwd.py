# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
Fused Attention
===============
This is a Triton implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf; Rabe and Staats https://arxiv.org/pdf/2112.05682v2.pdf)
"""
######################################## Imports ######################################## 

# import numpy as np
import pytest
import torch

import triton
import triton.language as tl
######################################## Imports ######################################## 


@triton.jit
def flash_fwd_kernel(Q, K, V, sm_scale,  #
                L, M,  #
                Out,  #
                stride_qz, stride_qh, stride_qm, stride_qk,  #
                stride_kz, stride_kh, stride_kn, stride_kk,  #
                stride_vz, stride_vh, stride_vk, stride_vn,  #
                stride_oz, stride_oh, stride_om, stride_on,  #
                Z, H, N_CTX, D0,  #
                BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # TODO: may replace with TMA store without range offset
    # initialize offsets for store
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    stride_qh_2d = stride_qh // stride_qm // stride_qk

    q_tile_ptr = tl.make_block_ptr(
        base=Q,
        shape=(D0, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(off_hz * stride_qh_2d + start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    k_tile_ptr = tl.make_block_ptr(
        base=K,
        shape=(D0, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(off_hz * stride_qh_2d, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    v_tile_ptr = tl.make_block_ptr(
        base=V,
        shape=(D0, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(off_hz * stride_qh_2d, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    out_tile_ptr = tl.make_block_ptr(
        base=Out,
        shape=(D0, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(off_hz * stride_qh_2d + start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # load q: it will stay in SRAM throughout
    q = tl.load(q_tile_ptr)

    # loop over k, v and update accumulators
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        # -- compute qk ----
        k = tl.load(k_tile_ptr, boundary_check=(0, 1))
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        # compute new m
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        # correct old l
        l_prev *= tl.exp(m_prev - m_curr)
        # attention weights
        p = tl.exp(qk - m_curr[:, None])
        l_curr = tl.sum(p, 1) + l_prev
        # rescale operands of matmuls
        l_rcp = 1. / l_curr
        p *= l_rcp[:, None]
        acc *= (l_prev * l_rcp)[:, None]
        # update acc
        p = p.to(tl.float16)
        v = tl.load(v_tile_ptr, boundary_check=(0, 1))
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr
        # update pointers
        k_tile_ptr = tl.advance(k_tile_ptr, [BLOCK_N, 0])
        v_tile_ptr = tl.advance(v_tile_ptr, [BLOCK_N, 0])
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_prev)
    tl.store(m_ptrs, m_prev)

    acc = acc.to(tl.float16)
    tl.store(out_tile_ptr, acc, boundary_check=(0, 1))

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



empty = torch.empty(128, device="cuda")


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        BLOCK = 128
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        num_warps = 4 if Lk <= 64 else 8
        D0 = q.shape[0] * q.shape[1] * q.shape[2]
        flash_fwd_kernel[grid](
            q, k, v, sm_scale,  #
            L, m,  #
            o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1], q.shape[2], D0,  #
            # BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_DMODEL=Lk,  #
            # num_warps=num_warps, num_stages=2
            )

        ctx.save_for_backward(q, k, v, o, L, m)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        return o


attention = _attention.apply


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [
    (4, 48, 128, 64),
    (4, 48, 256, 64),
    (4, 48, 512, 64),
    (4, 48, 1024, 64),
    (4, 48, 2048, 64),
    (4, 48, 4096, 64),
    #  (4, 48, 8192, 64), out of memory
])
@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason="requires arch 9+")
def test_op(Z, H, N_CTX, D_HEAD, request, dtype=torch.float16):
    
    set_seed()

    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2).requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2).requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2).requires_grad_()
    sm_scale = 0.2
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    for z in range(Z):
        for h in range(H):
            p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)

    # triton implementation
    tri_out = attention(q, k, v, sm_scale)


    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = tri_out.clone().detach().cpu()
    ################################################################### 

    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)

def calculate_flash_attention_fwd_tflops(params: dict, ms: float) -> float:
    Z, H, N_CTX, D_HEAD = params['Z'], params['H'], params['N_CTX'], params['D_HEAD']
    flops = 2 * Z * H * N_CTX * N_CTX * D_HEAD 
    tflops = flops / (ms / 1000) / 1e12
    return tflops

def calculate_flash_attention_fwd_gbps(params: dict, ms: float) -> float:
    Z, H, N_CTX, D_HEAD = params['Z'], params['H'], params['N_CTX'], params['D_HEAD']
    dtype_str = params.get('dtype_str', 'fp16') 
    current_dtype = torch.float16
    if dtype_str == 'fp32': current_dtype = torch.float32
    # bf16 was removed from perf test, but keep for completeness if added back
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16 
    element_size = torch.tensor([], dtype=current_dtype).element_size()
    bytes_q, bytes_k, bytes_v, bytes_o_write = [Z * H * N_CTX * D_HEAD * element_size] * 4
    bytes_L_write, bytes_M_write = [Z * H * N_CTX * 4] * 2
    total_bytes = bytes_q + bytes_k + bytes_v + bytes_o_write + bytes_L_write + bytes_M_write
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

OP_NAME_FOR_BENCHMARK = "flash_attention_fwd_triton_perf"

# --- NEW Performance Test Function using original test_op's parametrization ---
# It mirrors test_op's parametrize for Z, H, N_CTX, D_HEAD.
# It adds its own parametrize for dtype_str, EXCLUDING bf16 for now.
@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [
    (4, 48, 128, 64), (4, 48, 256, 64), (4, 48, 512, 64),
    (4, 48, 1024, 64), (4, 48, 2048, 64), (4, 48, 4096, 64),
    # (2, 12, 1024, 64), (1, 8, 2048, 32) # Example additional shapes if desired
])
@pytest.mark.parametrize('dtype_str', ['fp16']) # MODIFIED: Only fp16 for now to avoid bf16 JIT errors
# @pytest.mark.parametrize('dtype_str', ['fp16', 'fp32']) # Can add fp32 if it works
def test_performance(Z, H, N_CTX, D_HEAD, dtype_str, request): 
    
    cap = torch.cuda.get_device_capability()
    if cap[0] < 9: 
         pytest.skip("Original test requires arch 90+ for this flash attention kernel version.")
    # No bf16 specific skip needed as bf16 is removed from parametrize for this function

    # Shared memory OOM skip for D_HEAD=128 (if such cases were added)
    # This logic relies on knowing the hardcoded BLOCK and num_stages in _attention.forward
    hardcoded_block_in_wrapper = 128 
    hardcoded_num_stages_in_wrapper = 2
    if D_HEAD == 128 and \
       hardcoded_block_in_wrapper == 128 and \
       hardcoded_num_stages_in_wrapper == 2:
        pytest.skip(f"Skipping D_HEAD={D_HEAD} with BLOCK=128, num_stages=2 due to high shared memory demand.")
    
    set_seed()
    
    if dtype_str == 'fp32': current_dtype = torch.float32
    # elif dtype_str == 'bf16': current_dtype = torch.bfloat16 # bf16 removed
    else: current_dtype = torch.float16 

    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=current_dtype, device="cuda").normal_(mean=0.1, std=0.2).requires_grad_(False)
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=current_dtype, device="cuda").normal_(mean=0.4, std=0.2).requires_grad_(False)
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=current_dtype, device="cuda").normal_(mean=0.3, std=0.2).requires_grad_(False)
    sm_scale = 0.2 

    op_lambda = lambda: attention(q, k, v, sm_scale)

    bench_config = do_bench_config(warm_up=10, repetition=50) 
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)
    current_params_for_logs_and_calc = {
        "Z": Z, "H": H, "N_CTX": N_CTX, "D_HEAD": D_HEAD,
        "dtype_str": dtype_str, "sm_scale": sm_scale
    }
    
    perf_result = benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
                                            gbps_calculator=calculate_flash_attention_fwd_gbps,
                                            tflops_calculator=calculate_flash_attention_fwd_tflops)


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

try:
    from flash_attn.flash_attn_interface import flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64
# vary seq length for fixed head and batch=4
configs = [
    triton.testing.Benchmark(
        x_names=['N_CTX'],
        x_vals=[2**i for i in range(10, 14)],
        line_arg='provider',
        line_vals=['triton'] + (['flash'] if HAS_FLASH else []),
        line_names=['Triton'] + (['Flash'] if HAS_FLASH else []),
        styles=[('red', '-'), ('blue', '-')],
        ylabel='ms',
        plot_name=f'fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}',
        args={
            'H': N_HEADS,
            'BATCH': BATCH,
            'D_HEAD': D_HEAD,
            'dtype': torch.float16,
            'mode': mode,
        },
    ) for mode in ['fwd', 'bwd']
]


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, mode, provider, dtype=torch.float16, device="cuda"):
    assert mode in ['fwd', 'bwd']
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, sm_scale)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
        return ms
    if provider == "flash":
        lengths = torch.full((BATCH, ), fill_value=N_CTX, device=device)
        cu_seqlens = torch.zeros((BATCH + 1, ), device=device, dtype=torch.int32)
        cu_seqlens[1:] = lengths.cumsum(0)
        qkv = torch.randn((BATCH * N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, cu_seqlens, 0., N_CTX, causal=True)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
        return ms


