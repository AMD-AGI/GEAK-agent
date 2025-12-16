# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/algorithmicsuperintelligence/openevolve  - Apache License 2.0

import os 
import sys
import math
from typing import Optional, Tuple, Any
import time
import argparse

import torch
import triton
import triton.language as tl
import triton.testing
## Write triton autotuning code here



#************************** BASELINE FUNCTION **************************
@triton.jit
def _baseline_fwd_kernel(
    Q, K, V, sm_scale,
    L,
    O,
    stride_q_bs, stride_q_head, stride_q_seqlen, stride_q_dim,
    stride_k_bs, stride_k_head, stride_k_seqlen, stride_k_dim,
    stride_v_bs, stride_v_head, stride_v_seqlen, stride_v_dim,
    stride_o_bs, stride_o_head, stride_o_seqlen, stride_o_dim,
    BS, HEAD, SEQLEN,
    BLOCK_M: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bs_head = tl.program_id(1)

    qkv_base_offset = off_bs_head * stride_q_head
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_q_seqlen, stride_q_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_base_offset,
        shape=(DIM, SEQLEN),
        strides=(stride_k_dim, stride_k_seqlen),
        offsets=(0, 0),
        block_shape=(DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_k_seqlen, stride_v_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_N, DIM),
        order=(1, 0),
    )
    off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    max = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    denom = tl.zeros([BLOCK_M], dtype=tl.float32)
    out_buffer = tl.zeros([BLOCK_M, DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(tl.float16)
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else SEQLEN
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(off_m[:, None] >= (start_n + off_n[None, :]), qk, float("-inf"))
        qk += tl.dot(q, k)

        max_new = tl.maximum(max, tl.max(qk, 1))
        alpha = tl.math.exp2(max - max_new)
        nume = tl.math.exp2(qk - max_new[:, None])
        out_scale = denom * 0 + alpha
        out_buffer *= out_scale[:, None]
        out_buffer += tl.dot(nume.to(tl.float16), v)
        denom = denom * alpha + tl.sum(nume, 1)
        max = max_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    out_buffer = out_buffer / denom[:, None]
    l_ptr = L + off_bs_head * SEQLEN + off_m
    tl.store(l_ptr, max + tl.math.log2(denom))
    O_block_ptr = tl.make_block_ptr(
        base=O + qkv_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_o_seqlen, stride_o_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, DIM),
        order=(1, 0),
    )
    tl.store(O_block_ptr, out_buffer.to(tl.float16))

def baseline_flash_attn_triton(q, k, v, causal=True, sm_scale=1):
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    o = torch.empty_like(q)
    
    BLOCK_M = 128
    BLOCK_N = 64
    NUM_STAGES = 2
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    num_warps = 4 if Lk <= 64 else 8
    _baseline_fwd_kernel[grid](
        q, k, v, sm_scale,
        L,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, DIM=Lk,
        IS_CAUSAL=causal,
        num_warps=num_warps,
        num_stages=NUM_STAGES)
    
    return o
#************************** BASELINE FUNCTION END **************************

#************************** CANDIDATE TRITON KERNEL **************************
@triton.autotune(
    configs=[
        # EVOLVE-BLOCK-START
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'DIM': 16, 'IS_CAUSAL': True}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'DIM': 32, 'IS_CAUSAL': True}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'DIM': 64, 'IS_CAUSAL': True}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'DIM': 128, 'IS_CAUSAL': True}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'DIM': 16, 'IS_CAUSAL': False}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'DIM': 32, 'IS_CAUSAL': False}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'DIM': 64, 'IS_CAUSAL': False}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'DIM': 128, 'IS_CAUSAL': False}, num_stages=2, num_warps=8),
        # EVOLVE-BLOCK-END
    ],
    key=['BLOCK_M', 'BLOCK_N', 'DIM', 'IS_CAUSAL'],
    key_type=[tl.int32, tl.int32, tl.int32, tl.bool],
    num_stages=2,
    num_warps=4,
    pre_hook=lambda args: triton.testing.print_once(
        f"Autotuning baseline kernel with BLOCK_M={args['BLOCK_M']}, BLOCK_N={args['BLOCK_N']}, DIM={args['DIM']}, IS_CAUSAL={args['IS_CAUSAL']}"
    ),
    post_hook=lambda args, config: triton.testing.print_once(
        f"Autotuned baseline kernel with BLOCK_M={config['BLOCK_M']}, BLOCK_N={config['BLOCK_N']}, DIM={config['DIM']}, IS_CAUSAL={config['IS_CAUSAL']}, num_warps={config['num_warps'] if 'num_warps' in config else 'default'}"
    ),
) 
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    L,
    O,
    stride_q_bs, stride_q_head, stride_q_seqlen, stride_q_dim,
    stride_k_bs, stride_k_head, stride_k_seqlen, stride_k_dim,
    stride_v_bs, stride_v_head, stride_v_seqlen, stride_v_dim,
    stride_o_bs, stride_o_head, stride_o_seqlen, stride_o_dim,
    BS, HEAD, SEQLEN,
    BLOCK_M: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # **********************************************EVOLVE-BLOCK-START
    start_m = tl.program_id(0)
    off_bs_head = tl.program_id(1)

    qkv_base_offset = off_bs_head * stride_q_head
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_q_seqlen, stride_q_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_base_offset,
        shape=(DIM, SEQLEN),
        strides=(stride_k_dim, stride_k_seqlen),
        offsets=(0, 0),
        block_shape=(DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_k_seqlen, stride_v_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_N, DIM),
        order=(1, 0),
    )
    off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    max = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    denom = tl.zeros([BLOCK_M], dtype=tl.float32)
    out_buffer = tl.zeros([BLOCK_M, DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(tl.float16)
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else SEQLEN
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(off_m[:, None] >= (start_n + off_n[None, :]), qk, float("-inf"))
        qk += tl.dot(q, k)

        max_new = tl.maximum(max, tl.max(qk, 1))
        alpha = tl.math.exp2(max - max_new)
        nume = tl.math.exp2(qk - max_new[:, None])
        out_scale = denom * 0 + alpha
        out_buffer *= out_scale[:, None]
        out_buffer += tl.dot(nume.to(tl.float16), v)
        denom = denom * alpha + tl.sum(nume, 1)
        max = max_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    out_buffer = out_buffer / denom[:, None]
    l_ptr = L + off_bs_head * SEQLEN + off_m
    tl.store(l_ptr, max + tl.math.log2(denom))
    O_block_ptr = tl.make_block_ptr(
        base=O + qkv_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_o_seqlen, stride_o_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, DIM),
        order=(1, 0),
    )
    tl.store(O_block_ptr, out_buffer.to(tl.float16))
    # **********************************************EVOLVE-BLOCK-END

def flash_attn_triton(q, k, v, causal=True, sm_scale=1):
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    o = torch.empty_like(q)
    # **********************************************EVOLVE-BLOCK-START
    BLOCK_M = 128
    BLOCK_N = 64
    NUM_STAGES = 2
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    num_warps = 4 if Lk <= 64 else 8
    _fwd_kernel[grid](
        q, k, v, sm_scale,
        L,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, DIM=Lk,
        IS_CAUSAL=causal,
        num_warps=num_warps,
        num_stages=NUM_STAGES)
    # **********************************************EVOLVE-BLOCK-END
    return o
#************************** CANDIDATE TRITON KERNEL END **************************

##################################################################################################################################################
# Test cases for the flash_attn_triton function
def test_flash_attn_triton():
    batch_size = 2
    num_heads = 2
    seq_len = 128
    dim = 64

    # Create random input tensors
    q = torch.randn((batch_size, num_heads, seq_len, dim), dtype=torch.float16, device='cuda')
    k = torch.randn((batch_size, num_heads, seq_len, dim), dtype=torch.float16, device='cuda')
    v = torch.randn((batch_size, num_heads, seq_len, dim), dtype=torch.float16, device='cuda')

    # Test with causal=True
    output_causal = flash_attn_triton(q, k, v, causal=True, sm_scale=1.0)
    baseline_causal = baseline_flash_attn_triton(q, k, v, causal=True, sm_scale=1.0)

    # Test with causal=False
    output_non_causal = flash_attn_triton(q, k, v, causal=False, sm_scale=1.0)
    baseline_non_causal = baseline_flash_attn_triton(q, k, v, causal=False, sm_scale=1.0)

    result = baseline_causal is not None and baseline_non_causal is not None and (baseline_causal == output_causal).all() and (baseline_non_causal == output_non_causal).all()

    return result

def test_kernel_call():
    """
    Test the flash_attn_triton function with a simple call.
    This is a sanity check to ensure the kernel can be called without errors.
    """
    batch_size = 2
    num_heads = 2
    seq_len = 128
    dim = 64

    # Create random input tensors
    q = torch.randn((batch_size, num_heads, seq_len, dim), dtype=torch.float16, device='cuda')
    k = torch.randn((batch_size, num_heads, seq_len, dim), dtype=torch.float16, device='cuda')
    v = torch.randn((batch_size, num_heads, seq_len, dim), dtype=torch.float16, device='cuda')

    # Call the flash_attn_triton function
    try:
        output = flash_attn_triton(q, k, v, causal=True, sm_scale=1.0)
        return True
    except Exception as e:
        print(f"Error during kernel call: {e}", file=sys.stderr)
        return False

def run_benchmark(batch_size, num_head, seq_len, dim, sm_scale):
    """    Run the benchmark for the flash_attn_triton function with given parameters.
    Args:        batch_size (int): Batch size for the input tensors.
        num_head (int): Number of attention heads.
        seq_len (int): Sequence length of the input tensors.
        dim (int): Dimension of the input tensors.
        sm_scale (float): Scale factor for the softmax computation.
    """

    # Create random input tensors
    q = torch.randn((batch_size, num_head, seq_len, dim), dtype=torch.float16, device='cuda')
    k = torch.randn((batch_size, num_head, seq_len, dim), dtype=torch.float16, device='cuda')
    v = torch.randn((batch_size, num_head, seq_len, dim), dtype=torch.float16, device='cuda')

    # Benchmark the flash_attn_triton function
    ms, min_ms, max_ms =  triton.testing.do_bench(
        lambda: flash_attn_triton(q, k, v, causal=True, sm_scale=sm_scale),
        warmup=500,
        rep=1000,
        quantiles=[0.5, 0.8, 0.2],
        return_mode="median"
    )

    # Benchmark the flash_attn_triton function
    baseline_ms, baseline_min_ms, baseline_max_ms =  triton.testing.do_bench(
        lambda: baseline_flash_attn_triton(q, k, v, causal=True, sm_scale=sm_scale),
        warmup=500,
        rep=1000,
        quantiles=[0.5, 0.8, 0.2],
        return_mode="median"
    )

    return {
        "batch_size": batch_size,
        "num_head": num_head,
        "seq_len": seq_len,
        "dim": dim,
        "sm_scale": sm_scale,
        "median_time_ms": ms,
        "min_time_ms": min_ms,
        "max_time_ms": max_ms,
        "baseline_median_time_ms": baseline_ms,
        "baseline_min_time_ms": baseline_min_ms,
        "baseline_max_time_ms": baseline_max_ms,
        "speedup": baseline_ms / ms if ms > 0 else None,
    }

def benchmark_kernel():
    """
    Benchmark triton kernel using Triton's benchmarking utilities.
    """
    # Define the benchmark parameters
    batch_sizes = [2, 4, 8]
    num_heads = [40]
    seq_lens = [128, 512, 1024]
    dims = [128]
    sm_scale = 1.0

    all_args = []
    for batch_size in batch_sizes:
        for num_head in num_heads:
            for seq_len in seq_lens:
                for dim in dims:
                    all_args.append((batch_size, num_head, seq_len, dim, sm_scale))

    benchmark_results = []

    for batch_size, num_head, seq_len, dim, sm_scale in all_args:
                    print(f"Benchmarking with batch_size={batch_size}, num_head={num_head}, seq_len={seq_len}, dim={dim}")
                    # Run the benchmark
                    run_result = run_benchmark(batch_size, num_head, seq_len, dim, sm_scale)
                    benchmark_results.append(run_result)

    ## compute average speedup across all runs
    total_speedup = 0
    count = 0
    for result in benchmark_results:
        if result["speedup"] is not None:
            total_speedup += result["speedup"]
            count += 1

    average_speedup = total_speedup / count if count > 0 else None
    return average_speedup    

def test_kernel_correctness():
    return test_flash_attn_triton()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark and test Triton Flash Attention kernel.")
    parser.add_argument("--call", '-c', action="store_true", help="Run correctness test for the Triton Flash Attention kernel.")
    parser.add_argument("--test", '-t', action="store_true", help="Run correctness test for the Triton Flash Attention kernel.")
    parser.add_argument("--benchmark", '-b', action="store_true", help="Run performance benchmark for the Triton Flash Attention kernel.")
    parser.add_argument("--all", '-a', action="store_true", help="Run all tests and benchmarks.")
    args = parser.parse_args()

    call, test, benchmark = None, None, None
    err_msg = " "
    if args.call or args.all:
        try:
            call = test_kernel_call()
        except Exception as e:
            print(f"Error during kernel call: {e}", file=sys.stderr)
            err_msg = str(e)
            call = False

    if (args.test or args.all) and call:
        try:
            test = test_kernel_correctness()
        except Exception as e:
            print(f"Error during kernel correctness test: {e}", file=sys.stderr)
            err_msg = str(e)
            test = False

    if (args.benchmark or args.all) and call and test:
        try:
            benchmark = benchmark_kernel()
        except Exception as e:
            print(f"Error during kernel benchmark: {e}", file=sys.stderr)
            err_msg = str(e)
            benchmark = None

    print(f"{call}#*#*{test}#*#*{benchmark}#*#*{err_msg}")
