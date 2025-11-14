# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/thunlp/TritonBench - Apache License 2.0
import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_destindex_copy_kv(
    KV_nope,
    KV_rope,
    Dest_loc,
    O_nope,
    O_rope,
    stride_kv_nope_bs,
    stride_kv_nope_h,
    stride_kv_nope_d,
    stride_kv_rope_bs,
    stride_kv_rope_h,
    stride_kv_rope_d,
    stride_o_nope_bs,
    stride_o_nope_h,
    stride_o_nope_d,
    stride_o_rope_bs,
    stride_o_rope_h,
    stride_o_rope_d,
    kv_nope_head_num,
    kv_rope_head_num,
    BLOCK_DMODEL_NOPE: tl.constexpr,
    BLOCK_DMODEL_ROPE: tl.constexpr,
):
    cur_index = tl.program_id(0)
    offs_d_nope = tl.arange(0, BLOCK_DMODEL_NOPE)
    offs_d_rope = tl.arange(0, BLOCK_DMODEL_ROPE)
    dest_index = tl.load(Dest_loc + cur_index)

    kv_nope_ptrs = KV_nope + cur_index * stride_kv_nope_bs + stride_kv_nope_d * offs_d_nope[None, :]
    kv_rope_ptrs = KV_rope + cur_index * stride_kv_rope_bs + stride_kv_rope_d * offs_d_rope[None, :]

    o_nope_ptrs = O_nope + dest_index * stride_o_nope_bs + stride_o_nope_d * offs_d_nope[None, :]
    o_rope_ptrs = O_rope + dest_index * stride_o_rope_bs + stride_o_rope_d * offs_d_rope[None, :]

    kv_nope = tl.load(kv_nope_ptrs)
    kv_rope = tl.load(kv_rope_ptrs)

    tl.store(o_nope_ptrs, kv_nope)
    tl.store(o_rope_ptrs, kv_rope)
    return


@torch.no_grad()
def destindex_copy_kv(KV_nope, KV_rope, DestLoc, O_nope, O_rope):
    seq_len = DestLoc.shape[0]
    kv_nope_head_num = KV_nope.shape[1]
    kv_rope_head_num = KV_rope.shape[1]

    kv_nope_head_dim = KV_nope.shape[2]
    kv_rope_head_dim = KV_rope.shape[2]

    aligned_d_nope = triton.next_power_of_2(kv_nope_head_dim) # 调整为2的幂次方
    aligned_d_rope = triton.next_power_of_2(kv_rope_head_dim) # 调整为2的幂次方

    assert KV_nope.shape[1] == O_nope.shape[1]
    assert KV_nope.shape[2] == O_nope.shape[2]
    assert KV_rope.shape[1] == O_rope.shape[1]
    assert KV_rope.shape[2] == O_rope.shape[2]
    grid = (seq_len,)
    num_warps = 2

    _fwd_kernel_destindex_copy_kv[grid](
        KV_nope,
        KV_rope,
        DestLoc,
        O_nope,
        O_rope,
        KV_nope.stride(0),
        KV_nope.stride(1),
        KV_nope.stride(2),
        KV_rope.stride(0),
        KV_rope.stride(1),
        KV_rope.stride(2),
        O_nope.stride(0),
        O_nope.stride(1),
        O_nope.stride(2),
        O_rope.stride(0),
        O_rope.stride(1),
        O_rope.stride(2),
        kv_nope_head_num,
        kv_rope_head_num,
        # BLOCK_DMODEL_NOPE=kv_nope_head_dim,
        # BLOCK_DMODEL_ROPE=kv_rope_head_dim,
        BLOCK_DMODEL_NOPE=aligned_d_nope,  # 传递对齐后的值
        BLOCK_DMODEL_ROPE=aligned_d_rope,  # 传递对齐后的值
        num_warps=num_warps,
        num_stages=1,
    )
    return




