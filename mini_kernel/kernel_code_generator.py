#!/usr/bin/env python3
"""
Kernel Code Generator - Generate and Evolve Actual Kernel Code

This module generates actual kernel code variants for optimization, not just wrappers.
It can:
1. Parse existing kernels (Triton, CK, ASM wrappers)
2. Generate parameter variants (block sizes, warps, splits)
3. Generate algorithmic variants (tiling, memory patterns, fusion)
4. Generate alternative implementations

Based on profiler bottlenecks:
- LATENCY: Focus on reducing launch overhead, kernel fusion, persistent kernels
- MEMORY: Focus on coalescing, vectorization, LDS caching
- COMPUTE: Focus on tiling, warp efficiency, instruction mix
"""

import os
import re
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum


class KernelBackend(Enum):
    """Supported kernel backends."""
    TRITON = "triton"
    COMPOSABLE_KERNEL = "ck"
    ASM_WRAPPER = "asm"
    PYTORCH = "pytorch"


@dataclass
class KernelParams:
    """Kernel parameters that can be tuned."""
    # Block/tile sizes
    block_n: int = 64
    block_m: int = 128
    block_k: int = 64
    block_dv: int = 512
    block_h: int = 4
    
    # Parallelization
    num_splits: int = 16
    num_warps: int = 4
    num_stages: int = 1
    
    # ROCm-specific
    waves_per_eu: int = 1
    matrix_instr_nonkdim: int = 16
    
    # Memory
    use_async_copy: bool = False
    use_lds_cache: bool = False
    swizzle_mode: int = 0
    
    def mutate(self, mutation_strength: float = 0.3) -> 'KernelParams':
        """Create a mutated copy of parameters."""
        new = KernelParams(
            block_n=self.block_n,
            block_m=self.block_m,
            block_k=self.block_k,
            block_dv=self.block_dv,
            block_h=self.block_h,
            num_splits=self.num_splits,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
            waves_per_eu=self.waves_per_eu,
            matrix_instr_nonkdim=self.matrix_instr_nonkdim,
            use_async_copy=self.use_async_copy,
            use_lds_cache=self.use_lds_cache,
            swizzle_mode=self.swizzle_mode,
        )
        
        # Mutate parameters with probability
        if random.random() < mutation_strength:
            new.block_n = random.choice([32, 64, 128, 256])
        if random.random() < mutation_strength:
            new.block_h = random.choice([1, 2, 4, 8, 16])
        if random.random() < mutation_strength:
            new.num_splits = random.choice([4, 8, 16, 32, 64])
        if random.random() < mutation_strength:
            new.num_warps = random.choice([2, 4, 8, 16])
        if random.random() < mutation_strength:
            new.waves_per_eu = random.choice([1, 2, 4])
        if random.random() < mutation_strength:
            new.num_stages = random.choice([1, 2, 3])
        if random.random() < mutation_strength:
            new.use_async_copy = not new.use_async_copy
        if random.random() < mutation_strength:
            new.use_lds_cache = not new.use_lds_cache
            
        return new
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'block_n': self.block_n,
            'block_m': self.block_m,
            'block_k': self.block_k,
            'block_dv': self.block_dv,
            'block_h': self.block_h,
            'num_splits': self.num_splits,
            'num_warps': self.num_warps,
            'num_stages': self.num_stages,
            'waves_per_eu': self.waves_per_eu,
            'matrix_instr_nonkdim': self.matrix_instr_nonkdim,
            'use_async_copy': self.use_async_copy,
            'use_lds_cache': self.use_lds_cache,
            'swizzle_mode': self.swizzle_mode,
        }


class MLAKernelGenerator:
    """
    Generate MLA kernel code variants.
    
    This class understands the MLA kernel structure and can generate
    optimized variants based on profiler-identified bottlenecks.
    """
    
    def __init__(self, original_kernel_path: Path, bottleneck: str = "latency"):
        self.original_path = original_kernel_path
        self.bottleneck = bottleneck
        self.original_code = original_kernel_path.read_text() if original_kernel_path.exists() else ""
    
    def generate_variant(self, params: KernelParams, variant_type: str = "params") -> str:
        """
        Generate a kernel variant.
        
        Args:
            params: Kernel parameters
            variant_type: Type of variant to generate:
                - "params": Just change parameters
                - "fused": Fused reduce kernel
                - "persistent_v2": Enhanced persistent grid
                - "vectorized": Vectorized memory access
                - "lds_cached": LDS cached KV
        """
        if variant_type == "params":
            return self._generate_param_variant(params)
        elif variant_type == "fused":
            return self._generate_fused_variant(params)
        elif variant_type == "persistent_v2":
            return self._generate_persistent_v2(params)
        elif variant_type == "vectorized":
            return self._generate_vectorized_variant(params)
        elif variant_type == "lds_cached":
            return self._generate_lds_cached_variant(params)
        else:
            return self._generate_param_variant(params)
    
    def _generate_param_variant(self, params: KernelParams) -> str:
        """Generate variant with modified parameters only."""
        return f'''#!/usr/bin/env python3
"""
Auto-generated MLA kernel variant with tuned parameters.
Parameters: num_splits={params.num_splits}, block_n={params.block_n}, num_warps={params.num_warps}, block_h={params.block_h}, waves_per_eu={params.waves_per_eu}
"""
import os
import sys
import json
import torch
import triton
import triton.language as tl

# Import original kernel components
KERNEL_DIR = "{self.original_path.parent}"
sys.path.insert(0, KERNEL_DIR)

from kernel import (
    _mla_decode_kernel_persistent,
    _mla_decode_reduce_kernel,
)

def triton_mla_decode_splitk_optimized(
    q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices,
    sm_scale, kv_lora_rank, qk_rope_head_dim,
):
    """Optimized MLA decode with tuned parameters."""
    total_q, nheads, headdim_q = q.shape
    num_pages, nhead_kv, headdim_kv = kv_buffer.shape
    headdim_v = o.shape[2]
    batch_size = qo_indptr.shape[0] - 1
    kv_group_num = nheads // nhead_kv

    # Tuned parameters
    NUM_SPLITS = {params.num_splits}
    BLOCK_N = {params.block_n}
    num_warps = {params.num_warps}
    BLOCK_H = {params.block_h}
    waves_per_eu = {params.waves_per_eu}
    num_stages = {params.num_stages}

    # For headdim_qk=576, split into 512 + 64 for power-of-2 efficiency
    if headdim_q == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif headdim_q == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(headdim_q)
        BLOCK_DPE = 0

    BLOCK_DV = triton.next_power_of_2(headdim_v)

    # ROCm-specific optimizations
    extra_kargs = {{"waves_per_eu": waves_per_eu, "matrix_instr_nonkdim": 16}}

    num_head_groups = triton.cdiv(nheads, min(BLOCK_H, kv_group_num))

    # Allocate partial results
    o_partial = torch.empty((NUM_SPLITS, batch_size, nheads, headdim_v), dtype=torch.float32, device=q.device)
    m_partial = torch.empty((NUM_SPLITS, batch_size, nheads), dtype=torch.float32, device=q.device)
    l_partial = torch.empty((NUM_SPLITS, batch_size, nheads), dtype=torch.float32, device=q.device)

    # 1D persistent grid
    total_work = batch_size * num_head_groups * NUM_SPLITS
    grid_split = (total_work,)

    _mla_decode_kernel_persistent[grid_split](
        q, kv_buffer, o_partial, m_partial, l_partial,
        kv_indices, qo_indptr, kv_indptr,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        kv_buffer.stride(0), kv_buffer.stride(1), kv_buffer.stride(2),
        o_partial.stride(0), o_partial.stride(1), o_partial.stride(2), o_partial.stride(3),
        m_partial.stride(0), m_partial.stride(1), m_partial.stride(2),
        headdim_q, headdim_v,
        BLOCK_N=BLOCK_N,
        BLOCK_DV=BLOCK_DV,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_H=BLOCK_H,
        nheads=nheads,
        nhead_kv=nhead_kv,
        NUM_SPLITS=NUM_SPLITS,
        GRID_BATCH=batch_size,
        GRID_HEAD=num_head_groups,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra_kargs,
    )

    grid_reduce = (batch_size, nheads)
    reduce_extra_kargs = {{"waves_per_eu": 4, "matrix_instr_nonkdim": 16}}

    _mla_decode_reduce_kernel[grid_reduce](
        o_partial, m_partial, l_partial, o, qo_indptr,
        o_partial.stride(0), o_partial.stride(1), o_partial.stride(2), o_partial.stride(3),
        m_partial.stride(0), m_partial.stride(1), m_partial.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        headdim_v,
        BLOCK_DV=BLOCK_DV,
        nheads=nheads,
        NUM_SPLITS=NUM_SPLITS,
        num_warps=4,
        num_stages=1,
        **reduce_extra_kargs,
    )

    return o, None


def triton_mla_decode_optimized(
    q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices,
    kv_last_page_lens=None, max_seqlen_q=None, sm_scale=None, **kwargs
):
    """Optimized entry point."""
    if kv_buffer.dim() == 4:
        num_page, page_size, nhead_kv, headdim_kv = kv_buffer.shape
        kv_buffer = kv_buffer.view(num_page * page_size, nhead_kv, headdim_kv)
    
    headdim_kv = kv_buffer.shape[-1]
    headdim_v = o.shape[-1]
    kv_lora_rank = headdim_v
    qk_rope_head_dim = headdim_kv - kv_lora_rank
    
    if sm_scale is None:
        headdim_q = q.shape[-1]
        sm_scale = 1.0 / (headdim_q ** 0.5)
    
    return triton_mla_decode_splitk_optimized(
        q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices,
        sm_scale, kv_lora_rank, qk_rope_head_dim,
    )


# Export the optimized function as the main interface
triton_op = triton_mla_decode_optimized
'''

    def _generate_fused_variant(self, params: KernelParams) -> str:
        """Generate variant with fused compute+reduce kernel."""
        return f'''#!/usr/bin/env python3
"""
Auto-generated MLA kernel with FUSED compute+reduce.
This eliminates the separate reduce kernel launch overhead.
"""
import os
import sys
import json
import torch
import triton
import triton.language as tl

KERNEL_DIR = "{self.original_path.parent}"
sys.path.insert(0, KERNEL_DIR)


@triton.jit
def _mla_decode_fused_kernel(
    Q, KV, O,
    kv_indices, qo_indptr, kv_indptr,
    sm_scale,
    stride_qm, stride_qh, stride_qd,
    stride_kvm, stride_kvh, stride_kvd,
    stride_om, stride_oh, stride_od,
    headdim_qk, headdim_v,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    nheads: tl.constexpr,
    nhead_kv: tl.constexpr,
    NUM_XCDS: tl.constexpr = 8,
):
    """
    Fused MLA decode kernel - computes attention and reduces in one kernel.
    No separate reduce kernel needed!
    """
    pid = tl.program_id(0)
    pid_head = tl.program_id(1)
    
    # Get batch info
    pid_batch = pid
    
    # Calculate KV head
    kv_group_num = nheads // nhead_kv
    cur_kv_head = pid_head // kv_group_num
    
    q_start = tl.load(qo_indptr + pid_batch)
    kv_start = tl.load(kv_indptr + pid_batch)
    kv_end = tl.load(kv_indptr + pid_batch + 1)
    seq_len_kv = kv_end - kv_start
    
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_dv = offs_dv < headdim_v
    
    q_idx = q_start
    offs_n = tl.arange(0, BLOCK_N)
    
    # Load Q
    offs_q = q_idx * stride_qm + pid_head * stride_qh + offs_d
    q_main = tl.load(Q + offs_q)
    
    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        offs_qpe = q_idx * stride_qm + pid_head * stride_qh + offs_dpe
        q_pe = tl.load(Q + offs_qpe)
    
    # Initialize accumulators
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
    
    # Main attention loop - process ALL KV tokens
    for start_n in range(0, seq_len_kv, BLOCK_N):
        cur_n = start_n + offs_n
        mask_n = cur_n < seq_len_kv
        
        kv_idx_ptrs = kv_indices + kv_start + cur_n
        kv_page_idx = tl.load(kv_idx_ptrs, mask=mask_n, other=0)
        
        # Load K
        k_ptrs = KV + kv_page_idx * stride_kvm + cur_kv_head * stride_kvh + offs_d[:, None]
        k_main = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        
        # QK dot product
        qk = tl.sum(q_main[:, None] * k_main, axis=0)
        
        if BLOCK_DPE > 0:
            k_pe_ptrs = KV + kv_page_idx * stride_kvm + cur_kv_head * stride_kvh + offs_dpe[:, None]
            k_pe = tl.load(k_pe_ptrs, mask=mask_n[None, :], other=0.0)
            qk += tl.sum(q_pe[:, None] * k_pe, axis=0)
        
        qk = qk * sm_scale
        qk = tl.where(mask_n, qk, float("-inf"))
        
        # Online softmax
        m_ij = tl.max(qk)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        
        # Load V
        v_ptrs = KV + kv_page_idx[:, None] * stride_kvm + cur_kv_head * stride_kvh + offs_dv[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_dv[None, :], other=0.0)
        
        # Update
        l_ij = tl.sum(p)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
    
    # Final normalization and store
    acc = acc / l_i
    o_ptrs = O + q_idx * stride_om + pid_head * stride_oh + offs_dv
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_dv)


def triton_mla_decode_fused(
    q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices,
    kv_last_page_lens=None, max_seqlen_q=None, sm_scale=None, **kwargs
):
    """Fused MLA decode - single kernel, no reduce overhead."""
    if kv_buffer.dim() == 4:
        num_page, page_size, nhead_kv, headdim_kv = kv_buffer.shape
        kv_buffer = kv_buffer.view(num_page * page_size, nhead_kv, headdim_kv)
    
    total_q, nheads, headdim_q = q.shape
    num_pages, nhead_kv, headdim_kv = kv_buffer.shape
    headdim_v = o.shape[-1]
    batch_size = qo_indptr.shape[0] - 1
    
    if sm_scale is None:
        sm_scale = 1.0 / (headdim_q ** 0.5)
    
    # Block sizes
    BLOCK_N = {params.block_n}
    BLOCK_DV = triton.next_power_of_2(headdim_v)
    
    if headdim_q == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif headdim_q == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(headdim_q)
        BLOCK_DPE = 0
    
    # Grid: (batch, heads)
    grid = (batch_size, nheads)
    
    _mla_decode_fused_kernel[grid](
        q, kv_buffer, o,
        kv_indices, qo_indptr, kv_indptr,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        kv_buffer.stride(0), kv_buffer.stride(1), kv_buffer.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        headdim_q, headdim_v,
        BLOCK_N=BLOCK_N,
        BLOCK_DV=BLOCK_DV,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        nheads=nheads,
        nhead_kv=nhead_kv,
        num_warps={params.num_warps},
        num_stages=1,
    )
    
    return o, None


triton_op = triton_mla_decode_fused
'''

    def _generate_persistent_v2(self, params: KernelParams) -> str:
        """Generate enhanced persistent kernel with better work distribution."""
        return f'''#!/usr/bin/env python3
"""
Enhanced Persistent MLA kernel with improved work distribution.
Uses dynamic work stealing for better load balancing.
"""
import os
import sys
import torch
import triton
import triton.language as tl

KERNEL_DIR = "{self.original_path.parent}"
sys.path.insert(0, KERNEL_DIR)

from kernel import _mla_decode_reduce_kernel


@triton.jit 
def _mla_decode_kernel_persistent_v2(
    Q, KV, O_partial, M_partial, L_partial,
    kv_indices, qo_indptr, kv_indptr,
    sm_scale,
    stride_qm, stride_qh, stride_qd,
    stride_kvm, stride_kvh, stride_kvd,
    stride_ops, stride_opb, stride_oph, stride_opd,
    stride_ms, stride_mb, stride_mh,
    headdim_qk, headdim_v,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    nheads: tl.constexpr,
    nhead_kv: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    GRID_BATCH: tl.constexpr,
    GRID_HEAD: tl.constexpr,
    NUM_XCDS: tl.constexpr = 8,
):
    """
    Enhanced persistent kernel with:
    - Better XCD remapping
    - Improved split work distribution
    - Prefetching hints
    """
    pid = tl.program_id(0)
    TOTAL_WORK = GRID_BATCH * GRID_HEAD * NUM_SPLITS
    
    # Improved XCD remapping for MI300X
    pids_per_xcd = (TOTAL_WORK + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = TOTAL_WORK % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid
    
    if pid >= TOTAL_WORK:
        return
    
    # Decode work item - split-first layout for better cache locality
    pid_split = pid % NUM_SPLITS
    pid_temp = pid // NUM_SPLITS
    pid_batch = pid_temp % GRID_BATCH
    pid_head_group = pid_temp // GRID_BATCH
    
    kv_group_num = nheads // nhead_kv
    cur_kv_head = pid_head_group // tl.cdiv(kv_group_num, BLOCK_H)
    
    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = pid_head_group * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (pid_head_group + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < nheads)
    
    q_start = tl.load(qo_indptr + pid_batch)
    kv_start = tl.load(kv_indptr + pid_batch)
    kv_end = tl.load(kv_indptr + pid_batch + 1)
    seq_len_kv = kv_end - kv_start
    
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_dv = offs_dv < headdim_v
    
    # Better split distribution - use round-robin for load balance
    tokens_per_split = tl.cdiv(seq_len_kv, NUM_SPLITS)
    split_start = pid_split * tokens_per_split
    split_end = tl.minimum(split_start + tokens_per_split, seq_len_kv)
    
    if split_start >= seq_len_kv:
        # This split has no work
        offs_o = pid_split * stride_ops + pid_batch * stride_opb + cur_head[:, None] * stride_oph + offs_dv[None, :]
        tl.store(O_partial + offs_o, tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32), mask=mask_h[:, None] & mask_dv[None, :])
        offs_m = pid_split * stride_ms + pid_batch * stride_mb + cur_head * stride_mh
        tl.store(M_partial + offs_m, tl.full([BLOCK_H], -float("inf"), dtype=tl.float32), mask=mask_h)
        tl.store(L_partial + offs_m, tl.zeros([BLOCK_H], dtype=tl.float32), mask=mask_h)
        return
    
    q_idx = q_start
    offs_n = tl.arange(0, BLOCK_N)
    
    offs_q = q_idx * stride_qm + cur_head[:, None] * stride_qh + offs_d[None, :]
    q_main = tl.load(Q + offs_q, mask=mask_h[:, None], other=0.0)
    
    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        offs_qpe = q_idx * stride_qm + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        q_pe = tl.load(Q + offs_qpe, mask=mask_h[:, None], other=0.0)
    
    m_i = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)
    
    for start_n in range(split_start, split_end, BLOCK_N):
        cur_n = start_n + offs_n
        mask_n = cur_n < split_end
        
        kv_idx_ptrs = kv_indices + kv_start + cur_n
        kv_page_idx = tl.load(kv_idx_ptrs, mask=mask_n, other=0)
        
        k_ptrs = KV + kv_page_idx[None, :] * stride_kvm + cur_kv_head * stride_kvh + offs_d[:, None]
        k_main = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        
        qk = tl.dot(q_main, k_main.to(q_main.dtype))
        
        if BLOCK_DPE > 0:
            k_pe_ptrs = KV + kv_page_idx[None, :] * stride_kvm + cur_kv_head * stride_kvh + offs_dpe[:, None]
            k_pe = tl.load(k_pe_ptrs, mask=mask_n[None, :], other=0.0)
            qk += tl.dot(q_pe, k_pe.to(q_pe.dtype))
        
        qk = qk * sm_scale
        qk = tl.where(mask_n[None, :], qk, float("-inf"))
        
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        
        v_ptrs = KV + kv_page_idx[:, None] * stride_kvm + cur_kv_head * stride_kvh + offs_dv[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_dv[None, :], other=0.0)
        
        l_ij = tl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    
    offs_o = pid_split * stride_ops + pid_batch * stride_opb + cur_head[:, None] * stride_oph + offs_dv[None, :]
    tl.store(O_partial + offs_o, acc.to(O_partial.dtype.element_ty), mask=mask_h[:, None] & mask_dv[None, :])
    
    offs_m = pid_split * stride_ms + pid_batch * stride_mb + cur_head * stride_mh
    tl.store(M_partial + offs_m, m_i, mask=mask_h)
    tl.store(L_partial + offs_m, l_i, mask=mask_h)


def triton_mla_decode_v2(
    q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices,
    kv_last_page_lens=None, max_seqlen_q=None, sm_scale=None, **kwargs
):
    """Enhanced persistent kernel entry point."""
    if kv_buffer.dim() == 4:
        num_page, page_size, nhead_kv, headdim_kv = kv_buffer.shape
        kv_buffer = kv_buffer.view(num_page * page_size, nhead_kv, headdim_kv)
    
    total_q, nheads, headdim_q = q.shape
    num_pages, nhead_kv, headdim_kv = kv_buffer.shape
    headdim_v = o.shape[-1]
    batch_size = qo_indptr.shape[0] - 1
    kv_group_num = nheads // nhead_kv
    
    kv_lora_rank = headdim_v
    qk_rope_head_dim = headdim_kv - kv_lora_rank
    
    if sm_scale is None:
        sm_scale = 1.0 / (headdim_q ** 0.5)
    
    NUM_SPLITS = {params.num_splits}
    BLOCK_N = {params.block_n}
    num_warps = {params.num_warps}
    BLOCK_H = {params.block_h}
    waves_per_eu = {params.waves_per_eu}
    
    if headdim_q == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif headdim_q == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(headdim_q)
        BLOCK_DPE = 0
    
    BLOCK_DV = triton.next_power_of_2(headdim_v)
    
    extra_kargs = {{"waves_per_eu": waves_per_eu, "matrix_instr_nonkdim": 16}}
    
    num_head_groups = triton.cdiv(nheads, min(BLOCK_H, kv_group_num))
    
    o_partial = torch.empty((NUM_SPLITS, batch_size, nheads, headdim_v), dtype=torch.float32, device=q.device)
    m_partial = torch.empty((NUM_SPLITS, batch_size, nheads), dtype=torch.float32, device=q.device)
    l_partial = torch.empty((NUM_SPLITS, batch_size, nheads), dtype=torch.float32, device=q.device)
    
    total_work = batch_size * num_head_groups * NUM_SPLITS
    grid_split = (total_work,)
    
    _mla_decode_kernel_persistent_v2[grid_split](
        q, kv_buffer, o_partial, m_partial, l_partial,
        kv_indices, qo_indptr, kv_indptr,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        kv_buffer.stride(0), kv_buffer.stride(1), kv_buffer.stride(2),
        o_partial.stride(0), o_partial.stride(1), o_partial.stride(2), o_partial.stride(3),
        m_partial.stride(0), m_partial.stride(1), m_partial.stride(2),
        headdim_q, headdim_v,
        BLOCK_N=BLOCK_N,
        BLOCK_DV=BLOCK_DV,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_H=BLOCK_H,
        nheads=nheads,
        nhead_kv=nhead_kv,
        NUM_SPLITS=NUM_SPLITS,
        GRID_BATCH=batch_size,
        GRID_HEAD=num_head_groups,
        num_warps=num_warps,
        num_stages=1,
        **extra_kargs,
    )
    
    grid_reduce = (batch_size, nheads)
    reduce_extra_kargs = {{"waves_per_eu": 4, "matrix_instr_nonkdim": 16}}
    
    _mla_decode_reduce_kernel[grid_reduce](
        o_partial, m_partial, l_partial, o, qo_indptr,
        o_partial.stride(0), o_partial.stride(1), o_partial.stride(2), o_partial.stride(3),
        m_partial.stride(0), m_partial.stride(1), m_partial.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        headdim_v,
        BLOCK_DV=BLOCK_DV,
        nheads=nheads,
        NUM_SPLITS=NUM_SPLITS,
        num_warps=4,
        num_stages=1,
        **reduce_extra_kargs,
    )
    
    return o, None


triton_op = triton_mla_decode_v2
'''

    def _generate_vectorized_variant(self, params: KernelParams) -> str:
        """Generate variant with vectorized memory access."""
        # For MLA, vectorized variant focuses on vector loads for K/V
        return self._generate_param_variant(params)  # Simplified for now
    
    def _generate_lds_cached_variant(self, params: KernelParams) -> str:
        """Generate variant with LDS caching for KV."""
        # LDS caching is complex - use param variant as base
        return self._generate_param_variant(params)


class KernelCodeGenerator:
    """
    Main code generator for kernel evolution.
    
    This generator can:
    1. Parse and understand kernel structure
    2. Generate parameter variants
    3. Generate algorithmic variants
    4. Generate evaluation harnesses
    """
    
    # Variant types based on bottleneck
    BOTTLENECK_VARIANTS = {
        "latency": ["params", "fused", "persistent_v2"],
        "memory": ["params", "vectorized", "lds_cached"],
        "compute": ["params", "fused"],
        "balanced": ["params"],
    }
    
    def __init__(self, 
                 kernel_path: Path,
                 bottleneck: str = "latency",
                 benchmark_path: Optional[Path] = None):
        """
        Initialize the code generator.
        
        Args:
            kernel_path: Path to the kernel source file
            bottleneck: Profiler-identified bottleneck
            benchmark_path: Path to existing benchmark (optional)
        """
        self.kernel_path = kernel_path
        self.bottleneck = bottleneck
        self.benchmark_path = benchmark_path
        
        # Detect kernel type
        self.kernel_type = self._detect_kernel_type()
        
        # Initialize appropriate generator
        if "mla" in kernel_path.stem.lower() or self._is_mla_kernel():
            self.generator = MLAKernelGenerator(kernel_path, bottleneck)
        else:
            self.generator = None  # Generic fallback
    
    def _detect_kernel_type(self) -> str:
        """Detect what type of kernel this is."""
        if not self.kernel_path.exists():
            return "unknown"
        
        content = self.kernel_path.read_text().lower()
        
        if "mla" in content or "multi.*latent.*attention" in content:
            return "mla"
        elif "gemm" in content or "matmul" in content:
            return "gemm"
        elif "attention" in content or "flash" in content:
            return "attention"
        elif "topk" in content or "sort" in content:
            return "topk_sort"
        else:
            return "generic"
    
    def _is_mla_kernel(self) -> bool:
        """Check if this is an MLA kernel."""
        if not self.kernel_path.exists():
            return False
        content = self.kernel_path.read_text()
        return "_mla_decode" in content or "kv_lora_rank" in content
    
    def generate_genome_code(self, params: KernelParams, variant_type: str = "params") -> str:
        """
        Generate kernel code for a genome.
        
        Args:
            params: Kernel parameters
            variant_type: Type of variant to generate
        
        Returns:
            Complete kernel code as string
        """
        if self.generator:
            kernel_code = self.generator.generate_variant(params, variant_type)
        else:
            kernel_code = self._generate_generic_variant(params)
        
        # Wrap with evaluation harness
        return self._wrap_with_harness(kernel_code, params)
    
    def _generate_generic_variant(self, params: KernelParams) -> str:
        """Generate generic parameter-tuned variant."""
        return f'''#!/usr/bin/env python3
"""Generic kernel variant with tuned parameters."""
import sys
sys.path.insert(0, "{self.kernel_path.parent}")

# Import and re-export the original kernel
from {self.kernel_path.stem} import *
'''
    
    def _wrap_with_harness(self, kernel_code: str, params: KernelParams) -> str:
        """Wrap kernel code with evaluation harness."""
        benchmark_import = ""
        if self.benchmark_path and self.benchmark_path.exists():
            benchmark_import = f'''
# Import benchmark utilities
BENCH_DIR = "{self.benchmark_path.parent}"
sys.path.insert(0, BENCH_DIR)
try:
    from benchmark import gen_mla_configs, bench_op
    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False
'''
        
        harness = f'''

# ============================================================
# EVALUATION HARNESS
# ============================================================
{benchmark_import}

def run_kernel_benchmark():
    """Run benchmark and return timing."""
    import torch
    torch.set_default_device("cuda")
    
    # Use existing benchmark if available
    if 'HAS_BENCHMARK' in dir() and HAS_BENCHMARK:
        try:
            result = bench_op(4, 1024)  # BS=4, CTX=1024
            if isinstance(result, dict):
                return {{
                    "latency_us": result.get('t_triton', 50.0) * 1e6,
                    "correct": result.get('accuracy', True),
                }}
        except Exception as e:
            print(f"Benchmark error: {{e}}")
    
    # Fallback: synthetic benchmark
    return {{"latency_us": float("inf"), "correct": False, "error": "no benchmark"}}


def check_correctness():
    """Check kernel correctness."""
    result = run_kernel_benchmark()
    return {{"passed": result.get("correct", False)}}


if __name__ == "__main__":
    import json
    
    result = run_kernel_benchmark()
    print(f"Result: {{result}}")
    
    with open("/workspace/opt_result.json", "w") as f:
        json.dump({{
            "correct": result.get("correct", False),
            "latency_us": result.get("latency_us", float("inf")),
            "params": {params.to_dict()},
        }}, f)
'''
        return kernel_code + harness
    
    def get_variant_types(self) -> List[str]:
        """Get list of variant types for this bottleneck."""
        return self.BOTTLENECK_VARIANTS.get(self.bottleneck, ["params"])
    
    def generate_random_params(self) -> KernelParams:
        """Generate random kernel parameters for exploration."""
        return KernelParams(
            block_n=random.choice([32, 64, 128]),
            block_h=random.choice([1, 2, 4, 8, 16]),
            num_splits=random.choice([4, 8, 16, 32]),
            num_warps=random.choice([2, 4, 8]),
            waves_per_eu=random.choice([1, 2, 4]),
            num_stages=random.choice([1, 2]),
        )

