# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/thunlp/TritonBench - Apache License 2.0
import triton
import triton.language as tl
import torch

@triton.jit
def kernel_f8_to_f16(Y, X, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    tl.store(Y + offs, x, mask=mask)
    tl.store(Y + offs, x, mask=mask)

def f8_to_f16(x, dtypes=tl.float8e5) -> torch.Tensor:
    assert x.dtype == torch.int8, f"torch.int8 expected but got {x.dtype}"
    assert "cuda" in str(x.device), f"CUDA tensors only but got {x.device}"
    ret = torch.empty_like(x, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']),)
    numel = ret.untyped_storage().size() // ret.element_size()
    kernel_f8_to_f16[grid](ret, triton.reinterpret(x, dtypes), numel, BLOCK_SIZE=1024)
    return ret

@triton.jit
def kernel_f16_to_f8(Y, X, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    tl.store(Y + offs, x, mask=mask)

def f16_to_f8(x: torch.Tensor, dtypes=tl.float8e5) -> torch.Tensor:
    assert x.dtype in [torch.float16, torch.float32]
    assert "cuda" in str(x.device), f"CUDA tensors only but got {x.device}"
    ret = torch.empty_like(x, dtype=torch.int8)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']),)
    numel = x.untyped_storage().size() // x.element_size()
    kernel_f16_to_f8[grid](triton.reinterpret(ret, dtypes), x, numel, BLOCK_SIZE=1024)
    return ret




