# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/thunlp/TritonBench - Apache License 2.0
import triton
import triton.language as tl
import torch

@triton.jit
def relu_kernel(x_ptr, out_ptr, N: tl.constexpr, block_size: tl.constexpr):
    # Get the index of the current thread
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < N

    # Load elements from global memory
    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute linear layer
    result = tl.where(x >= 0, x, 0.0)

    # Write result to global memory
    if pid == 0:
        tl.store(out_ptr + offsets, result, mask=mask)

def relu(x):
    # Prepare output tensor
    out = torch.empty_like(x, dtype=torch.float32, device=x.device)
    N = out.numel()

    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE  # Calculate the number of blocks needed

    # Launch Triton kernel
    grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE), )

    relu_kernel[grid](x, out, N, BLOCK_SIZE)

    return out



