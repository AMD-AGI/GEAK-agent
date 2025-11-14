# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/thunlp/TritonBench - Apache License 2.0
import torch
import triton
import triton.language as tl

# Kernel function using Triton
@triton.jit
def kernel_function(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # x_ptr: pointer to input data
    # output_ptr: pointer to output data
    # n_elements: number of elements to process
    # BLOCK_SIZE: block size for Triton kernel
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.math.sin(x)
    tl.store(output_ptr + offsets, output, mask=mask)

# Function to call the Triton kernel
def call_kernel(x):
    # x: input tensor
    n_elements = x.numel()
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    kernel_function[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output




