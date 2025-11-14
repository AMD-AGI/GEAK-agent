# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/thunlp/TritonBench - Apache License 2.0
import torch
import triton
import triton.language as tl
import math

# Kernel function: Computes the cosine of each element in the input tensor.
@triton.jit
def cos_func(a, b, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block and thread
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Determine which elements are valid within the block
    mask = offset < n_elements
    # Load elements from tensor `a` into `a_value`
    a_value = tl.load(a + offset, mask=mask)
    # Compute the cosine of each element in `a_value`
    b_value = tl.cos(a_value.to(tl.float32))
    # Store the result back to tensor `b`
    tl.store(b + offset, b_value, mask=mask)  

# Function to invoke the Triton kernel and perform the computation
def cos(A):
    # Prepare output tensor `B` with the same shape and type as `A`
    B = torch.empty_like(A)
    # Determine the total number of elements in the input tensor `A`
    n_elements = A.numel()
    # Calculate the optimal block size
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    # Determine the grid size needed to cover all elements
    grid_size = triton.cdiv(n_elements, block_size)
    # Launch the Triton kernel
    cos_func[(grid_size, 1, 1)](A, B, n_elements, block_size)
    return B




