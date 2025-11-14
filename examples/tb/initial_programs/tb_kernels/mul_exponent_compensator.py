# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/thunlp/TritonBench - Apache License 2.0
import torch
import triton
import triton.language as tl

# Triton kernel to multiply each element in the source tensor by a constant exponent compensator
@triton.jit
def mul_kernel(src, dst, BLOCK_SIZE: tl.constexpr):
    # Define a constant exponent compensator
    exponent_compensator: tl.constexpr = 2.0 ** (127 - 15)
    # Calculate the indices for the current program ID
    idxs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Load elements from the source tensor
    x = tl.load(src + idxs)
    # Multiply each element by the exponent compensator
    y = x * exponent_compensator
    # Store the result in the destination tensor
    tl.store(dst + idxs, y)

# Function to launch the Triton kernel
def launch_mul_kernel(src, BLOCK_SIZE=1):
    # Create an empty tensor for the result
    dst = torch.empty(src.shape, dtype=torch.float32, device='cuda')
    # Launch the Triton kernel
    mul_kernel[(src.shape[0] // BLOCK_SIZE,)](src, dst, BLOCK_SIZE)
    return dst




