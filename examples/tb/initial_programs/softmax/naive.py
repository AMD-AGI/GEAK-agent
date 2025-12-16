# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/algorithmicsuperintelligence/openevolve  - Apache License 2.0

import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(in_ptr, output_ptr, row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    in_max = -float('inf')
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_range = tl.arange(0, BLOCK_SIZE)
        col_mask = col_range + offset < n_cols
        in_data = tl.load(in_ptr + pid * row_stride + col_range + offset, mask=col_mask, other=-float('inf'))
        in_max = tl.maximum(in_max, tl.max(in_data, axis=-1))
    
    in_exp_sum = 0.0
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_range = tl.arange(0, BLOCK_SIZE)
        col_mask = col_range + offset < n_cols
        in_data = tl.load(in_ptr + pid * row_stride + col_range + offset, mask=col_mask, other=-float('inf'))
        in_exp_sum = in_exp_sum + tl.sum(tl.exp(in_data - in_max), axis=-1)
    
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_range = tl.arange(0, BLOCK_SIZE)
        col_mask = col_range + offset < n_cols
        in_data = tl.load(in_ptr + pid * row_stride + col_range + offset, mask=col_mask)
        in_exp = tl.exp(in_data - in_max)
        tl.store(output_ptr + pid * row_stride + col_range + offset, in_exp / in_exp_sum, mask=col_mask)

def softmax(x):
    n_rows, n_cols = x.shape
    # The block size is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row o
    # f the input matrix
    softmax_kernel[(n_rows, )](
        y,
        x,
        x.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
