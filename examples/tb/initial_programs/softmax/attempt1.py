# EVOLVE-BLOCK-START
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=2),
    ],
    key=['n_cols']
)
@triton.jit
def softmax_kernel(in_ptr, output_ptr, row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    base = tl.arange(0, BLOCK_SIZE)

    running_max = -float('inf')
    running_sum = 0.0
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_range = base + offset
        col_mask = col_range < n_cols
        x = tl.load(in_ptr + pid * row_stride + col_range, mask=col_mask, other=-float('inf'))
        block_max = tl.max(x, axis=-1)
        new_max = tl.maximum(running_max, block_max)
        running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(tl.exp(x - new_max), axis=-1)
        running_max = new_max
    
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_range = base + offset
        col_mask = col_range < n_cols
        x = tl.load(in_ptr + pid * row_stride + col_range, mask=col_mask, other=-float('inf'))
        tl.store(output_ptr + pid * row_stride + col_range, tl.exp(x - running_max) / running_sum, mask=col_mask)

def softmax(x):
    n_rows, n_cols = x.shape

    y = torch.empty_like(x)
    # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row o
    # f the input matrix
    softmax_kernel[(n_rows, )](
        y,
        x,
        x.stride(0),
        n_cols,
        # BLOCK_SIZE=2048,  # You can adjust this based on your GPU's capabilities

    )
    return y
# EVOLVE-BLOCK-END
