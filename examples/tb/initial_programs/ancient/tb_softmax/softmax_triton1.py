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

#************************** BASELINE FUNCTION **************************
@triton.jit
def baseline_softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

def baseline_softmax(x):
    n_rows, n_cols = x.shape
    # The block size is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # Allocate output
    y = torch.empty_like(x)
    # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row o
    # f the input matrix
    baseline_softmax_kernel[(n_rows, )](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y

#************************** BASELINE FUNCTION END **************************

#************************** CANDIDATE TRITON KERNEL **************************
# EVOLVE-BLOCK-START
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

def softmax(x):
    n_rows, n_cols = x.shape
    # The block size is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # Allocate output
    y = torch.empty_like(x)
    # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row o
    # f the input matrix
    softmax_kernel[(n_rows, )](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
# EVOLVE-BLOCK-END
#************************** CANDIDATE TRITON KERNEL END **************************

##################################################################################################################################################
# Test cases for the flash_attn_triton function
def test_kernel_triton():
    dim1, dim2 = 64, 64
    # Create random input tensors
    x = torch.randn((dim1, dim2), dtype=torch.float16, device='cuda')
    # Test with causal=True
    output = softmax(x)
    baseline = baseline_softmax(x)

    result = output.allclose(baseline, atol=1e-2, rtol=1e-2)

    return result

def test_kernel_call():
    """
    Test the flash_attn_triton function with a simple call.
    This is a sanity check to ensure the kernel can be called without errors.
    """
    dim1, dim2 = 64, 64
    # Create random input tensors
    x = torch.randn((dim1, dim2), dtype=torch.float16, device='cuda')

    # Call the flash_attn_triton function
    try:
        softmax(x)
        return True
    except Exception as e:
        print(f"Error during kernel call: {e}", file=sys.stderr)
        return False

def run_benchmark(dim1, dim2):
    """    Run the benchmark for the flash_attn_triton function with given parameters.
    Args:        batch_size (int): Batch size for the input tensors.
        num_head (int): Number of attention heads.
        seq_len (int): Sequence length of the input tensors.
        dim (int): Dimension of the input tensors.
        sm_scale (float): Scale factor for the softmax computation.
    """

    # Create random input tensors
    x = torch.randn((dim1, dim2), dtype=torch.float16, device='cuda')
 
    # Benchmark the flash_attn_triton function
    ms, min_ms, max_ms =  triton.testing.do_bench(
        lambda: softmax(x),
        warmup=500,
        rep=1000,
        quantiles=[0.5, 0.8, 0.2],
        return_mode="median"
    )

    # Benchmark the flash_attn_triton function
    baseline_ms, baseline_min_ms, baseline_max_ms =  triton.testing.do_bench(
        lambda: baseline_softmax(x),
        warmup=500,
        rep=1000,
        quantiles=[0.5, 0.8, 0.2],
        return_mode="median"
    )

    return {
        "dim1": dim1,
        "dim2": dim2,
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
    dim1s = [64, 128, 256, 512]
    dim2s = [64, 128, 256, 512]

    all_args = []
    for dim1 in dim1s:
        for dim2 in dim2s:
            all_args.append((dim1, dim2))
    benchmark_results = []

    for dim1, dim2 in all_args:
                    print(f"Benchmarking with dim1={dim1}, dim2={dim2}")
                    # Run the benchmark
                    run_result = run_benchmark(dim1, dim2)
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
    return test_kernel_triton()

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
