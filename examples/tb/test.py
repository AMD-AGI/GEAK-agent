# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/algorithmicsuperintelligence/openevolve  - Apache License 2.0

import os
import sys
sys.path.append("/home/vinajosh/openevolve/examples/tb/initial_programs/softmax")

import math
from typing import Optional, Tuple, Any
import time
import argparse

import torch
import triton
import triton.language as tl
import triton.testing
from tqdm import tqdm

DEVICE = triton.runtime.driver.active.get_active_torch_device()

from baseline import baseline_softmax

#************************** CANDIDATE TRITON KERNEL **************************
# *Only edit the code below withing EVOLVE-BLOCK-START and EVOLVE-BLOCK-END tags *
# EVOLVE-BLOCK-START
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32, 'waves_per_eu': 1, 'kpack': 1, 'matrix_instr_nonkdim': 16}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_SIZE': 64, 'waves_per_eu': 1, 'kpack': 1, 'matrix_instr_nonkdim': 16}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 64, 'waves_per_eu': 2, 'kpack': 2, 'matrix_instr_nonkdim': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128, 'waves_per_eu': 1, 'kpack': 2, 'matrix_instr_nonkdim': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128, 'waves_per_eu': 2, 'kpack': 2, 'matrix_instr_nonkdim': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256, 'waves_per_eu': 2, 'kpack': 2, 'matrix_instr_nonkdim': 16}, num_warps=16, num_stages=3),
    ],
    key=['n_cols']
)
@triton.jit
def softmax_kernel(in_ptr, output_ptr, row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_offset = pid * row_stride

    # If the entire row fits in one block, use a single-pass softmax to reduce memory traffic.
    if n_cols <= BLOCK_SIZE:
        col_range = tl.arange(0, BLOCK_SIZE)
        mask = col_range < n_cols
        block = tl.load(in_ptr + row_offset + col_range, mask=mask, other=-float('inf'))
        max_val = tl.max(block, axis=0)
        exp_block = tl.exp(block - max_val)
        sum_exp = tl.sum(exp_block, axis=0)
        out = exp_block / sum_exp
        tl.store(output_ptr + row_offset + col_range, out, mask=mask)
    else:
        max_val = -float('inf')
        exp_sum = 0.0
        for offset in range(0, n_cols, BLOCK_SIZE):
            col_range = tl.arange(0, BLOCK_SIZE)
            col_mask = col_range < (n_cols - offset)
            block_ptr = in_ptr + row_offset + offset
            block = tl.load(block_ptr + col_range, mask=col_mask, other=-float('inf'))
            block_max = tl.max(block, axis=-1)
            new_max = tl.maximum(max_val, block_max)
            exp_sum = exp_sum * tl.exp(max_val - new_max) + tl.sum(tl.exp(block - new_max), axis=-1)
            max_val = new_max

        for offset in range(0, n_cols, BLOCK_SIZE):
            col_range = tl.arange(0, BLOCK_SIZE)
            col_mask = col_range < (n_cols - offset)
            block_ptr = in_ptr + row_offset + offset
            out_ptr = output_ptr + row_offset + offset
            block = tl.load(block_ptr + col_range, mask=col_mask)
            out = tl.exp(block - max_val) / exp_sum
            tl.store(out_ptr + col_range, out, mask=col_mask)

def softmax(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row of the input matrix.
    softmax_kernel[(n_rows, )](
        y,
        x,
        x.stride(0),
        n_cols,
    )
    return y
# EVOLVE-BLOCK-END
#************************** CANDIDATE TRITON KERNEL **************************

########################### DO NOT EDIT BELOW THIS LINE ###########################
# Test cases for the triton function
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
    dim1, dim2 = 64, 64
    # Create random input tensors
    x = torch.randn((dim1, dim2), dtype=torch.float16, device='cuda')
    # Call the triton function
    softmax(x)
    return True

def run_benchmark(dim1, dim2):
    # Create random input tensors
    x = torch.randn((dim1, dim2), dtype=torch.float16, device='cuda')

    # Benchmark the triton function
    ms, min_ms, max_ms =  triton.testing.do_bench(
        lambda: softmax(x),
        warmup=500,
        rep=1000,
        quantiles=[0.5, 0.8, 0.2],
        return_mode='median'
    )

    # Benchmark the triton function
    baseline_ms, baseline_min_ms, baseline_max_ms =  triton.testing.do_bench(
        lambda: baseline_softmax(x),
        warmup=500,
        rep=1000,
        quantiles=[0.5, 0.8, 0.2],
        return_mode='median'
    )

    return {
        'dim1': dim1,
        'dim2': dim2,
        'median_time_ms': ms,
        'min_time_ms': min_ms,
        'max_time_ms': max_ms,
        'baseline_median_time_ms': baseline_ms,
        'baseline_min_time_ms': baseline_min_ms,
        'baseline_max_time_ms': baseline_max_ms,
        'speedup': baseline_ms / ms if ms > 0 else None,
    }

def benchmark_kernel(verbose: Optional[int] = 0) -> Optional[float]:
    # Define the benchmark parameters
    dim1s = [64, 8192] #128, 256, 512, 1024, 2048, 4096, 8192]
    dim2s = [64, 8192] #128, 256, 512, 1024, 2048, 4096, 8192]

    all_args = []
    for dim2 in dim1s:
        for dim1 in dim2s:
            all_args.append((dim1, dim2))
    benchmark_results = []

    for dim1, dim2 in all_args:
        if verbose > 1:
            print(f'Benchmarking with dim1={dim1}, dim2={dim2}')
        # Run the benchmark
        run_result = run_benchmark(dim1, dim2)
        benchmark_results.append(run_result)

    ## compute average speedup across all runs
    total_speedup = 0
    count = 0
    for result in benchmark_results:
        if result['speedup'] is not None:
            total_speedup += result['speedup']
            count += 1

    average_speedup = total_speedup / count if count > 0 else None
    return average_speedup, benchmark_results



def test_kernel_correctness():
    return test_kernel_triton()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark and test Triton kernel.')
    parser.add_argument('--call', '-c', action='store_true', help='Run correctness test for the Triton kernel.')
    parser.add_argument('--test', '-t', action='store_true', help='Run correctness test for the Triton kernel.')
    parser.add_argument('--benchmark', '-b', action='store_true', help='Run performance benchmark for the Triton kernel.')
    parser.add_argument('--all', '-a', action='store_true', help='Run all tests and benchmarks.')
    parser.add_argument('--npasses', '-n', type=int, default=1, help='Number of passes for the benchmark.')
    parser.add_argument('--verbose', '-v', type=int, default=0, help='Set verbosity level (0: no output, 1: basic output, 2: detailed output).')
    parser.add_argument('--ret_params', '-r', type=str, default='median_time_ms,min_time_ms,max_time_ms,baseline_median_time_ms,baseline_min_time_ms,baseline_max_time_ms,speedup',
                        help='Comma-separated list of return parameters. Options: call, test, benchmark, err_msg.')
    args = parser.parse_args()

    for iN in tqdm(range(args.npasses)):
        if args.verbose > 0:
            print(f'Running pass {iN + 1}/{args.npasses}...')
        call, test, benchmark = None, None, None
        err_msg = ' '

        if args.call or args.all:
            try:
                call = test_kernel_call()
            except Exception as e:
                print(f'Error during kernel call: {e}', file=sys.stderr)
                err_msg = f"Error in calling kernel: {e}"
                call = False

        if (args.test or args.all) and call:
            try:
                test = test_kernel_correctness()
            except Exception as e:
                print(f'Error during kernel correctness test: {e}', file=sys.stderr)
                err_msg = f"Error in testing kernel correctness: {e}"
                test = False

        results_list = []
        if (args.benchmark or args.all) and call and test:
            try:
                benchmark, results_list = benchmark_kernel(args.verbose)
            except Exception as e:
                print(f'Error during kernel benchmark: {e}', file=sys.stderr)
                err_msg = f"Error in benchmarking kernel: {e}"
                benchmark = None

        param_vals = None
        if results_list:
            _param_vals = []
            input_shapes = []
            metrics = []
            baselines = []
            # params_to_return = args.ret_params.split(',')
            # for param in params_to_return:
            #     param_vals += f"{results_list.get(param, None)};"
            for result in results_list:
                _input_shapes = []
                _metrics = []
                _baselines = []
                for k, v in result.items():
                    if "dim" in k:
                        _input_shapes.append(f"{v}")
                    elif "baseline_median_time_ms" in k:
                        _baselines.append(f"{v}")
                    elif "median_time_ms" in k:
                        _metrics.append(f"{v}")
                input_shapes = "(" + "x".join(_input_shapes) + ")"
                metrics = "(" + ",".join(_metrics) + ")"
                baselines = "(" + ",".join(_baselines) + ")"
                _param_vals.append(f" For input shapes: {input_shapes} generated kernel achieved median latency of {metrics} ms, with baseline median latency of {baselines} ms.  ")
            param_vals = ";".join(_param_vals)

        print(f'{call}#*#*{test}#*#*{benchmark}#*#*{param_vals}#*#*{err_msg}')