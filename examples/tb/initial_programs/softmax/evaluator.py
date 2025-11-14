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
from tqdm import tqdm
from baseline import softmax as baseline_softmax
import importlib

# Test cases for the triton function
def test_kernel_triton():
    dim1, dim2 = 64, 64
    # Create random input tensors
    x = torch.randn((dim1, dim2), dtype=torch.float16, device='cuda')
    # Test with causal=True
    output = softmax(x)
    baseline = baseline_softmax(x)

    result = output.allclose(baseline, atol=1e-2) #, rtol=1e-2)

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
    baseline_ms, baseline_min_ms, baseline_max_ms =  triton.testing.do_bench(
        lambda: baseline_softmax(x),
        warmup=500,
        rep=1000,
        quantiles=[0.5, 0.8, 0.2],
        return_mode='median'
    )

    # Benchmark the triton function
    ms, min_ms, max_ms =  triton.testing.do_bench(
        lambda: softmax(x),
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
    dim1s = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    dim2s = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

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
    baseline_avg_med = 0
    avg_med = 0
    for result in benchmark_results:
        if result['baseline_median_time_ms'] is not None:
            baseline_avg_med += result['baseline_median_time_ms']
            count += 1

        if result['median_time_ms'] is not None:
            avg_med += result['median_time_ms']
            count += 1

        if result['speedup'] is not None:
            total_speedup += result['speedup']
            count += 1

    average_speedup = baseline_avg_med / avg_med if avg_med > 0 else None
    return average_speedup, benchmark_results

def test_kernel_correctness():
    return test_kernel_triton()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark and test Triton kernel.')
    parser.add_argument('--call', '-c', action='store_true', help='Run correctness test for the Triton kernel.')
    parser.add_argument('--test', '-t', action='store_true', help='Run correctness test for the Triton kernel.')
    parser.add_argument('--benchmark', '-b', action='store_true', help='Run performance benchmark for the Triton kernel.')
    parser.add_argument('--all', '-a', action='store_true', help='Run all tests and benchmarks.')
    parser.add_argument('--kernel_path', '-k', type=str, default='baseline.py',
                        help='Path to the Triton kernel file. Default is "baseline.py".')
    parser.add_argument('--npasses', '-n', type=int, default=1, help='Number of passes for the benchmark.')
    parser.add_argument('--verbose', '-v', type=int, default=0, help='Set verbosity level (0: no output, 1: basic output, 2: detailed output).')  
    parser.add_argument('--ret_params', '-r', type=str, default='median_time_ms,min_time_ms,max_time_ms,baseline_median_time_ms,baseline_min_time_ms,baseline_max_time_ms,speedup',
                        help='Comma-separated list of return parameters. Options: call, test, benchmark, err_msg.')
    args = parser.parse_args()

    kernel_path = args.kernel_path.strip()

    print(f"Reading kernel path from command line argument: {kernel_path}")

    assert os.path.exists(kernel_path), f"Kernel path {kernel_path} does not exist."

    spec = importlib.util.spec_from_file_location("softmax", kernel_path)
    if spec is None:
        raise ImportError(f"Could not find module {kernel_path}")
    softmax_module = importlib.util.module_from_spec(spec)
    sys.modules["softmax"] = softmax_module
    spec.loader.exec_module(softmax_module)
    globals().update(softmax_module.__dict__)

    for iN in tqdm(range(args.npasses)):
        if args.verbose > 0:
            print(f'Running pass {iN + 1}/{args.npasses}...')
        call, test, benchmark = None, None, None
        err_msg = ' '

        if args.call or args.all:
            try:
                call = test_kernel_call()
                if not call:
                    err_msg = "Error in calling kernel."
            except Exception as e:
                print(f'Error during kernel call: {e}', file=sys.stderr)
                err_msg = f"Error in calling kernel: {e}"
                call = False

        if (args.test or args.all) and call:
            try:
                test = test_kernel_correctness()
                if not test:
                    err_msg = "Error in testing kernel correctness."
            except Exception as e:
                print(f'Error during kernel correctness test: {e}', file=sys.stderr)
                err_msg = f"Error in testing kernel correctness: {e}"
                test = False

        results_list = []
        if (args.benchmark or args.all) and call and test:
            try:
                benchmark, results_list = benchmark_kernel(args.verbose)
                if not benchmark:
                    err_msg = "Error in benchmarking kernel."
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
                        _baselines.append(f"{v:.6f}")
                    elif "median_time_ms" in k:
                        _metrics.append(f"{v:.6f}")
                input_shapes = "(" + "x".join(_input_shapes) + ")"
                metrics = ",".join(_metrics)
                baselines = ",".join(_baselines)
                _param_vals.append(f" For input shapes: {input_shapes} generated kernel achieved median latency of {metrics} milliseconds, with baseline kernel median latency of {baselines} milliseconds.  ")
            param_vals = "\n".join(_param_vals)

        print(f'{call}#*#*{test}#*#*{benchmark}#*#*{param_vals}#*#*{err_msg}')
