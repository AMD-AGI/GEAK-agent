#!/usr/bin/env python3
"""
Quick test of the Triton evaluator
"""
import sys
sys.path.insert(0, '/home/sapmajum/geak-openevolve')

from examples.tb.triton_evaluator import evaluate

# Test with the sample kernel
kernel_path = '/home/sapmajum/geak-openevolve/examples/tb/samples/kernel.py'
test_suite_path = '/home/sapmajum/geak-openevolve/examples/tb/samples/test_suite.py'
ref_wrapper_path = '/home/sapmajum/geak-openevolve/examples/tb/samples/kernel.py'
unit_tests_path = '/home/sapmajum/geak-openevolve/examples/tb/samples/unit_tests.py'

print("=" * 70)
print("Testing Triton Evaluator with Sample Add Kernel")
print("=" * 70)

result = evaluate(
    test_suite_path=test_suite_path,
    program_text=kernel_path,
    ref_wrapper_path=ref_wrapper_path,
    wrapper_fn_name='test_add',
    unit_tests_path=unit_tests_path,
    n_warmup=5,
    n_iters=20,
    atol=1e-3,
    rtol=1e-3,
    verbose=False,
    gpu_id=0
)

print("\n" + "=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)
print(f"Success: {result['success']}")
print(f"Final Score: {result['final_score']}")
print(f"Correctness Score: {result['correctness_score']}")
print(f"Combined Score: {result['combined_score']}")
print(f"\nSummary: {result['summary']}")
if result.get('error'):
    print(f"\nError: {result['error']}")
if result.get('benchmark_results'):
    print(f"\nBenchmark Results:")
    for br in result['benchmark_results']:
        print(f"  - {br}")
print("=" * 70)

