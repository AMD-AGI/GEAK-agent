import torch
import triton
import triton.testing as testing
import argparse
import importlib.util
import os
import sys
import json 
import random

def import_variable_from_file(file_path, variable_name):
    """
    Dynamically imports a variable from a Python file.

    Parameters:
    - file_path (str): The path to the Python file.
    - variable_name (str): The name of the variable to import.

    Returns:
    - The value of the specified variable, or None if not found.
    """

    if not os.path.isfile(file_path):
        # raise FileNotFoundError(f"No file found at {file_path}")
        return None, f"No file found at {file_path}"

    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # Create a module spec
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        return None, f"Could not load specification for module {module_name}"

    # Create a new module based on the spec
    module = importlib.util.module_from_spec(spec)
    if module is None:
        return None, f"Could not create module {module_name} from spec"

    # Execute the module in its own namespace
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return None, f"Could not execute module {module_name} due to {e}"

    # Retrieve the variable from the module
    return getattr(module, variable_name, None), "true"

def unit_test_formatter(x):
    if isinstance(x, torch.Tensor):
        return f"Tensor(shape={x.shape}, dtype={x.dtype}, device={x.device})"
    elif isinstance(x, (list, tuple)):
        return type(x).__name__ + "[" + ", ".join(unit_test_formatter(i) for i in x) + "]"
    elif isinstance(x, dict):
        return "{" + ", ".join(f"{k}: {unit_test_formatter(v)}" for k, v in x.items()) + "}"
    else:
        return repr(x)

def test_suite(wrapper_ref, wrapper, unit_tests,
               n_warmup=100, 
               n_iters=1000,
               atol=1e-3, rtol=1e-3,
               verbose=True):
    """
    Compare correctness and measure performance of two Triton kernel wrappers.
    Note that this function must be named as `test_suite` to be recognized by agents.
    Args:
        wrapper_ref: Callable reference kernel wrapper (ground truth)
        wrapper: Callable kernel wrapper to test
        device: Target device ('cuda' or 'hip')
        n_warmup: Warmup iterations before timing
        n_iters: Benchmark iterations per input
        atol, rtol: Tolerances for correctness check
        verbose: Print detailed output

    Returns:
        dict with 'avg_speedup', 'failed_tests', and 'num_tests'
    """

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    total_time_ref = 0.0
    total_time_test = 0.0
    failed_tests = 0
    num_tests = 0

    unit_benchmark = []

    for inputs in unit_tests:
        num_tests += 1

        # Create random input(s) – assuming single tensor input for simplicity
        # You can extend this to multiple arguments if needed.
        x = inputs
        
        try:
            if isinstance(x, list) or isinstance(x, tuple):
                y_ref = wrapper_ref(*x)
                y_ref = wrapper_ref(*x)  # Run twice to avoid cold start issues
            elif isinstance(x, dict):
                y_ref = wrapper_ref(**x)
                y_ref = wrapper_ref(**x)  # Run twice to avoid cold start issues
            else:
                y_ref = wrapper_ref(x)
                y_ref = wrapper_ref(x)  # Run twice to avoid cold start issues
        except Exception as e:
            return {
                "status": False,
                "msg": f"Error during execution for reference function: {e}"
            }

        try:
            if isinstance(x, list) or isinstance(x, tuple):
                y_test = wrapper(*x)
                y_test = wrapper(*x)  # Run twice to avoid cold start issues
            elif isinstance(x, dict):
                y_test = wrapper(**x)
                y_test = wrapper(**x)  # Run twice to avoid cold start issues   
            else:
                y_test = wrapper(x)
                y_test = wrapper(x)  # Run twice to avoid cold start issues
        except Exception as e:
            return {
                "status": False,
                "msg": f"Error during execution for generated function: {e}"
            }

        # Validate correctness
        try:
            torch.testing.assert_close(y_test, y_ref, atol=atol, rtol=rtol)
            if verbose:
                print(f"✅ Unit test {num_tests} passed.")
        except AssertionError as e:
            return {
                "status": False,
                "msg": f"❌ Unit test mismatch | {e}"
            }

        # Benchmark both
        if isinstance(x, list) or isinstance(x, tuple):
            def run_ref(): wrapper_ref(*x)
            def run_test(): wrapper(*x)
        elif isinstance(x, dict):
            def run_ref(): wrapper(**x)
            def run_test(): wrapper(**x)  # Run twice to avoid cold start issues   
        else:
            def run_ref():  wrapper(x)
            def run_test():  wrapper(x)  # Run twice to avoid cold start issues

        # Randomize order to avoid systematic bias
        if random.random() < 0.5:
            time_ref = testing.do_bench(run_ref, warmup=n_warmup, rep=n_iters, return_mode='median')
            time_test = testing.do_bench(run_test, warmup=n_warmup, rep=n_iters, return_mode='median')
        else:
            time_test = testing.do_bench(run_test, warmup=n_warmup, rep=n_iters, return_mode='median')
            time_ref = testing.do_bench(run_ref, warmup=n_warmup, rep=n_iters, return_mode='median')

        total_time_ref += time_ref
        total_time_test += time_test

        avg_time_ref = total_time_ref / num_tests
        avg_time_test = total_time_test / num_tests

        unit_benchmark.append(
            "For unit test {}  baseline kernel achieved {:.6f} ms median latency and the generated kernel achieved {:.6f} ms median latency.".format(unit_test_formatter(x), time_ref, time_test)
        )

        if verbose:
            print(f"⏱  ref={time_ref:.4f} ms | test={time_test:.4f} ms | speedup={time_ref/time_test:.2f}x")

    avg_speedup = total_time_ref / total_time_test if total_time_test > 0 else 0.0

    print("\n===== Test Summary =====")
    print(f"✅ Passed: {num_tests - failed_tests}/{num_tests}")
    print(f"⚡ Average speedup: {avg_speedup:.2f}x")

    return {
        "status": True,
        "avg_speedup": avg_speedup if failed_tests == 0 else 0.0,
        "benchmark": unit_benchmark,
        "failed_tests": failed_tests,
        "num_tests": num_tests,
        "avg_time_ref": avg_time_ref,
        "avg_time_test": avg_time_test,
        "msg": "All tests passed!" if failed_tests == 0 else f"{failed_tests} tests failed."
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Suite for Triton Kernels")
    parser.add_argument("--wrapper_ref_path", type=str, required=True, help="Reference wrapper function name")
    parser.add_argument("--wrapper_path", type=str, required=True, help="Test wrapper function name")
    parser.add_argument("--wrapper_name", type=str, default="add_wrapper", help="Name of the wrapper function to import from the files")
    parser.add_argument("--unit_test_path", type=str, default="templates/unit_tests.py", help="Path to the unit test definitions")
    parser.add_argument("--device", type=str, default="cuda", help="Target device GPU device")
    parser.add_argument("--n_warmup", type=int, default=10, help="Warmup iterations before timing")
    parser.add_argument("--n_iters", type=int, default=100, help="Benchmark iterations per input")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for correctness check")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for correctness check")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")

    args = parser.parse_args()

    get_input_tensors, msg_ut = import_variable_from_file(args.unit_test_path, "get_input_tensors")
    if get_input_tensors is None:
        print_msg = {
            "status": False,
            "msg": f"Error importing unit test function: {msg_ut}"
        }
        print(
            "```json\n" + json.dumps(print_msg) + "\n```"
        )
        sys.exit(1)

    unit_tests = get_input_tensors()

    wrapper_ref, msg_ref = import_variable_from_file(args.wrapper_ref_path, args.wrapper_name)
    if wrapper_ref is None:
        print_msg = {
            "status": False,
            "msg": f"Error importing reference wrapper: {msg_ref}"
        }
        print(
            "```json\n" + json.dumps(print_msg) + "\n```"
        )
        sys.exit(1)

    wrapper, msg_test = import_variable_from_file(args.wrapper_path, args.wrapper_name)
    if wrapper is None:
        print_msg = {
            "status": False,
            "msg": f"Error importing test wrapper: {msg_test}"
        }
        print(
            "```json\n" + json.dumps(print_msg) + "\n```"
        )
        sys.exit(1)

    results = test_suite(
        wrapper_ref, wrapper, unit_tests,
        # device=args.device,
        n_warmup=args.n_warmup,
        n_iters=args.n_iters,
        atol=args.atol,
        rtol=args.rtol,
        verbose=args.verbose
    )

    print(
        "```json\n" + json.dumps(results) + "\n```"
    )

"""
Example usage:
python test_suite.py --wrapper_ref_path templates/sample.py --wrapper_path templates/kernel.py --wrapper_name add_wrapper --device cuda --n_warmup 10 --n_iters 100 --atol 1e-3 --rtol 1e-3 --verbose   
"""
