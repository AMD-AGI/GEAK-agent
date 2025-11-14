# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
# Imports:  
# --------  
  
import argparse  
import itertools  
import os  
import sys  
from typing import Any, Callable, Optional  
  
import pytest  
import torch  
from torch import Tensor  
  
import triton  
import triton.language as tl  
 

######################## HELPER UTILS #####################
# Autotune configurations for Triton GEMM implemented with `tl.dot`.  
def get_triton_dot_autotune_configs() -> list[triton.Config]:  
    block_size_n_range: list[int] = [16, 32]  
    block_size_k_range: list[int] = [128, 256, 512]  
    kpack_range: list[int] = [1, 2]  
    num_warps_range: list[int] = [1, 2]  
    return [  
        triton.Config(  
            {  
                "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": block_size_n, "BLOCK_SIZE_K": block_size_k, "waves_per_eu": 0,  
                "matrix_instr_nonkdim": 16, "kpack": kpack  
            }, num_warps=num_warps, num_stages=2) for block_size_n, block_size_k, kpack, num_warps in itertools.product(  
                block_size_n_range, block_size_k_range, kpack_range, num_warps_range)  
    ]  
  
  
def get_triton_autotune_key() -> list[str]:  
    return ["M", "N", "K"]  
  
  
def get_triton_heuristics() -> dict[str, Callable[[dict[str, Any]], Any]]:  
    return {"EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0}  

###############################################################
# Triton GEMM:  
# ------------  
# Core Triton GEMM kernel.  
# Triton GEMM kernel implemented with `tl.dot`.  
@triton.autotune(configs=get_triton_dot_autotune_configs(), key=get_triton_autotune_key())  
@triton.heuristics(get_triton_heuristics())  
@triton.jit  
def triton_matmul_kernel(a_ptr, b_ptr, c_ptr, bias_ptr,  #  
                         M: int, N: int, K: int,  #  
                         stride_am: int, stride_ak: int,  #  
                         stride_bk: int, stride_bn: int,  #  
                         stride_cm: int, stride_cn: int,  #  
                         stride_bias: int,  #  
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #  
                         USE_BIAS: tl.constexpr, USE_DOT: tl.constexpr, EVEN_K: tl.constexpr  #  
                         ):  
    # Compute program ID:  
    pid = tl.program_id(axis=0)  
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  
    pid_m = pid // num_pid_n  
    pid_n = pid % num_pid_n  
  
    # Compute A and B base pointers:  
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  
    offs_k = tl.arange(0, BLOCK_SIZE_K)  
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak  
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn  
  
    # Load BIAS:  
    if USE_BIAS:  
        bias_ptrs = bias_ptr + offs_am * stride_bias  
        bias = tl.load(bias_ptrs, mask=offs_am < M, other=0)  
  
    # Initialize accumulator:  
    acc_dtype = tl.float32 if a_ptr.type.element_ty != tl.int8 else tl.int32  
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)  
  
    # GEMM loop:  
  
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):  
        if EVEN_K:  
            # Unmasked load of A and B:  
            a = tl.load(a_ptrs)  
            b = tl.load(b_ptrs)  
        else:  
            # Masked load of A and B:  
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)  
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)  
        # Compute dot product:  
        if USE_DOT:  
            accumulator += tl.dot(a, b)  
        else:  
            a = tl.reshape(a, (BLOCK_SIZE_M, BLOCK_SIZE_K, 1)).to(acc_dtype)  
            b = tl.reshape(b, (1, BLOCK_SIZE_K, BLOCK_SIZE_N)).to(acc_dtype)  
            accumulator += tl.sum(a * b, axis=1)  
        # Advance A and B pointers:  
        a_ptrs += BLOCK_SIZE_K * stride_ak  
        b_ptrs += BLOCK_SIZE_K * stride_bk  
  
    # Convert accumulator back to C's type:  
    c = accumulator.to(c_ptr.type.element_ty)  
  
    # Add BIAS:  
    if USE_BIAS:  
        c += bias[:, None]  
  
    # Compute C pointers and store C:  
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn  
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)  
    tl.store(c_ptrs, c, mask=c_mask)  
  
  
# # Triton GEMM kernel implemented with `tl.dot`.  
# @triton.autotune(configs=get_triton_dot_autotune_configs(), key=get_triton_autotune_key())  
# @triton.heuristics(get_triton_heuristics())  
# @triton.jit  
# def triton_dot_matmul_kernel(a_ptr, b_ptr, c_ptr, bias_ptr,  #  
#                              M: int, N: int, K: int,  #  
#                              stride_am: int, stride_ak: int,  #  
#                              stride_bk: int, stride_bn: int,  #  
#                              stride_cm: int, stride_cn: int,  #  
#                              stride_bias: int,  #  
#                              BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #  
#                              USE_BIAS: tl.constexpr, EVEN_K: tl.constexpr  #  
#                              ):  
#     triton_matmul_kernel(a_ptr, b_ptr, c_ptr, bias_ptr,  #  
#                          M, N, K,  #  
#                          stride_am, stride_ak,  #  
#                          stride_bk, stride_bn,  #  
#                          stride_cm, stride_cn,  #  
#                          stride_bias,  #  
#                          BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,  #  
#                          USE_BIAS=USE_BIAS, USE_DOT=True, EVEN_K=EVEN_K)  
 

  
  
##################################################################################################################################################   
  
# Test Triton GEMM, comparing it to PyTorch GEMM reference implementation:  
######################################## HELPERS for Eval ######################################## 
import numpy as np
import random
import torch 
import argparse  
import itertools  
import os  
import sys  
from typing import Any, Callable, Optional  
import pytest  
import torch  
from torch import Tensor  
import triton  
import triton.language as tl  
from tb_eval.perf.ROCm.performance_utils_pytest import PytestBenchmarker, do_bench_config, save_all_benchmark_results
from typing import Dict

result_gold = {}


def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility across multiple libraries and configure PyTorch for deterministic behavior.

    Args:
        seed (int): The seed value to set. Default is 42.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)
    # Set seed for NumPy
    np.random.seed(seed)
    # Set seed for PyTorch on CPU
    torch.manual_seed(seed)
    # Set seed for PyTorch on all GPUs (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variable for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)

######################################## HELPERS for Eval ######################################## 

    
def triton_matmul(triton_provider: str, a: Tensor, b: Tensor, bias: Optional[Tensor]) -> Tensor:  
    assert triton_provider in ["triton-dot"]  
  
    M: int  
    N: int  
    K: int  
    M, K = a.shape  
    _, N = b.shape  
  
    c: Tensor = torch.empty((M, N), device=a.device, dtype=a.dtype)  
  
    def grid(args: dict[str, Any]) -> tuple[int]:  
        return (triton.cdiv(M, args["BLOCK_SIZE_M"]) * triton.cdiv(N, args["BLOCK_SIZE_N"]), )  
  
    matmult_kernel = triton_matmul_kernel #triton_dot_matmul_kernel
  
    matmult_kernel[grid](  
        # Data pointers  
        a,  
        b,  
        c,  
        bias,  
        # Size of matrices  
        M,  
        N,  
        K,  
        # Strides  
        a.stride(0),  
        a.stride(1),  
        b.stride(0),  
        b.stride(1),  
        c.stride(0),  
        c.stride(1),  
        bias.stride(0) if bias is not None else 0,  
        # Other kernel parameters  
        USE_BIAS=bias is not None,  
        USE_DOT=True
    )  
  
    return c  



# PyTorch GEMM:  
# -------------  
  
  
def torch_matmul(a: Tensor, b: Tensor, bias: Optional[Tensor]) -> Tensor:  
    c: Tensor = torch.matmul(a, b)  
    if bias is not None:  
        c += bias[:, None]  
    return c  



# Wrapper for calling PyTorch GEMM or Triton GEMM:  
# ------------------------------------------------  
  
  
def matmul(provider: str, a: Tensor, b: Tensor, bias: Optional[Tensor]) -> Tensor:  
    assert provider in ["torch", "triton-dot"]  
  
    assert a.is_cuda, "Matrix A must be in GPU."  
    assert a.is_contiguous(), "Matrix A must be contiguous."  
    assert b.is_cuda, "Matrix B must be in GPU."  
    assert a.device == b.device, "Matrix A and matrix B must be in the same GPU."  
    assert a.dtype == b.dtype, "Matrix A and matrix B must have the same data type."  
    assert a.dim() == b.dim() == 2, "Matrix A and matrix B must be two-dimensional tensors."  
    assert a.shape[1] == b.shape[0], "Matrix A columns must be equal to matrix B rows."  
  
    if bias is not None:  
        assert bias.is_cuda, "Bias vector must be in GPU."  
        assert bias.is_contiguous(), "Bias vector must be continuous."  
        assert bias.device == a.device, "Matrix A and bias vector must be in the same GPU."  
        assert bias.dtype == a.dtype, "Matrix A and bias vector must have the same data type."  
        assert bias.dim() == 1, "Bias vector must be one-dimensional tensor."  
        assert bias.shape == (a.shape[0], ), "Bias vector length must be equal to matrix A rows."  
  
    if provider == "torch":  
        return torch_matmul(a, b, bias)  
  
    return triton_matmul(provider, a, b, bias)  

    
# Input generation:  
# -----------------  
  
  
def gen_input(M: int, N: int, K: int, use_bias: bool, device: str = "cuda") -> tuple[Tensor, Tensor, Optional[Tensor]]:  
    set_seed()
    assert M > 0, "M for input generation must be positive."  
    assert M <= 8, "M for input generation must be less or equal to 8."  
    assert N > 0, "N for input generation must be positive."  
    assert K > 0, "K for input generation must be positive."  
    a: Tensor = torch.randn((M, K), dtype=torch.float16, device=device)  
    b: Tensor = torch.randn((N, K), dtype=a.dtype, device=a.device).T  
    bias: Optional[Tensor] = torch.randn(M, dtype=a.dtype, device=a.device) if use_bias else None  
  
    return a, b, bias  

def gen_input_benchmark(M: int, N: int, K: int, use_bias: bool, device: str = "cuda", dtype=torch.float16) -> tuple[Tensor, Tensor, Optional[Tensor]]:
    set_seed()
    # Original gen_input had M <= 8 assertion, which might be too restrictive for general benchmarks.
    # Let's remove it for performance testing, assuming M can be larger.
    # assert M > 0, "M for input generation must be positive."
    # assert M <= 8, "M for input generation must be less or equal to 8." # Removed for perf
    # assert N > 0, "N for input generation must be positive."
    # assert K > 0, "K for input generation must be positive."
    a: Tensor = torch.randn((M, K), dtype=dtype, device=device)
    # Original b was (N,K).T. For GEMM A(M,K) @ B(K,N), b should be (K,N)
    b: Tensor = torch.randn((K, N), dtype=a.dtype, device=a.device) # Corrected shape for B
    bias: Optional[Tensor] = torch.randn(M, dtype=a.dtype, device=a.device) if use_bias else None
    return a, b, bias

def get_target_shapes() -> list[tuple[int, int, int]]:  
    # yapf: disable  
    return [  
        (1, 8192, 28672),   # Llama 70B  
        (1, 6144, 6144),    # Grok  
        (1, 4096, 4096),    # Generic GEMM  
        (2, 16384, 16384),  # Generic GEMM  
        (1, 4096, 3078),    # Uneven K  
        (1, 23, 31),        # Very small shape, uneven K  
        (1, 23, 128),       # Very small shape, even K  
    ]  
    # yapf: enable  
  
  
def allclose(x: Tensor, y: Tensor) -> bool:  
    return torch.allclose(x, y, atol=1e-3, rtol=1e-2)  
  
  
@pytest.mark.parametrize("use_bias", [False, True])  
@pytest.mark.parametrize("M, N, K", get_target_shapes())  
def test_matmul(M: int, N: int, K: int, use_bias: bool,request) -> None:  
    a: Tensor  
    b: Tensor  
    bias: Optional[Tensor]  
    a, b, bias = gen_input(M, N, K, use_bias)  
  
    c_torch: Tensor = matmul("torch", a, b, bias)  
    c_triton_dot: Tensor = matmul("triton-dot", a, b, bias)  

    ################### save c_triton_dot in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = c_triton_dot.clone().detach().cpu()
    ###################################################################
    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    assert allclose(c_torch, c_triton_dot), "PyTorch and Triton Dot results don't match."  


# --- Define TFLOPS and GB/s calculators for GEMM ---
def calculate_gemm_tflops(params: dict, ms: float) -> float:
    M, N, K = params['M'], params['N'], params['K']
    use_bias = params.get('use_bias', False)
    flops = 2 * M * N * K
    if use_bias: flops += M * N # Add M*N for bias addition
    tflops = flops / (ms / 1000) / 1e12
    return tflops

def calculate_gemm_gbps(params: dict, ms: float) -> float:
    M, N, K = params['M'], params['N'], params['K']
    use_bias = params.get('use_bias', False)
    dtype_str = params.get('dtype_str', 'fp16')

    if dtype_str == 'fp32': current_dtype = torch.float32
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16
    else: current_dtype = torch.float16
    element_size = torch.tensor([], dtype=current_dtype).element_size()

    bytes_a = M * K * element_size
    bytes_b = K * N * element_size
    bytes_c = M * N * element_size
    total_bytes = bytes_a + bytes_b + bytes_c
    if use_bias: total_bytes += M * element_size # Read bias
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

def get_target_shapes_for_perf() -> list[tuple[int, int, int]]: # Renamed for clarity
    return [
        (128, 8192, 4096),   # Larger M
        (512, 4096, 4096),
        (1024, 1024, 1024),  # Square
        (4096, 512, 2048),   # Different aspect ratios
        (1, 4096, 3078),     # Uneven K from original
        (16, 2048, 2048),    # Smaller M, larger N/K
        # (1, 23, 31),       # Very small shapes might not be ideal for peak perf
        # (1, 23, 128),
    ]

# --- Name for the benchmark output JSON file ---
OP_NAME_FOR_BENCHMARK = "multreduce_matmul_dot_perf"

# --- Pytest parametrize for performance testing ---
GEMM_DTYPES_FOR_PERF = ['fp16', 'bf16'] # 'fp32' can be added

@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("M, N, K", get_target_shapes_for_perf())
@pytest.mark.parametrize("dtype_str", GEMM_DTYPES_FOR_PERF)
def test_performance(M: int, N: int, K: int, use_bias: bool, dtype_str: str, request) -> None:
    set_seed()
    if dtype_str == 'fp32': current_dtype = torch.float32
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16
    else: current_dtype = torch.float16

    a, b, bias = gen_input_benchmark(M, N, K, use_bias, dtype=current_dtype)

    # --- Create op_lambda for benchmarking ---
    # We want to benchmark the triton_dot_matmul_kernel directly via its triton_matmul wrapper
    op_lambda = lambda: triton_matmul("triton-dot", a, b, bias)

    # --- Benchmarking ---
    bench_config = do_bench_config(warm_up=25, repetition=100)
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)

    current_params_for_logs_and_calc = {
        "M": M, "N": N, "K": K, "use_bias": use_bias, "dtype_str": dtype_str,
        "provider": "triton-dot"
    }

    benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
                              gbps_calculator=calculate_gemm_gbps,
                              tflops_calculator=calculate_gemm_tflops)


######################################## HELPERS for Eval ########################################     
# --- Pytest hook to save the dictionary at the end of the session ---  
def test_save_results():  
    """  
    Called after whole test run finished, right before returning the exit status to the system.  
    """
    print('Inside session finish...')
    if "_CALL_SUCCESS_" not in result_gold:
        result_gold['_CALL_SUCCESS_'] = torch.tensor([[0.0]])
    OUTPUT_FILENAME = __file__.replace('.','_') + '.pt'
    print(f"\nSaving all c_triton_dot results to {OUTPUT_FILENAME}...")  
    # Ensure the directory for the output file exists if it's in a subdirectory  
    output_dir = os.path.dirname(OUTPUT_FILENAME)  
    if output_dir and not os.path.exists(output_dir):  
        os.makedirs(output_dir, exist_ok=True)  
    torch.save(result_gold, OUTPUT_FILENAME)       
    print(f"Successfully saved {len(result_gold)} c_triton_dot tensors to {OUTPUT_FILENAME}.")  

def test_save_performance_results():
    """
    Called after the test_performance function finishes.
    This is a separate hook to ensure performance results are saved.
    """
    print('\nPytest session finishing... Saving benchmark results...')

    output_directory = os.path.join(os.path.dirname(__file__), "perf")  # Save in a "perf" subdirectory next to the test file
    os.makedirs(output_directory, exist_ok=True)
    
    save_all_benchmark_results(output_directory)
    print(f"All benchmark results attempted to save to: {output_directory}")

######################################## HELPERS for Eval ######################################## 
  
# Benchmark Triton GEMM, comparing it to PyTorch GEMM reference implementation:  
# -----------------------------------------------------------------------------  
  
  
# Convert milliseconds to GiB/s.  
def ms_to_gibps(M: int, N: int, K: int, milliseconds: float) -> float:  
    read_elems: int = M * K + K * N  
    write_elems: int = M * N  
    transf_elems: int = read_elems + write_elems  
    transf_bytes: int = 2 * transf_elems  # times 2 due to fp16  
    transf_gibibytes: float = 2**-30 * transf_bytes  
    seconds: float = 1e-3 * milliseconds  
    return round(transf_gibibytes / seconds, 2)  
  
  
def run_benchmark(use_bias: bool) -> None:  
    perf_unit: str = "GiB/s"  
    line_vals: list[str] = ["torch", "triton-dot"]  
    line_names: list[str] = [f"{x.replace('-', ' ').title()} ({perf_unit})" for x in line_vals]  
  
    # Triton benchmark:  
    @triton.testing.perf_report(  
        triton.testing.Benchmark(  
            x_names=["M", "N", "K"],  
            x_vals=get_target_shapes(),  
            line_arg="provider",  
            line_vals=line_vals,  
            line_names=line_names,  
            ylabel=perf_unit,  
            args={},  
            plot_name=f"fp16_{os.path.splitext(os.path.basename(__file__))[0]}",  
        ))  
    def benchmark(M: int, N: int, K: int, provider: str) -> tuple[float, float, float]:  
  
        def perf(milliseconds: float) -> float:  
            return ms_to_gibps(M, N, K, milliseconds)  
  
        a: Tensor  
        b: Tensor  
        bias: Optional[Tensor]  
        a, b, bias = gen_input(M, N, K, use_bias)  
  
        p20_ms: float  
        p50_ms: float  
        p80_ms: float  
        p20_ms, p50_ms, p80_ms = triton.testing.do_bench(lambda: matmul(provider, a, b, bias),  
                                                         quantiles=[0.2, 0.5, 0.8])  
  
        p20_gibps: float = perf(p80_ms)  
        p50_gibps: float = perf(p50_ms)  
        p80_gibps: float = perf(p20_ms)  
  
        print(", ".join([  
            f"(M, N, K) = {(M, N, K)}",  
            f"provider = {provider}",  
            f"p20 = {p20_gibps} {perf_unit}",  
            f"p50 = {p50_gibps} {perf_unit}",  
            f"p80 = {p80_gibps} {perf_unit}",  
        ]))  
  
        if provider == "triton-dot":  
            print(f"Triton Dot kernel best config = {triton_matmul_kernel.best_config}")   #{triton_dot_matmul_kernel.best_config}")   
  
        return p50_gibps, p20_gibps, p80_gibps  
  
    print(f"Running benchmark (use_bias = {use_bias})...")  
    benchmark.run(show_plots=False, print_data=True)  
    print("Done.")  
  
  
# Script entry point:  
# -------------------  
  
  
def positive_int(value: str) -> int:  
    try:  
        int_value = int(value)  
    except ValueError:  
        raise argparse.ArgumentTypeError(f"{value} is not an integer.")  
    if int_value <= 0:  
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer.")  
    return int_value  
  
  
def parse_args() -> argparse.Namespace:  
    parser = argparse.ArgumentParser(  
        description="C = A * B + BIAS matrix multiplication kernel for small matrices (M ≤ 8)",  
        formatter_class=argparse.RawTextHelpFormatter)  
    parser.add_argument(  
        "mode", choices=["bench"], help="mode of operation:\n"  
        "  run: run Triton kernel for a given (M, N, K) shape\n"  
        "  bench: benchmark performance for target shapes\n")  
    shape_group = parser.add_argument_group("kernel shape arguments")  
    shape_group.add_argument("-M", type=positive_int, help="rows of matrix A (must be less or equal to 8)")  
    shape_group.add_argument("-N", type=positive_int, help="columns of matrix A / rows of matrix B")  
    shape_group.add_argument("-K", type=positive_int, help="columns of matrix B")  
    shape_group.add_argument("--use-bias", default=False, action="store_true", help="use BIAS vector")  
    shape_group.add_argument("--use-dot", default=False, action="store_true", help="use tl.dot for dot product")  
    args = parser.parse_args()  
    if args.mode == "run":  
        try:  
            sizes: tuple[Optional[int], ...] = tuple(size for size in (args.M, args.N, args.K))  
            if any(size is None for size in sizes):  
                raise ValueError(f"(M, N, K) = {sizes}, all sizes must be specified together.")  
            if args.M > 8:  
                raise ValueError(f"M = {args.M} is too big, this kernel was designed for M ≤ 8.")  
        except ValueError as arg_error:  
            print(arg_error)  
            sys.exit(1)  
    return args  
  
  
def main() -> int:  
    args: argparse.Namespace = parse_args()  
    status: int = 0  
    try:  
        match args.mode:  
            case "bench":  
                run_benchmark(args.use_bias)  
    except KeyboardInterrupt:  
        print("\nInterrupted.")  
    except Exception as error:  
        print(f"\nUnexpected error: {error}")  
        status = 1  
    return status  
  
  
if __name__ == "__main__":  
    sys.exit(main())  
