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
  

# Triton GEMM:  
# ------------  

######################## HELPER UTILS #####################  
# Autotune configurations for Triton GEMM implemented with explicit dot product.  
def get_triton_multreduce_autotune_configs() -> list[triton.Config]:  
    block_size_k_range: list[int] = [128, 256, 512]  
    kpack_range: list[int] = [1, 2]  
    return [  
        triton.Config(  
            {"BLOCK_SIZE_M": 1, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": block_size_k, "waves_per_eu": 0, "kpack": kpack},  
            num_warps=8, num_stages=2) for block_size_k, kpack in itertools.product(block_size_k_range, kpack_range)  
    ]  
  
  
def get_triton_autotune_key() -> list[str]:  
    return ["M", "N", "K"]  
  
  
def get_triton_heuristics() -> dict[str, Callable[[dict[str, Any]], Any]]:  
    return {"EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0}  
  
######################## HELPER UTILS #####################


# Core Triton GEMM kernel.  
# Triton GEMM kernel implemented with explicit dot product.  
@triton.autotune(configs=get_triton_multreduce_autotune_configs(), key=get_triton_autotune_key())  
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
  
  
# # Triton GEMM kernel implemented with explicit dot product.  
# @triton.autotune(configs=get_triton_multreduce_autotune_configs(), key=get_triton_autotune_key())  
# @triton.heuristics(get_triton_heuristics())  
# @triton.jit  
# def triton_multreduce_matmul_kernel(a_ptr, b_ptr, c_ptr, bias_ptr,  #  
#                                     M: int, N: int, K: int,  #  
#                                     stride_am: int, stride_ak: int,  #  
#                                     stride_bk: int, stride_bn: int,  #  
#                                     stride_cm: int, stride_cn: int,  #  
#                                     stride_bias: int,  #  
#                                     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,  
#                                     BLOCK_SIZE_K: tl.constexpr,  #  
#                                     USE_BIAS: tl.constexpr, EVEN_K: tl.constexpr  #  
#                                     ):  
#     triton_matmul_kernel(a_ptr, b_ptr, c_ptr, bias_ptr,  #  
#                          M, N, K,  #  
#                          stride_am, stride_ak,  #  
#                          stride_bk, stride_bn,  #  
#                          stride_cm, stride_cn,  #  
#                          stride_bias,  #  
#                          BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,  #  
#                          USE_BIAS=USE_BIAS, USE_DOT=False, EVEN_K=EVEN_K)  
  
  

  

  
  
##################################################################################################################################################  

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

# PyTorch GEMM:  
# -------------  
  
  
def torch_matmul(a: Tensor, b: Tensor, bias: Optional[Tensor]) -> Tensor:  
    c: Tensor = torch.matmul(a, b)  
    if bias is not None:  
        c += bias[:, None]  
    return c  
  
  
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

# Wrapper for calling PyTorch GEMM or Triton GEMM:  
# ------------------------------------------------  
  


def triton_matmul(triton_provider: str, a: Tensor, b: Tensor, bias: Optional[Tensor]) -> Tensor:  
    assert triton_provider in ["triton-multreduce"]  
  
    M: int  
    N: int  
    K: int  
    M, K = a.shape  
    _, N = b.shape  
  
    c: Tensor = torch.empty((M, N), device=a.device, dtype=a.dtype)  
  
    def grid(args: dict[str, Any]) -> tuple[int]:  
        return (triton.cdiv(M, args["BLOCK_SIZE_M"]) * triton.cdiv(N, args["BLOCK_SIZE_N"]), )  
  
    matmult_kernel = triton_matmul_kernel #triton_multreduce_matmul_kernel  
  
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
        USE_DOT=False,  # Explicitly set for "multreduce" behavior
    )  
  
    return c  



def matmul(provider: str, a: Tensor, b: Tensor, bias: Optional[Tensor]) -> Tensor:  
    assert provider in ["torch", "triton-multreduce"]  
  
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
    assert M > 0, "M for input generation must be positive."  
    assert M <= 8, "M for input generation must be less or equal to 8."  
    assert N > 0, "N for input generation must be positive."  
    assert K > 0, "K for input generation must be positive."  
  
    set_seed()
  
    a: Tensor = torch.randn((M, K), dtype=torch.float16, device=device)  
    b: Tensor = torch.randn((N, K), dtype=a.dtype, device=a.device).T  
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
def test_matmul(M: int, N: int, K: int, use_bias: bool, request) -> None:  
    a: Tensor  
    b: Tensor  
    bias: Optional[Tensor]  
    a, b, bias = gen_input(M, N, K, use_bias)  
  
    c_torch: Tensor = matmul("torch", a, b, bias)  
    c_triton_multreduce: Tensor = matmul("triton-multreduce", a, b, bias)  

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    ################### save c_triton_dot in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = c_triton_multreduce.clone().detach().cpu()
    ###################################################################
    
    assert allclose(c_torch, c_triton_multreduce), "PyTorch and Triton Multreduce results don't match."  
  

def gen_input_for_perf(M: int, N: int, K: int, use_bias: bool, dtype_str: str = "float16", device: str = "cuda") -> tuple[Tensor, Tensor, Optional[Tensor]]:  
    if dtype_str == 'fp32': current_dtype = torch.float32
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16
    else: current_dtype = torch.float16 
    set_seed(42) 
    a: Tensor = torch.randn((M, K), dtype=current_dtype, device=device)  
    b: Tensor = torch.randn((K, N), dtype=current_dtype, device=device) 
    bias: Optional[Tensor] = torch.randn(M, dtype=current_dtype, device=device) if use_bias else None  
    return a, b, bias  

def multreduce_matmul_triton_wrapper(a_tensor, b_tensor, c_buffer, bias_tensor,
                                     M_dim, N_dim, K_dim, 
                                     block_m_const, block_n_const, block_k_const_tile, # block_k_const_tile is K-tile for kernel
                                     use_bias_flag, num_warps_launch, num_stages_launch): # num_stages for launch
    grid = (triton.cdiv(M_dim, block_m_const) * triton.cdiv(N_dim, block_n_const), )
    even_k_flag = (K_dim % block_k_const_tile == 0)

    # Call the CORE kernel directly, not the autotuned one, to avoid meta-param conflict
    triton_matmul_kernel[grid]( # Calling the non-autotuned version
        a_tensor, b_tensor, c_buffer, bias_tensor,
        M_dim, N_dim, K_dim,
        a_tensor.stride(0), a_tensor.stride(1),
        b_tensor.stride(0), b_tensor.stride(1),
        c_buffer.stride(0), c_buffer.stride(1),
        bias_tensor.stride(0) if use_bias_flag and bias_tensor is not None else 0,
        # BLOCK_SIZE_M=block_m_const, 
        # BLOCK_SIZE_N=block_n_const, 
        # BLOCK_SIZE_K=block_k_const_tile, # This is the K-tile size
        USE_BIAS=use_bias_flag, 
        USE_DOT=False, # Explicitly set for "multreduce" behavior
        EVEN_K=even_k_flag,
        # num_warps=num_warps_launch,
        # num_stages=num_stages_launch # Pass num_stages
    )
    return c_buffer

def calculate_gemm_tflops(params: dict, ms: float) -> float:
    M, N, K = params['M'], params['N'], params['K']
    flops = 2 * M * N * K 
    if params.get('use_bias', False): flops += M * N 
    tflops = flops / (ms / 1000) / 1e12
    return tflops

def get_torch_dtype_from_str(dtype_str: str, default_dtype=torch.float16) -> torch.dtype:
    if dtype_str == 'fp32': return torch.float32
    if dtype_str == 'bf16': return torch.bfloat16
    if dtype_str == 'fp16': return torch.float16
    return default_dtype

def calculate_gemm_gbps(params: dict, ms: float) -> float:
    M, N, K = params['M'], params['N'], params['K']
    use_bias = params.get('use_bias', False)
    dtype_str = params.get('dtype_str', 'fp16') 
    current_dtype = get_torch_dtype_from_str(dtype_str)
    element_size = torch.tensor([], dtype=current_dtype).element_size()
    bytes_a, bytes_b, bytes_c_write = [dim * element_size for dim in [M*K, K*N, M*N]]
    total_bytes = bytes_a + bytes_b + bytes_c_write
    if use_bias: total_bytes += M * element_size 
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

OP_NAME_FOR_BENCHMARK = "triton_multreduce_matmul_perf"

GEMM_PERF_SHAPES = [
    (512, 512, 512), (1024, 1024, 512), (2048, 1024, 256), 
]

# --- REVISED BLOCK CONFIGS focusing on BM * BN <= 4096 (for fp16 inputs, fp32 acc) ---
# And also BM * BK_tile + BK_tile * BN for the tl.load part.
GEMM_PERF_BLOCK_CONFIGS = []
# Target BM * BN <= 4096 (for fp32 acc, if 4 temp buffers of this size are needed)
# Target BM * BN <= 2048 (for fp32 acc, if 2 temp buffers of this size, plus A and B inputs)

# Small BM * BN products
for bk_tile in [32, 64]: # K-tile size
    # BM * BN approx 1024
    for bm, bn in [(32,32), (16,64), (64,16)]:
        GEMM_PERF_BLOCK_CONFIGS.append((bm,bn,bk_tile))
    # BM * BN approx 2048
    for bm, bn in [(32,64), (64,32), (16,128), (128,16)]:
        GEMM_PERF_BLOCK_CONFIGS.append((bm,bn,bk_tile))
    # BM * BN approx 4096
    for bm, bn in [(64,64), (32,128), (128,32)]:
        GEMM_PERF_BLOCK_CONFIGS.append((bm,bn,bk_tile))

# Try some original autotune-like configs for multreduce (BM=1)
for bk_tile in [128, 256, 512]:
    GEMM_PERF_BLOCK_CONFIGS.append((1, 64, bk_tile))


# Remove duplicates and ensure K_tile is not too large for shared mem of A, B load
unique_block_configs = []
seen_block_configs_str = set()
for bm, bn, bk_tile in GEMM_PERF_BLOCK_CONFIGS:
    # Check initial load shared memory: (bm * bk_tile + bk_tile * bn) * elem_size_input
    # Assume fp16 inputs (2 bytes) for this pre-filter
    smem_load_fp16_elements = bm * bk_tile + bk_tile * bn
    if smem_load_fp16_elements * 2 > 65536 * 0.8 : # Check against 80% of 64KB
        continue

    cfg_str = f"bm{bm}bn{bn}bk{bk_tile}"
    if cfg_str not in seen_block_configs_str:
        unique_block_configs.append((bm,bn,bk_tile))
        seen_block_configs_str.add(cfg_str)
GEMM_PERF_BLOCK_CONFIGS = unique_block_configs
print(f"Generated {len(GEMM_PERF_BLOCK_CONFIGS)} unique block configurations for multreduce matmul.")


GEMM_PERF_DTYPES = ['fp16'] # Start with fp16, as fp32 inputs will double shared mem for A/B
GEMM_PERF_BIAS = [False]    # Start simple
GEMM_PERF_NUM_STAGES = [1]  # Start with num_stages=1 to minimize shared memory
GEMM_PERF_NUM_WARPS = [4]   # Start with 4 warps

@pytest.mark.parametrize("m_n_k_shape", GEMM_PERF_SHAPES)
@pytest.mark.parametrize("block_config", GEMM_PERF_BLOCK_CONFIGS)
@pytest.mark.parametrize("use_bias_flag", GEMM_PERF_BIAS)
@pytest.mark.parametrize("dtype_str", GEMM_PERF_DTYPES)
@pytest.mark.parametrize("num_stages_val", GEMM_PERF_NUM_STAGES)
@pytest.mark.parametrize("num_warps_val", GEMM_PERF_NUM_WARPS)
def test_performance(m_n_k_shape, block_config, use_bias_flag, dtype_str, 
                                       num_stages_val, num_warps_val, request):
    set_seed()
    M, N, K = m_n_k_shape
    BLOCK_M_const, BLOCK_N_const, BLOCK_K_tile_const = block_config

    if K % BLOCK_K_tile_const != 0 : pytest.skip(f"K={K} not multiple of BLOCK_K_tile={BLOCK_K_tile_const}")
    if M % BLOCK_M_const !=0 : pytest.skip(f"M={M} not multiple of BLOCK_M={BLOCK_M_const}")
    
    current_dtype = get_torch_dtype_from_str(dtype_str)
    elem_size = torch.tensor([], dtype=current_dtype).element_size()
    
    # More refined shared memory check for the USE_DOT=False path
    # Acc (BM, BN) is fp32 (4 bytes)
    # A_casted (BM, BK_tile) is fp32
    # B_casted (BK_tile, BN) is fp32
    # If intermediate product P(BM, BK_tile, BN) is materialized, that's the main issue.
    # Let's assume Triton tiles the sum over K_tile.
    # The critical part might be holding one (BM,BN) slice of P in fp32, plus A and B.
    # Required fp32 elements for this slice: BM*BN
    # Required fp32 elements for A_casted tile: BM*BK_tile
    # Required fp32 elements for B_casted tile: BK_tile*BN
    # Total_fp32_elements_approx = BM*BN + BM*BK_tile + BK_tile*BN
    # This must be an underestimate if the error is 131072 bytes for BM=64,BN=128,BK=32
    # (64*128) + (64*32) + (32*128) = 8192 + 2048 + 4096 = 14336 fp32 elements.
    # 14336 * 4 bytes = 57344 bytes. This *should* fit if num_stages=1.

    # The error 131072 bytes = 32768 fp32 elements.
    # If BM=64, BN=128, then BM*BN = 8192.
    # 32768 / 8192 = 4. This implies 4 buffers of size (BM,BN) in fp32.
    # (e.g. accumulator, one slice of P, and perhaps two more for some reason).
    # So, the constraint is roughly: 4 * BLOCK_M * BLOCK_N * sizeof(fp32) <= 65536
    # BLOCK_M * BLOCK_N <= 65536 / 16 = 4096
    if BLOCK_M_const * BLOCK_N_const > 4096 :
         pytest.skip(f"Skipping BM={BLOCK_M_const}, BN={BLOCK_N_const} as BM*BN > 4096, likely OOM for USE_DOT=False path.")


    a, b, bias = gen_input_for_perf(M, N, K, use_bias_flag, dtype_str=dtype_str)
    # Output C is always fp16 or fp32 (based on input), acc is fp32. Kernel casts final acc.
    # The kernel itself casts output to c_ptr.type.element_ty.
    # Let's match input dtype for c_buffer for simplicity, kernel handles final cast.
    c_buffer = torch.empty((M, N), device='cuda', dtype=current_dtype)


    op_lambda = lambda: multreduce_matmul_triton_wrapper(
        a, b, c_buffer, bias, M, N, K,
        BLOCK_M_const, BLOCK_N_const, BLOCK_K_tile_const,
        use_bias_flag, num_warps_val, num_stages_val
    )

    bench_config = do_bench_config(warm_up=10, repetition=50)
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)
    current_params_for_logs_and_calc = {
        "M": M, "N": N, "K": K,
        "BLOCK_M": BLOCK_M_const, "BLOCK_N": BLOCK_N_const, "BLOCK_K_tile": BLOCK_K_tile_const,
        "use_bias": use_bias_flag, "dtype_str": dtype_str,
        "num_stages": num_stages_val, "num_warps": num_warps_val,
        "USE_DOT": False 
    }
    
    perf_result = benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
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
    print(f"\nSaving all c_triton_multreduce results to {OUTPUT_FILENAME}...")  
    # Ensure the directory for the output file exists if it's in a subdirectory  
    output_dir = os.path.dirname(OUTPUT_FILENAME)  
    if output_dir and not os.path.exists(output_dir):  
        os.makedirs(output_dir, exist_ok=True)  
    torch.save(result_gold, OUTPUT_FILENAME)       
    print(f"Successfully saved {len(result_gold)} c_triton_multreduce tensors to {OUTPUT_FILENAME}.")  

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
    line_vals: list[str] = ["torch", "triton-multreduce"]  
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
  
        if provider == "triton-multreduce":  
            print(f"Triton Multreduce kernel best config = {triton_matmul_kernel.best_config}")   #{triton_multreduce_matmul_kernel.best_config}")  
  
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
