# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
# Optimized Fused Grouped FP8 GEMM Kernel for GEAK-OpenEvolve
######################################## Imports ########################################
import torch
import triton
import triton.language as tl
from typing import List
######################################## Imports ########################################


# -------------------------------------------------------------------------
# Optimized Fused Grouped FP8 GEMM Kernel with Autotuning
# -------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
    ],
    key=['total_tiles', 'K'],
)
@triton.jit
def grouped_fused_fp8_gemm_kernel_v2(
    A_ptr, C_ptr,
    B_ptr_array,
    M_cumsum_ptr,
    M_array, N_array, K,
    scales_ptr,
    tile_to_group_ptr, tile_offset_ptr,
    stride_ak, stride_bk, stride_bn, stride_ck,
    total_tiles,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    """
    Grouped FP8 GEMM kernel that processes multiple GEMMs in a single fused kernel.
    
    For each group g:
      C_g = A_g @ B_g (with FP8 quantization and dequantization)
    
    A is concatenated vertically: shape (sum(M_splits), K)
    B_g has shape (K, N) for each group
    C is concatenated vertically: shape (sum(M_splits), N)
    """
    pid = tl.program_id(0)
    
    # Load group mapping for this tile
    group_id = tl.load(tile_to_group_ptr + pid)
    local_tile_id = tl.load(tile_offset_ptr + pid)
    
    # Load group-specific dimensions
    M = tl.load(M_array + group_id)
    N = tl.load(N_array + group_id)
    m_offset = tl.load(M_cumsum_ptr + group_id)
    
    # Compute tile position within this group using L2-cache-friendly tiling
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id_tile = local_tile_id // num_pid_in_group
    first_pid_m = group_id_tile * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (local_tile_id % group_size_m)
    pid_n = (local_tile_id % num_pid_in_group) // group_size_m
    
    # Load FP8 scales for this group (qscale_a, qscale_b, dscale_a, dscale_b)
    scale_base = group_id * 4
    qscale_a = tl.load(scales_ptr + scale_base + 0)
    qscale_b = tl.load(scales_ptr + scale_base + 1)
    dscale_a = tl.load(scales_ptr + scale_base + 2)
    dscale_b = tl.load(scales_ptr + scale_base + 3)
    
    # Load B pointer for this group
    b_ptr = tl.load(B_ptr_array + group_id).to(tl.pointer_type(tl.bfloat16))
    
    # Initialize block offsets
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Compute A and B pointers
    a_ptrs = A_ptr + (m_offset + offs_am[:, None]) * stride_ak + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main GEMM loop with FP8 quantization
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        
        # A mask: check M bounds (rows) and K bounds (cols)
        k_mask_a = (offs_am[:, None] < M) & (offs_k[None, :] < k_remaining)
        
        # B mask: check N bounds (cols) and K bounds (rows)
        k_mask_b = (offs_bn[None, :] < N) & (offs_k[:, None] < k_remaining)
        
        # Load and quantize to FP8
        a_bf16 = tl.load(a_ptrs, mask=k_mask_a, other=0.0)
        b_bf16 = tl.load(b_ptrs, mask=k_mask_b, other=0.0)
        
        a_fp8 = (a_bf16 * qscale_a).to(tl.float8e4b8)
        b_fp8 = (b_bf16 * qscale_b).to(tl.float8e4b8)
        
        # Accumulate dot product
        accumulator = tl.dot(a_fp8, b_fp8, accumulator)
        
        # Advance pointers
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_bk
    
    # Apply dequantization scales
    total_dscale = dscale_a * dscale_b
    c = accumulator * total_dscale
    
    # Store result
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + (m_offset + offs_cm[:, None]) * stride_ck + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    tl.store(c_ptrs, c.to(tl.bfloat16), mask=c_mask)


####################################################################################################################################################

# ==================================================================================
# SECTION 3: Benchmarking & Testing Code (pytest)
# ==================================================================================
import numpy as np
import random
import torch 
import os
from numpy.random import RandomState
import pytest
from torch.testing import assert_close
from geak_eval.perf.ROCm.performance_utils_pytest import PytestBenchmarker, do_bench_config, save_all_benchmark_results
from typing import Dict, List

import triton
import triton.language as tl

result_gold = {}

######################################## HELPERS for Eval ########################################

def get_fp8_dtype():
    """Return appropriate FP8 dtype based on platform (CUDA vs ROCm)."""
    if torch.version.hip is not None:
        return torch.float8_e4m3fnuz
    else:
        return torch.float8_e4m3fn


class OptimizedGroupedGEMM:
    """
    Wrapper class for the grouped FP8 GEMM kernel.
    Manages weight tensors, FP8 scales, and kernel scheduling.
    """
    def __init__(self, B_list: List[torch.Tensor], device='cuda'):
        self.device = device
        self.num_groups = len(B_list)
        self.K = B_list[0].shape[0]
        self.N = B_list[0].shape[1]
        self.FP8_MAX = 240.0
        
        # Store weight matrices
        self.B_list = [b.contiguous() for b in B_list]
        self.B_ptr_array = torch.tensor(
            [b.data_ptr() for b in self.B_list],
            device=device, dtype=torch.int64
        )
        
        # Precompute FP8 scales for weights
        self.B_qscales = torch.empty(self.num_groups, device=device, dtype=torch.float32)
        self.B_dscales = torch.empty(self.num_groups, device=device, dtype=torch.float32)
        for i, B in enumerate(self.B_list):
            max_b = B.abs().max().float().item()
            max_b = max(max_b, 1e-6)
            self.B_qscales[i] = self.FP8_MAX / max_b
            self.B_dscales[i] = max_b / self.FP8_MAX
        
        # Allocate scales buffer (4 scales per group: qscale_a, qscale_b, dscale_a, dscale_b)
        self.scales_buffer = torch.empty(self.num_groups * 4, device=device, dtype=torch.float32)
        for i in range(self.num_groups):
            self.scales_buffer[i*4 + 1] = self.B_qscales[i]
            self.scales_buffer[i*4 + 3] = self.B_dscales[i]
        
        # Cache for schedule and output
        self._cached_m_splits = None
        self._cached_schedule = None
        self._cached_C_out = None
        
    def _compute_schedule(self, M_splits: List[int], BLOCK_M=128, BLOCK_N=128):
        """Compute tile-to-group mapping for the kernel launch."""
        M_cumsum = [0]
        tile_to_group = []
        tile_offset = []
        M_array = []
        N_array = []
        
        for group_id, M in enumerate(M_splits):
            M_cumsum.append(M_cumsum[-1] + M)
            M_array.append(M)
            N_array.append(self.N)
            
            num_tiles_m = triton.cdiv(M, BLOCK_M)
            num_tiles_n = triton.cdiv(self.N, BLOCK_N)
            
            for local_id in range(num_tiles_m * num_tiles_n):
                tile_to_group.append(group_id)
                tile_offset.append(local_id)
        
        return {
            'M_cumsum': torch.tensor(M_cumsum[:-1], device=self.device, dtype=torch.int32),
            'M_array': torch.tensor(M_array, device=self.device, dtype=torch.int32),
            'N_array': torch.tensor(N_array, device=self.device, dtype=torch.int32),
            'tile_to_group': torch.tensor(tile_to_group, device=self.device, dtype=torch.int32),
            'tile_offset': torch.tensor(tile_offset, device=self.device, dtype=torch.int32),
            'total_tiles': len(tile_to_group),
            'total_M': sum(M_splits),
        }
    
    def _get_schedule(self, M_splits: List[int]):
        """Get or compute scheduling information."""
        m_tuple = tuple(M_splits)
        if self._cached_m_splits != m_tuple:
            self._cached_schedule = self._compute_schedule(M_splits)
            self._cached_m_splits = m_tuple
            total_M = self._cached_schedule['total_M']
            self._cached_C_out = torch.empty((total_M, self.N), device=self.device, dtype=torch.bfloat16)
        return self._cached_schedule
    
    def __call__(self, A_concat: torch.Tensor, M_splits: List[int]) -> torch.Tensor:
        """Execute the grouped GEMM kernel."""
        schedule = self._get_schedule(M_splits)
        
        # Compute FP8 scales for input activations
        start = 0
        for i, m in enumerate(M_splits):
            max_a = A_concat[start:start+m].abs().max().float().item()
            max_a = max(max_a, 1e-6)
            qscale_a = self.FP8_MAX / max_a
            self.scales_buffer[i*4 + 0] = qscale_a
            self.scales_buffer[i*4 + 2] = 1.0 / qscale_a
            start += m
        
        # Launch kernel
        grid = (schedule['total_tiles'],)
        
        grouped_fused_fp8_gemm_kernel_v2[grid](
            A_concat, self._cached_C_out,
            self.B_ptr_array,
            schedule['M_cumsum'],
            schedule['M_array'], schedule['N_array'], self.K,
            self.scales_buffer,
            schedule['tile_to_group'], schedule['tile_offset'],
            A_concat.stride(0), self.B_list[0].stride(0), self.B_list[0].stride(1), 
            self._cached_C_out.stride(0),
            schedule['total_tiles'],
            NUM_GROUPS=self.num_groups,
        )
        
        return self._cached_C_out


def reference_grouped_gemm(A_concat: torch.Tensor, B_list: List[torch.Tensor], M_splits: List[int]) -> torch.Tensor:
    """Reference implementation using PyTorch for correctness checking."""
    outputs = []
    start = 0
    for i, m in enumerate(M_splits):
        A_group = A_concat[start:start+m]
        C_group = torch.matmul(A_group.float(), B_list[i].float()).to(torch.bfloat16)
        outputs.append(C_group)
        start += m
    return torch.cat(outputs, dim=0)


# Helper function to define TFLOPS for grouped_gemm
def calculate_grouped_gemm_tflops(params: Dict, ms: float) -> float:
    """Calculate TFLOPS for grouped GEMM operation."""
    total_M = params['total_M']
    K = params['K']
    N = params['N']
    # FLOPs = 2 * M * N * K for matrix multiplication
    flops = 2 * total_M * N * K
    tflops = flops / (ms / 1000) / 1e12
    return tflops


# Helper function to define GB/s for grouped_gemm
def calculate_grouped_gemm_gbps(params: Dict, ms: float) -> float:
    """Calculate GB/s for grouped GEMM operation."""
    total_M = params['total_M']
    K = params['K']
    N = params['N']
    num_groups = params['num_groups']
    # Bytes: read A (total_M * K * 2), read B (num_groups * K * N * 2), write C (total_M * N * 2)
    bytes_per_element = 2  # bfloat16
    total_bytes = (total_M * K + num_groups * K * N + total_M * N) * bytes_per_element
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps


def set_seed(seed: int = 42) -> None:
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

######################################## HELPERS for Eval ########################################


# Test configurations: (num_groups, total_M, K, N, zipf_exponent)
TEST_CONFIGS = [
    (12, 208896, 3840, 3840, 0.5),   # Uniform-ish distribution
    (12, 208896, 3840, 3840, 1.0),   # Moderately skewed
    (12, 208896, 3840, 3840, 1.5),   # Highly skewed
]


def generate_m_splits(num_groups: int, total_M: int, zipf_exponent: float, alignment: int = 256):
    """Generate M splits for each group using Zipf distribution."""
    ranks = torch.arange(1, num_groups + 1, dtype=torch.float32).cuda()
    input_splits = (1.0 / torch.pow(ranks, zipf_exponent))
    input_splits = (input_splits * 1000).int() + 1
    input_splits = (input_splits.float() / input_splits.sum() * total_M).int()
    
    # Align to 256
    input_splits = ((input_splits + alignment - 1) // alignment) * alignment
    
    # Adjust to match total_M exactly
    current_sum = input_splits.sum()
    if current_sum != total_M:
        diff = total_M - current_sum
        if diff > 0:
            num_blocks_to_add = (diff + alignment - 1) // alignment
            sorted_indices = torch.argsort(input_splits, descending=True)
            for i in range(num_blocks_to_add):
                input_splits[sorted_indices[i % num_groups]] += alignment
        else:
            num_blocks_to_remove = (-diff + alignment - 1) // alignment
            sorted_indices = torch.argsort(input_splits, descending=True)
            for i in range(num_blocks_to_remove):
                idx = sorted_indices[i % num_groups]
                if input_splits[idx] > alignment:
                    input_splits[idx] -= alignment
        input_splits[-1] += total_M - input_splits.sum()
    
    return input_splits.tolist()


@pytest.mark.parametrize('num_groups,total_M,K,N,zipf_exp', TEST_CONFIGS)
def test_grouped_gemm_correctness(num_groups, total_M, K, N, zipf_exp, request):
    """Test correctness of grouped GEMM against PyTorch reference."""
    set_seed()
    device = 'cuda'
    dtype = torch.bfloat16
    
    # Generate M splits
    m_splits = generate_m_splits(num_groups, total_M, zipf_exp)
    
    # Create weight matrices for each group
    B_list = [torch.randn(K, N, device=device, dtype=dtype) for _ in range(num_groups)]
    
    # Create input activations
    A_concat = torch.randn(total_M, K, device=device, dtype=dtype)
    
    # Initialize Triton kernel
    triton_gemm = OptimizedGroupedGEMM(B_list, device=device)
    
    # Run Triton kernel
    tri_out = triton_gemm(A_concat, m_splits)
    
    # Run reference
    ref_out = reference_grouped_gemm(A_concat, B_list, m_splits)
    
    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    # Save output for golden comparison
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = tri_out.clone().detach().cpu()
    
    # FP8 quantization introduces some error, use relaxed tolerances
    assert_close(tri_out, ref_out, rtol=0.1, atol=0.5, check_dtype=False)


OP_NAME_FOR_BENCHMARK = "grouped_gemm_fp8_perf"


@pytest.mark.parametrize('num_groups,total_M,K,N,zipf_exp', TEST_CONFIGS)
def test_performance(num_groups, total_M, K, N, zipf_exp, request):
    """Benchmark grouped GEMM performance."""
    set_seed()
    device = 'cuda'
    dtype = torch.bfloat16
    
    # Generate M splits
    m_splits = generate_m_splits(num_groups, total_M, zipf_exp)
    
    # Create weight matrices for each group
    B_list = [torch.randn(K, N, device=device, dtype=dtype) for _ in range(num_groups)]
    
    # Create input activations
    A_concat = torch.randn(total_M, K, device=device, dtype=dtype)
    
    # Initialize Triton kernel
    triton_gemm = OptimizedGroupedGEMM(B_list, device=device)
    
    # Warmup
    for _ in range(5):
        _ = triton_gemm(A_concat, m_splits)
    torch.cuda.synchronize()
    
    # Define benchmark lambda
    op_lambda = lambda: triton_gemm(A_concat, m_splits)
    
    bench_config = do_bench_config(warm_up=25, repetition=100)
    benchmarker = PytestBenchmarker(
        op_callable=op_lambda,
        op_name=OP_NAME_FOR_BENCHMARK,
        config=bench_config
    )
    
    current_params = {
        "num_groups": num_groups,
        "total_M": total_M,
        "K": K,
        "N": N,
        "zipf_exp": zipf_exp,
    }
    
    benchmarker.run_benchmark(
        current_params_dict=current_params,
        gbps_calculator=calculate_grouped_gemm_gbps,
        tflops_calculator=calculate_grouped_gemm_tflops
    )


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
    print(f"\nSaving all y_triton results to {OUTPUT_FILENAME}...")  
    output_dir = os.path.dirname(OUTPUT_FILENAME)  
    if output_dir and not os.path.exists(output_dir):  
        os.makedirs(output_dir, exist_ok=True)  
    torch.save(result_gold, OUTPUT_FILENAME)       
    print(f"Successfully saved {len(result_gold)} y_triton tensors to {OUTPUT_FILENAME}.")  


def test_save_performance_results():
    """
    Called after the test_performance function finishes.
    This is a separate hook to ensure performance results are saved.
    """
    print('\nPytest session finishing... Saving benchmark results...')

    output_directory = os.path.join(os.path.dirname(__file__), "perf")
    os.makedirs(output_directory, exist_ok=True)
    
    save_all_benchmark_results(output_directory)
    print(f"All benchmark results attempted to save to: {output_directory}")


######################################## HELPERS for Eval ########################################
