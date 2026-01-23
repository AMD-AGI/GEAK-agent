# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
# Optimized Grouped GEMM Kernel for GEAK-OpenEvolve
######################################## Imports #####################################
###
import torch
import triton
import triton.language as tl
from typing import List
######################################## Imports #####################################
###

# -------------------------------------------------------------------------
# Grouped GEMM Kernel (BF16, Single Config for Simplicity)
# This kernel processes multiple GEMMs in a single fused kernel launch
# -------------------------------------------------------------------------
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64

@triton.jit
def grouped_gemm_kernel(
    A_ptr, C_ptr,
    B_ptr_array,
    M_cumsum_ptr,
    M_array, N_dim, K,
    stride_ak, stride_bk, stride_bn, stride_ck,
    total_tiles_m, num_tiles_n,
    NUM_GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Grouped GEMM kernel that processes multiple GEMMs in a single fused kernel.
    
    For each group g:
      C_g = A_g @ B_g
    
    A is concatenated vertically: shape (sum(M_splits), K)
    B_g has shape (K, N) for each group
    C is concatenated vertically: shape (sum(M_splits), N)
    """
    pid_m = tl.program_id(0)  # Global tile index in M dimension
    pid_n = tl.program_id(1)  # Tile index in N dimension
    
    # Find which group this M tile belongs to
    # Using linear search through cumulative sums
    global_row_start = pid_m * BLOCK_M
    
    # Find group by checking cumulative sums
    group_id = 0
    m_offset = 0
    local_m_offset = global_row_start
    
    for g in range(NUM_GROUPS):
        curr_m = tl.load(M_array + g)
        if global_row_start >= m_offset + curr_m:
            m_offset += curr_m
            local_m_offset = global_row_start - m_offset
            group_id = g + 1
    
    # Bounds check
    if group_id >= NUM_GROUPS:
        return
        
    # Load group-specific dimensions
    M = tl.load(M_array + group_id)
    N = N_dim  # N is same for all groups
    m_start_for_group = tl.load(M_cumsum_ptr + group_id)
    
    # Local M offset within this group
    local_row_in_group = global_row_start - m_start_for_group
    
    # Check if this tile is within the group's M dimension
    if local_row_in_group >= M:
        return
    
    # Load B pointer for this group
    b_ptr = tl.load(B_ptr_array + group_id).to(tl.pointer_type(tl.bfloat16))
    
    # Initialize block offsets
    offs_am = local_row_in_group + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Compute A and B pointers
    a_ptrs = A_ptr + (m_start_for_group + offs_am[:, None]) * stride_ak + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main GEMM loop
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        
        # A mask: check M bounds (rows) and K bounds (cols)
        k_mask_a = (offs_am[:, None] < M) & (offs_k[None, :] < k_remaining)
        
        # B mask: check N bounds (cols) and K bounds (rows)
        k_mask_b = (offs_bn[None, :] < N) & (offs_k[:, None] < k_remaining)
        
        # Load tiles
        a = tl.load(a_ptrs, mask=k_mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask_b, other=0.0)
        
        # Accumulate dot product
        accumulator = tl.dot(a, b, accumulator)
        
        # Advance pointers
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_bk
    
    # Store result
    offs_cm = local_row_in_group + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + (m_start_for_group + offs_cm[:, None]) * stride_ck + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    tl.store(c_ptrs, accumulator.to(tl.bfloat16), mask=c_mask)


######################################################################################
############################################################## 
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

######################################## HELPERS for Eval ############################
############
class OptimizedGroupedGEMM:
    """
    Wrapper class for the grouped GEMM kernel.
    Manages weight tensors and kernel scheduling.
    """
    def __init__(self, B_list: List[torch.Tensor], device='cuda'):
        self.device = device
        self.num_groups = len(B_list)
        self.K = B_list[0].shape[0]
        self.N = B_list[0].shape[1]
        
        # Store weight matrices
        self.B_list = [b.contiguous() for b in B_list]
        self.B_ptr_array = torch.tensor(
            [b.data_ptr() for b in self.B_list],
            device=device, dtype=torch.int64
        )
        
    def __call__(self, A_concat: torch.Tensor, M_splits: List[int]) -> torch.Tensor:
        """Execute the grouped GEMM kernel."""
        total_M = sum(M_splits)
        
        # Compute cumulative sums for M
        M_cumsum = [0]
        for m in M_splits:
            M_cumsum.append(M_cumsum[-1] + m)
        M_cumsum_tensor = torch.tensor(M_cumsum[:-1], device=self.device, dtype=torch.int32)
        M_array = torch.tensor(M_splits, device=self.device, dtype=torch.int32)
        
        # Allocate output
        C_out = torch.empty((total_M, self.N), device=self.device, dtype=torch.bfloat16)
        
        # Compute grid dimensions
        total_tiles_m = triton.cdiv(total_M, BLOCK_M)
        num_tiles_n = triton.cdiv(self.N, BLOCK_N)
        
        # 2D grid: (tiles_in_M, tiles_in_N)
        grid = (total_tiles_m, num_tiles_n)
        
        grouped_gemm_kernel[grid](
            A_concat, C_out,
            self.B_ptr_array,
            M_cumsum_tensor,
            M_array, self.N, self.K,
            A_concat.stride(0), self.B_list[0].stride(0), self.B_list[0].stride(1), 
            C_out.stride(0),
            total_tiles_m, num_tiles_n,
            NUM_GROUPS=self.num_groups,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        
        return C_out


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
    # FLOPs = 2 * M * N * K for each GEMM
    total_flops = 2 * total_M * N * K
    return total_flops / (ms * 1e-3) / 1e12


def calculate_grouped_gemm_gbps(params: Dict, ms: float) -> float:
    """Calculate memory bandwidth in GB/s for grouped GEMM operation."""
    total_M = params['total_M']
    K = params['K']
    N = params['N']
    num_groups = params['num_groups']
    bytes_per_elem = 2  # BF16
    # Read A + all B matrices + Write C
    total_bytes = (total_M * K + num_groups * K * N + total_M * N) * bytes_per_elem
    return total_bytes / (ms * 1e-3) / 1e9


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Test configurations: (num_groups, total_M, K, N)
TEST_CONFIGS = [
    (4, 4096, 1024, 1024),   # 4 groups, 1024 rows each
    (8, 8192, 1024, 1024),   # 8 groups, 1024 rows each
]


@pytest.mark.parametrize('num_groups,total_M,K,N', TEST_CONFIGS)
def test_grouped_gemm_correctness(num_groups, total_M, K, N, request):
    """Test correctness of grouped GEMM against PyTorch reference."""
    set_seed()
    device = 'cuda'
    dtype = torch.bfloat16
    
    # Use uniform splits
    m_per_group = total_M // num_groups
    m_splits = [m_per_group] * num_groups
    
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
    
    # BF16 should be fairly accurate
    assert_close(tri_out, ref_out, rtol=1e-2, atol=1e-2, check_dtype=False)


OP_NAME_FOR_BENCHMARK = "grouped_gemm_perf"


@pytest.mark.parametrize('num_groups,total_M,K,N', TEST_CONFIGS)
def test_performance(num_groups, total_M, K, N, request):
    """Benchmark grouped GEMM performance."""
    set_seed()
    device = 'cuda'
    dtype = torch.bfloat16
    
    # Use uniform splits
    m_per_group = total_M // num_groups
    m_splits = [m_per_group] * num_groups
    
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
    }
    
    benchmarker.run_benchmark(
        current_params_dict=current_params,
        gbps_calculator=calculate_grouped_gemm_gbps,
        tflops_calculator=calculate_grouped_gemm_tflops
    )


######################################## HELPERS for Eval ############################
############
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


######################################## HELPERS for Eval ############################
############
