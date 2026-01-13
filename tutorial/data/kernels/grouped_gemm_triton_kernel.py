
import torch
import triton
import triton.language as tl
import transformer_engine.pytorch as te
from transformer_engine.pytorch.module.grouped_linear import GroupedLinear
from transformer_engine.common.recipe import Float8CurrentScaling, Format
from typing import List
import math

def get_fp8_dtype():
    # Return appropriate FP8 dtype based on platform (CUDA vs ROCm).
    if torch.version.hip is not None:
        return torch.float8_e4m3fnuz
    else:
        return torch.float8_e4m3fn

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
    pid = tl.program_id(0)
    
    group_id = tl.load(tile_to_group_ptr + pid)
    local_tile_id = tl.load(tile_offset_ptr + pid)
    
    M = tl.load(M_array + group_id)
    N = tl.load(N_array + group_id)
    m_offset = tl.load(M_cumsum_ptr + group_id)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id_tile = local_tile_id // num_pid_in_group
    first_pid_m = group_id_tile * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (local_tile_id % group_size_m)
    pid_n = (local_tile_id % num_pid_in_group) // group_size_m
    
    scale_base = group_id * 4
    qscale_a = tl.load(scales_ptr + scale_base + 0)
    qscale_b = tl.load(scales_ptr + scale_base + 1)
    dscale_a = tl.load(scales_ptr + scale_base + 2)
    dscale_b = tl.load(scales_ptr + scale_base + 3)
    
    b_ptr = tl.load(B_ptr_array + group_id).to(tl.pointer_type(tl.bfloat16))
    
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = A_ptr + (m_offset + offs_am[:, None]) * stride_ak + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        
        # --- M/N/K Masking ---
        # A mask: check M bounds (rows) and K bounds (cols)
        k_mask_a = (offs_am[:, None] < M) & (offs_k[None, :] < k_remaining)
        
        # B mask: check N bounds (cols) and K bounds (rows)
        k_mask_b = (offs_bn[None, :] < N) & (offs_k[:, None] < k_remaining)
        
        a_bf16 = tl.load(a_ptrs, mask=k_mask_a, other=0.0)
        b_bf16 = tl.load(b_ptrs, mask=k_mask_b, other=0.0)
        
        a_fp8 = (a_bf16 * qscale_a).to(tl.float8e4b8)
        b_fp8 = (b_bf16 * qscale_b).to(tl.float8e4b8)
        
        accumulator = tl.dot(a_fp8, b_fp8, accumulator)
        
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_bk
    
    total_dscale = dscale_a * dscale_b
    c = accumulator * total_dscale
    
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + (m_offset + offs_cm[:, None]) * stride_ck + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    tl.store(c_ptrs, c.to(tl.bfloat16), mask=c_mask)


class OptimizedGroupedGEMM:
    def __init__(self, B_list: List[torch.Tensor], device='cuda'):
        self.device = device
        self.num_groups = len(B_list)
        self.K = B_list[0].shape[0]
        self.N = B_list[0].shape[1]
        self.FP8_MAX = 240.0
        
        self.B_list = [b.contiguous() for b in B_list]
        self.B_ptr_array = torch.tensor(
            [b.data_ptr() for b in self.B_list],
            device=device, dtype=torch.int64
        )
        
        self.B_qscales = torch.empty(self.num_groups, device=device, dtype=torch.float32)
        self.B_dscales = torch.empty(self.num_groups, device=device, dtype=torch.float32)
        for i, B in enumerate(self.B_list):
            max_b = B.abs().max().float().item()
            max_b = max(max_b, 1e-6)
            self.B_qscales[i] = self.FP8_MAX / max_b
            self.B_dscales[i] = max_b / self.FP8_MAX
        
        self.scales_buffer = torch.empty(self.num_groups * 4, device=device, dtype=torch.float32)
        for i in range(self.num_groups):
            self.scales_buffer[i*4 + 1] = self.B_qscales[i]
            self.scales_buffer[i*4 + 3] = self.B_dscales[i]
        
        self._cached_m_splits = None
        self._cached_schedule = None
        self._cached_C_out = None
        
    def _compute_schedule(self, M_splits: List[int], BLOCK_M=128, BLOCK_N=128):
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
        m_tuple = tuple(M_splits)
        if self._cached_m_splits != m_tuple:
            self._cached_schedule = self._compute_schedule(M_splits)
            self._cached_m_splits = m_tuple
            total_M = self._cached_schedule['total_M']
            self._cached_C_out = torch.empty((total_M, self.N), device=self.device, dtype=torch.bfloat16)
        return self._cached_schedule
    
    def __call__(self, A_concat: torch.Tensor, M_splits: List[int]) -> torch.Tensor:
        schedule = self._get_schedule(M_splits)
        
        start = 0
        for i, m in enumerate(M_splits):
            max_a = A_concat[start:start+m].abs().max().float().item()
            max_a = max(max_a, 1e-6)
            qscale_a = self.FP8_MAX / max_a
            self.scales_buffer[i*4 + 0] = qscale_a
            self.scales_buffer[i*4 + 2] = 1.0 / qscale_a
            start += m
        
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

# -------------------------------------------------------------------------
# Correctness Testing Support (Test Delimiter)
# -------------------------------------------------------------------------

# The agent harness expects a test_function to trigger execution
def test_grouped_gemm():
    # Setup
    torch.manual_seed(123)
    NUM_GROUPS = 4
    K = 1024
    N = 1024
    # Fixed sizes for simplicity in basic test
    M_splits = [128, 256, 64, 512]
    
    device = 'cuda'
    
    # Init Weights (B) - List of [K, N] matrices
    # We use same size K, N for all for this specific kernel variant
    B_list = []
    for _ in range(NUM_GROUPS):
        B = torch.randn((K, N), device=device, dtype=torch.bfloat16)
        B_list.append(B)
        
    # Init Inputs (A)
    # We concat them into one big tensor for the fused kernel
    A_list = []
    for m in M_splits:
        A = torch.randn((m, K), device=device, dtype=torch.bfloat16)
        A_list.append(A)
    
    A_concat = torch.cat(A_list, dim=0)
    
    # 1. Run Optimized Kernel
    opt_model = OptimizedGroupedGEMM(B_list, device=device)
    C_triton = opt_model(A_concat, M_splits)
    
    # 2. Run Torch Reference (BF16 matmul)
    C_refs = []
    for i, (A, B) in enumerate(zip(A_list, B_list)):
        C_refs.append(torch.matmul(A, B))
    C_ref = torch.cat(C_refs, dim=0)
    
    # 3. Compare
    # Note: FP8 lossy, so we use higher tolerance
    diff = torch.abs(C_triton - C_ref)
    max_diff = diff.max().item()
    print(f"Max Diff (FP8 vs BF16): {max_diff}")
    
    # Tolerance for FP8 vs BF16 is loose
    assert max_diff < 1.0, f"Difference too high: {max_diff}"
    print("Test Passed!")

if __name__ == "__main__":
    test_grouped_gemm()
