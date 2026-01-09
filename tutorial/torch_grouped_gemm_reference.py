import torch
import triton
import triton.language as tl
import transformer_engine.pytorch as te
from transformer_engine.pytorch.module.grouped_linear import GroupedLinear
from transformer_engine.common.recipe import Float8CurrentScaling, Format
from typing import List
import math


def get_fp8_dtype():
    """Return appropriate FP8 dtype based on platform (CUDA vs ROCm)."""
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


def benchmark_grouped_gemm_fp8():
    groups = 12
    input_dim = 208896
    hidden_dim = 3840

    device = "cuda"
    params_dtype = torch.bfloat16

    mod = GroupedLinear(
        num_gemms=groups,
        in_features=hidden_dim,
        out_features=hidden_dim,
        bias=False,
        device=device,
        params_dtype=params_dtype,
    )

    triton_B_list = []
    with torch.no_grad():
        for i in range(groups):
            w_te = getattr(mod, f"weight{i}")
            w_triton = w_te.t().contiguous()
            triton_B_list.append(w_triton)

    triton_gemm = OptimizedGroupedGEMM(triton_B_list, device=device)

    fp8_recipe = Float8CurrentScaling(fp8_format=Format.E4M3)
    total_flops = input_dim * hidden_dim * hidden_dim * 2
    peak_tflops = 2614.9 if torch.version.hip is not None else 1979.0
    print(f"Peak FP8 TFLOPS: {peak_tflops}")

    num_split_configs = 50
    results = []

    print(f"\nRunning Benchmark over {num_split_configs} configs...")
    print(f"{'Config':<8} {'Std':<10} {'TE(ms)':<10} {'Tri(ms)':<10} {'TE(TF)':<10} {'Tri(TF)':<10} {'Speedup':<10}")

    for config_idx in range(num_split_configs):
        zipf_exponent = 0.5 + (config_idx / num_split_configs) * 1.0
        ranks = torch.arange(1, groups + 1, dtype=torch.float32).cuda()
        input_splits = (1.0 / torch.pow(ranks, zipf_exponent))
        input_splits = (input_splits * 1000).int() + 1
        input_splits = (input_splits.float() / input_splits.sum() * input_dim).int()
        
        alignment = 256
        input_splits = ((input_splits + alignment - 1) // alignment) * alignment
        
        current_sum = input_splits.sum()
        if current_sum != input_dim:
            diff = input_dim - current_sum
            if diff > 0:
                num_blocks_to_add = (diff + alignment - 1) // alignment
                sorted_indices = torch.argsort(input_splits, descending=True)
                for i in range(num_blocks_to_add):
                    input_splits[sorted_indices[i % groups]] += alignment
            else:
                num_blocks_to_remove = (-diff + alignment - 1) // alignment
                sorted_indices = torch.argsort(input_splits, descending=True)
                for i in range(num_blocks_to_remove):
                    idx = sorted_indices[i % groups]
                    if input_splits[idx] > alignment:
                        input_splits[idx] -= alignment
            input_splits[-1] += input_dim - input_splits.sum()  
        # Ensure all splits are still aligned and positive
        assert (input_splits % alignment == 0).all(), "Not all splits are 256-aligned"
        assert input_splits.min() > 0, "Some splits are non-positive"

        print(f"Config {config_idx+1}/{num_split_configs}: min={input_splits.min().item()}, max={input_splits.max().item()}")

        # Calculate standard deviation of input splits
        splits_std = torch.std(input_splits.float()).item()
        assert input_splits.sum().item() == input_dim, "Input splits do not sum to input_dim"

        # Convert splits to list for GroupedLinear
        m_splits = input_splits.tolist()

        # Warmup with FP8
        for _ in range(5):
            x = torch.randn((input_dim, hidden_dim), device=device, dtype=params_dtype)
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                output = mod(x, m_splits)
            _ = triton_gemm(x, m_splits)
        torch.cuda.synchronize()

        num_iters = 20
        
        # TE Benchmark
        te_times = []
        for _ in range(num_iters):
            x = torch.randn((input_dim, hidden_dim), device=device, dtype=params_dtype)
            x = x.contiguous()
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                _ = mod(x, m_splits)
            end_event.record()
            torch.cuda.synchronize()
            te_times.append(start_event.elapsed_time(end_event))
            
        te_time = sum(te_times) / len(te_times)

        # Triton Benchmark
        tri_times = []
        for _ in range(num_iters):
            x = torch.randn((input_dim, hidden_dim), device=device, dtype=params_dtype)
            x = x.contiguous()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            _ = triton_gemm(x, m_splits)
            end_event.record()
            torch.cuda.synchronize()
            tri_times.append(start_event.elapsed_time(end_event))
            
        tri_time = sum(tri_times) / len(tri_times)

        te_tflops = (total_flops / (te_time / 1000.0)) / 1e12
        tri_tflops = (total_flops / (tri_time / 1000.0)) / 1e12
        speedup = te_time / tri_time

        print(f"{config_idx:<8} {splits_std:<10.1f} {te_time:<10.3f} {tri_time:<10.3f} {te_tflops:<10.2f} {tri_tflops:<10.2f} {speedup:<10.2f}")
        
        results.append({
            'std': splits_std,
            'te_time': te_time,
            'tri_time': tri_time,
            'te_tflops': te_tflops,
            'tri_tflops': tri_tflops,
            'speedup': speedup,
        })

    sorted_results = sorted(results, key=lambda x: x['std'])
    print("\n" + "=" * 90)
    print("Summary Statistics:")
    print("=" * 90)
    te_times = [r['te_time'] for r in sorted_results]
    tri_times = [r['tri_time'] for r in sorted_results]
    te_tflops_list = [r['te_tflops'] for r in sorted_results]
    tri_tflops_list = [r['tri_tflops'] for r in sorted_results]
    speedups = [r['speedup'] for r in sorted_results]
    
    print(f"TE Time (ms):     min={min(te_times):.3f}, max={max(te_times):.3f}, avg={sum(te_times)/len(te_times):.3f}")
    print(f"Triton Time (ms): min={min(tri_times):.3f}, max={max(tri_times):.3f}, avg={sum(tri_times)/len(tri_times):.3f}")
    print(f"TE TFLOPS:        avg={sum(te_tflops_list)/len(te_tflops_list):.2f}")
    print(f"Triton TFLOPS:    avg={sum(tri_tflops_list)/len(tri_tflops_list):.2f}")
    print(f"Speedup (TE/Tri): min={min(speedups):.2f}, max={max(speedups):.2f}, avg={sum(speedups)/len(speedups):.2f}")


if __name__ == "__main__":
    benchmark_grouped_gemm_fp8()
