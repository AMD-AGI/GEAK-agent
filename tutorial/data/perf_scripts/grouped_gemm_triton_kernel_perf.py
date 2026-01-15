# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

import sys
import os
import importlib.util

# Add current directory to path for generated kernel import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Import generated kernel (will be copied to same dir as this script)
from grouped_gemm_triton_kernel import OptimizedGroupedGEMM

# Import reference kernel from kernels directory using importlib to avoid cache
KERNELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'kernels'))
ref_spec = importlib.util.spec_from_file_location(
    "grouped_gemm_triton_kernel_ref",
    os.path.join(KERNELS_DIR, "grouped_gemm_triton_kernel.py")
)
ref_module = importlib.util.module_from_spec(ref_spec)
ref_spec.loader.exec_module(ref_module)
OptimizedGroupedGEMM_ref = ref_module.OptimizedGroupedGEMM

from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=torch.bfloat16, is_backward=False, **kwargs):
        super().__init__('grouped_gemm_triton_kernel', dtype=dtype, is_backward=is_backward, **kwargs)
        self.dtype = dtype
        
    def get_input_tensors(self):
        """Generate test inputs with varying group sizes."""
        self.input_tensors = []
        
        # Different configurations: (NUM_GROUPS, K, N, MIN_M, MAX_M)
        configs = [
            (4, 1024, 1024, 128, 512),
            (8, 2048, 2048, 256, 1024),
            (4, 4096, 4096, 512, 2048),
        ]
        
        for num_groups, K, N, min_m, max_m in configs:
            torch.manual_seed(0)
            M_splits = torch.randint(min_m, max_m, (num_groups,)).tolist()
            total_M = sum(M_splits)
            
            # Create weight matrices for each group
            B_list = [torch.randn((K, N), dtype=self.dtype) for _ in range(num_groups)]
            
            # Create input
            A_concat = torch.randn((total_M, K), dtype=self.dtype)
            
            # Store as tuple: (A_concat, M_splits, B_list)
            input_tensor = (A_concat, M_splits, B_list)
            self.input_tensors.append(input_tensor)

    def to_cuda(self, input_tensor):
        """Move tensors to GPU."""
        A_concat, M_splits, B_list = input_tensor
        A_cuda = A_concat.cuda()
        B_cuda = [b.cuda() for b in B_list]
        return (A_cuda, M_splits, B_cuda)

    def call_op(self, input_tensor):
        """Call generated kernel."""
        A_concat, M_splits, B_list = input_tensor
        model = OptimizedGroupedGEMM(B_list, device='cuda')
        return model(A_concat, M_splits)
    
    def call_op_ref(self, input_tensor):
        """Call reference kernel."""
        A_concat, M_splits, B_list = input_tensor
        model_ref = OptimizedGroupedGEMM_ref(B_list, device='cuda')
        return model_ref(A_concat, M_splits)

    def get_gbps(self, input_tensor, runtime):
        """Calculate memory bandwidth."""
        A_concat, M_splits, B_list = input_tensor
        K = B_list[0].shape[0]
        N = B_list[0].shape[1]
        
        # Bytes read: A + B, Bytes written: C
        total_M = sum(M_splits)
        bytes_A = total_M * K * A_concat.element_size()
        bytes_B = sum([K * N * b.element_size() for b in B_list])
        bytes_C = total_M * N * A_concat.element_size()
        total_bytes = bytes_A + bytes_B + bytes_C
        
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        """Calculate TFLOPS."""
        A_concat, M_splits, B_list = input_tensor
        K = B_list[0].shape[0]
        N = B_list[0].shape[1]
        
        # FLOPs = 2 * M * N * K for each group
        total_flops = 2 * sum([m * N * K for m in M_splits])
        TFLOPS = total_flops / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=25, rep=100)
    op_perf.run_benchmark()
