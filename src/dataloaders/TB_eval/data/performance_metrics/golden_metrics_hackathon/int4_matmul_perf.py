import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.int4_matmul import matmul_dequantize_int4_s2, quantize_int4
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('int4_matmul', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        for i in range(2, 12):
            M = 128 * i
            K = 128 * i
            N = 128 * i
            x = torch.rand((M, K), dtype=torch.float32)
            weight = torch.rand((K, N), dtype=torch.float32)
            qweight, scales, qzeros, group_size = quantize_int4(weight)
            self.input_tensors.append((x, qweight, scales, qzeros, group_size))

    def to_cuda(self, input_tensor):
        x, qweight, scales, qzeros, group_size = input_tensor
        return (x.cuda(), qweight.cuda(), scales.cuda(), qzeros.cuda(), group_size)

    def call_op(self, input_tensor):
        x, qweight, scales, qzeros, group_size = input_tensor
        return matmul_dequantize_int4_s2(x, qweight, scales, qzeros, group_size)

    def get_gbps(self, input_tensor, runtime):
        x, qweight, scales, qzeros, group_size = input_tensor
        total_bytes = x.numel() * x.element_size() + qweight.numel() * qweight.element_size() + scales.numel() * scales.element_size() + qzeros.numel() * qzeros.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        x, qweight, scales, qzeros, group_size = input_tensor
        M, K = x.shape
        N = scales.shape[1]
        K_effective = K * 4
        FLOPS = 2 * M * N * K_effective  # Assuming a standard matrix multiplication
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
