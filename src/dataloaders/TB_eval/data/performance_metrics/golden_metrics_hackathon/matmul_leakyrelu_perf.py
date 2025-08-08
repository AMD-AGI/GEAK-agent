import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.matmul_leakyrelu import matmul
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('matmul_leakyrelu', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 33):  # Adjust the range for different sizes
            M = N = K = 128 * i
            a = torch.rand((M, K), dtype=torch.float16)
            b = torch.rand((K, N), dtype=torch.float16)
            self.input_tensors.append((a, b))

    def to_cuda(self, input_tensor):
        a, b = input_tensor
        return a.cuda(), b.cuda()

    def call_op(self, input_tensor):
        a, b = input_tensor
        return matmul(a, b, activation="leaky_relu")

    def get_gbps(self, input_tensor, runtime):
        a, b = input_tensor
        M, K = a.shape
        K, N = b.shape
        total_bytes = (a.numel() + b.numel() + M * N) * a.element_size()  # A, B, and C
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        a, b = input_tensor
        M, K = a.shape
        K, N = b.shape
        FLOPS = 2 * M * N * K + 2 * M * N
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
