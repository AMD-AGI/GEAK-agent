import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.matrix_vector_multip import mv
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('matrix_vector_multip', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 20):  # Adjust the range for different sizes
            M = 128 * i
            N = 128 * (i - 1)  # Example: N is half of M
            matrix = torch.rand((N, M), dtype=torch.float32)
            vector = torch.rand((M,), dtype=torch.float32)
            self.input_tensors.append((matrix, vector))

    def to_cuda(self, input_tensor):
        matrix, vector = input_tensor
        return (matrix.cuda(), vector.cuda())

    def call_op(self, input_tensor):
        matrix, vector = input_tensor
        return mv(matrix, vector)

    def get_gbps(self, input_tensor, runtime):
        matrix, vector = input_tensor
        total_bytes = (matrix.numel() + vector.numel() + matrix.size(0)) * matrix.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        matrix, vector = input_tensor
        N, M = matrix.shape
        FLOPS = 2 * N * M  # Each element in the output involves M multiplications and M-1 additions
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
