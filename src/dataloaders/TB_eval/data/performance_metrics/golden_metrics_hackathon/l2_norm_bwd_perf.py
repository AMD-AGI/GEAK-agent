import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.l2_norm_bwd import _l2_norm_bwd
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('l2_norm_bwd', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(4, 15):
            size = 2 ** i
            x = torch.rand(size, dtype=torch.float32)
            dy = torch.rand(size, dtype=torch.float32)
            self.input_tensors.append((x, dy))

    def to_cuda(self, input_tensor):
        x, dy = input_tensor
        return x.cuda(), dy.cuda()

    def call_op(self, input_tensor):
        x, dy = input_tensor
        return _l2_norm_bwd(x, dy)

    def get_gbps(self, input_tensor, runtime):
        x, dy = input_tensor
        total_bytes = (x.numel() + dy.numel() + x.numel()) * x.element_size()  # x, dy, and dx
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        x, dy = input_tensor
        # Assuming each element involves a few FLOPs, e.g., multiplication, addition
        FLOPS = 2 * x.numel()  # Simplified estimation
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
