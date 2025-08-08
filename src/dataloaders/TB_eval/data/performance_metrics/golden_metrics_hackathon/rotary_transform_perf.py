import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.rotary_transform import apply_rotary
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('rotary_transform', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 16):  # Choose a reasonable range for testing
            batch_size = 2 ** i
            seqlen = 128  # Fixed sequence length
            nheads = 8  # Number of attention heads
            headdim = 64  # Dimension of each head
            rotary_dim = 32  # Rotary dimension
            x = torch.rand(batch_size, seqlen, nheads, headdim, dtype=torch.float32)
            cos = torch.rand(seqlen, rotary_dim // 2, dtype=torch.float32)
            sin = torch.rand(seqlen, rotary_dim // 2, dtype=torch.float32)
            self.input_tensors.append((x, cos, sin))

    def to_cuda(self, input_tensor):
        x, cos, sin = input_tensor
        return (x.cuda(), cos.cuda(), sin.cuda())

    def call_op(self, input_tensor):
        x, cos, sin = input_tensor
        return apply_rotary(x, cos, sin)

    def get_gbps(self, input_tensor, runtime):
        x, cos, sin = input_tensor
        total_bytes = x.numel() * x.element_size() + cos.numel() * cos.element_size() + sin.numel() * sin.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        x, cos, sin = input_tensor
        # Assuming each element in x is involved in a few operations (e.g., multiply and add)
        FLOPS = 2 * x.numel()  # Simplified estimation
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
