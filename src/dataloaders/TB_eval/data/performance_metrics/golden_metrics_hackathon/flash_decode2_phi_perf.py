import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.flash_decode2_phi import flash_decode_stage2
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('flash_decode2_phi', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        for i in range(2, 18):  # Adjust the range as needed for your testing
            batch_size = 2 ** i
            head_num = 8  # Example head number, adjust as needed
            seq_block_num = 16  # Example sequence block number, adjust as needed
            head_dim = 64  # Example head dimension, adjust as needed

            mid_out = torch.rand(batch_size, head_num, seq_block_num, head_dim, dtype=torch.float32)
            mid_out_logexpsum = torch.rand(batch_size, head_num, seq_block_num, dtype=torch.float32)
            B_Seqlen = torch.randint(1, seq_block_num * 32, (batch_size,), dtype=torch.int32)
            Out = torch.empty(batch_size, head_num, head_dim, dtype=torch.float32)

            self.input_tensors.append((mid_out, mid_out_logexpsum, B_Seqlen, Out))

    def to_cuda(self, input_tensor):
        mid_out, mid_out_logexpsum, B_Seqlen, Out = input_tensor
        return (mid_out.cuda(), mid_out_logexpsum.cuda(), B_Seqlen.cuda(), Out.cuda())

    def call_op(self, input_tensor):
        mid_out, mid_out_logexpsum, B_Seqlen, Out = input_tensor
        block_seq = 32  # Example block sequence size, adjust as needed
        flash_decode_stage2(mid_out, mid_out_logexpsum, B_Seqlen, Out, block_seq)
        return Out

    def get_gbps(self, input_tensor, runtime):
        mid_out, mid_out_logexpsum, B_Seqlen, Out = input_tensor
        total_bytes = mid_out.numel() * mid_out.element_size() + mid_out_logexpsum.numel() * mid_out_logexpsum.element_size() + B_Seqlen.numel() * B_Seqlen.element_size() + Out.numel() * Out.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        mid_out, _, _, _ = input_tensor
        FLOPS = 2 * mid_out.numel()  # Example calculation, adjust based on actual operations
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
