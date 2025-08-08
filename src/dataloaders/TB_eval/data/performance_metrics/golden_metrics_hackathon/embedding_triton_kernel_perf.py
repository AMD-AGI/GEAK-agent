import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TritonBench_v1.embedding_triton_kernel import embedding
from performance_utils import Performance_Metrics, do_bench_config

import torch
import triton
import triton.language as tl

class performance_metrics(Performance_Metrics):
    def __init__(self, dtype=None, is_backward=False, **kwargs):
        super().__init__('embedding_triton_kernel', dtype=dtype, is_backward=is_backward, **kwargs)
        
    def get_input_tensors(self):
        self.input_tensors = []
        vocab_size = 10000  # Example vocabulary size
        hidden_size = 768   # Example hidden size
        for i in range(2, 18):  # Example range for input sizes
            seq_length = 2 ** i
            input_ids = torch.randint(0, vocab_size, (seq_length,), dtype=torch.int32)
            weight = torch.rand(vocab_size, hidden_size, dtype=torch.float16)
            out = torch.zeros(seq_length, hidden_size, dtype=torch.float16)
            vob_start_id = 0
            vob_end_id = vocab_size
            self.input_tensors.append((input_ids, weight, vob_start_id, vob_end_id, out))

    def to_cuda(self, input_tensor):
        input_ids, weight, vob_start_id, vob_end_id, out = input_tensor
        return (input_ids.cuda(), weight.cuda(), vob_start_id, vob_end_id, out.cuda())

    def call_op(self, input_tensor):
        input_ids, weight, vob_start_id, vob_end_id, out = input_tensor
        embedding(input_ids, weight, vob_start_id, vob_end_id, out)
        return out

    def get_gbps(self, input_tensor, runtime):
        input_ids, weight, _, _, _ = input_tensor
        total_bytes = input_ids.numel() * input_ids.element_size() + weight.numel() * weight.element_size()
        GBPS = total_bytes / (runtime / 1000) / 1e9
        return GBPS
    
    def get_tflops(self, input_tensor, runtime):
        input_ids, weight, _, _, _ = input_tensor
        FLOPS = 2 * input_ids.numel() * weight.shape[1]  # Assuming 2 FLOPS per element (load and store)
        TFLOPS = FLOPS / (runtime / 1000) / 1e12
        return TFLOPS

if __name__ == '__main__':
    op_perf = performance_metrics()
    op_perf.get_input_tensors()
    op_perf.get_do_bench_config(warmup=100, rep=1000)
    op_perf.run_benchmark()
