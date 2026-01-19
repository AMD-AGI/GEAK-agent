#!/usr/bin/env python3
"""
Verify baseline numbers match spreadsheet for BS64.

Spreadsheet shows for topk & sort (BS64):
- Torch copy 1: 3.96 us
- Torch copy 2: 3.96 us  
- HIP grouped_topk: 4.22 us
- Triton append: 3.96 us
- CK P0_v2: 3.97 us
- CK P23: 4.71 us
- Total: ~24.78 us
"""

import subprocess
from pathlib import Path
import shutil

WORKSPACE = Path("/home/sapmajum/kernel_optimization_framework/profiler/verify_ws")
if WORKSPACE.exists():
    shutil.rmtree(WORKSPACE, ignore_errors=True)
WORKSPACE.mkdir(exist_ok=True)

VERIFY_SCRIPT = '''#!/usr/bin/env python3
"""Verify baseline matches spreadsheet numbers for BS64."""
import torch
import triton
import triton.language as tl
import numpy as np
import os
os.environ['TRITON_PRINT_AUTOTUNING'] = '0'

from aiter import biased_grouped_topk, moe_sorting_fwd

device = "cuda"

print("=" * 80)
print("BASELINE VERIFICATION - Comparing to Spreadsheet")
print("=" * 80)

print("\\nSPREADSHEET REFERENCE (BS64):")
print("  Torch copy 1:      3.96 us")
print("  Torch copy 2:      3.96 us")
print("  HIP grouped_topk:  4.22 us")
print("  Triton append:     3.96 us")
print("  CK P0_v2:          3.97 us")
print("  CK P23:            4.71 us")
print("  TOTAL:             24.78 us")

for M in [4, 64]:
    print(f"\\n{'='*80}")
    print(f"TESTING BATCH SIZE = {M}")
    print("=" * 80)
    
    K, S = 8, 1
    NUM_EXPERTS = 256
    TOTAL_EXPERTS = NUM_EXPERTS + S
    NUM_GROUPS = 8
    TOPK_GROUP = 4
    UNIT_SIZE = 64
    MAX_PAD = M * (K+S) + TOTAL_EXPERTS * UNIT_SIZE
    
    @triton.jit
    def append_kernel(ids_in, w_in, ids_out, w_out, num_experts, scale, num_tokens, K: tl.constexpr, S: tl.constexpr):
        pid = tl.program_id(0)
        if pid >= num_tokens:
            return
        in_off, out_off = pid * K, pid * (K + S)
        offs = tl.arange(0, K)
        tl.store(ids_out + out_off + offs, tl.load(ids_in + in_off + offs))
        tl.store(w_out + out_off + offs, tl.load(w_in + in_off + offs))
        tl.store(ids_out + out_off + K, num_experts)
        tl.store(w_out + out_off + K, scale)
    
    torch.manual_seed(42)
    gating = torch.randn(M, NUM_EXPERTS, dtype=torch.bfloat16, device=device)
    bias = torch.randn(NUM_EXPERTS, dtype=torch.bfloat16, device=device)
    
    topk_w = torch.empty((M, K), dtype=torch.float32, device=device)
    topk_ids = torch.empty((M, K), dtype=torch.int32, device=device)
    ids_out = torch.empty((M, K+S), dtype=torch.int32, device=device)
    w_out = torch.empty((M, K+S), dtype=torch.float32, device=device)
    sorted_ids = torch.empty(MAX_PAD, dtype=torch.int32, device=device)
    sorted_w = torch.empty(MAX_PAD, dtype=torch.float32, device=device)
    sorted_exp = torch.empty(MAX_PAD, dtype=torch.int32, device=device)
    num_valid = torch.empty(1, dtype=torch.int32, device=device)
    moe_buf = torch.empty(TOTAL_EXPERTS + 1, dtype=torch.int32, device=device)
    
    def full_pipeline():
        biased_grouped_topk(gating, bias.to(gating.dtype), topk_w, topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
        append_kernel[(M,)](topk_ids, topk_w, ids_out, w_out, NUM_EXPERTS, 1.0, M, K, S)
        num_valid.zero_()
        moe_sorting_fwd(ids_out, w_out, sorted_ids, sorted_w, sorted_exp, num_valid, moe_buf, TOTAL_EXPERTS, UNIT_SIZE)
    
    # Heavy warmup
    print(f"  Warming up (10000 iterations)...")
    for _ in range(10000):
        full_pipeline()
    torch.cuda.synchronize()
    
    # Profile each component
    print(f"  Profiling individual components...")
    
    # HIP topk
    hip_times = []
    for _ in range(5000):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        biased_grouped_topk(gating, bias.to(gating.dtype), topk_w, topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
        end.record()
        torch.cuda.synchronize()
        hip_times.append(start.elapsed_time(end) * 1000)
    
    # Triton append
    triton_times = []
    for _ in range(5000):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        append_kernel[(M,)](topk_ids, topk_w, ids_out, w_out, NUM_EXPERTS, 1.0, M, K, S)
        end.record()
        torch.cuda.synchronize()
        triton_times.append(start.elapsed_time(end) * 1000)
    
    # CK sorting (both phases)
    ck_times = []
    for _ in range(5000):
        num_valid.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        moe_sorting_fwd(ids_out, w_out, sorted_ids, sorted_w, sorted_exp, num_valid, moe_buf, TOTAL_EXPERTS, UNIT_SIZE)
        end.record()
        torch.cuda.synchronize()
        ck_times.append(start.elapsed_time(end) * 1000)
    
    # Full pipeline
    full_times = []
    for _ in range(5000):
        num_valid.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        full_pipeline()
        end.record()
        torch.cuda.synchronize()
        full_times.append(start.elapsed_time(end) * 1000)
    
    hip_mean = np.mean(hip_times)
    triton_mean = np.mean(triton_times)
    ck_mean = np.mean(ck_times)
    full_mean = np.mean(full_times)
    
    print(f"\\n  MEASURED RESULTS (BS={M}):")
    print(f"  -----------------------------------")
    print(f"  HIP grouped_topk:     {hip_mean:.2f} us")
    print(f"  Triton append:        {triton_mean:.2f} us")
    print(f"  CK sorting (P0+P23):  {ck_mean:.2f} us")
    print(f"  -----------------------------------")
    print(f"  Component sum:        {hip_mean + triton_mean + ck_mean:.2f} us")
    print(f"  Full pipeline:        {full_mean:.2f} us")
    
    if M == 64:
        print(f"\\n  COMPARISON TO SPREADSHEET (BS64):")
        print(f"  -----------------------------------")
        spreadsheet_hip = 4.22
        spreadsheet_triton = 3.96
        spreadsheet_ck = 3.97 + 4.71  # P0 + P23
        spreadsheet_total = 24.78  # Includes Torch copies
        
        print(f"  HIP:    Measured={hip_mean:.2f} vs Spreadsheet={spreadsheet_hip:.2f}  (diff={hip_mean-spreadsheet_hip:+.2f})")
        print(f"  Triton: Measured={triton_mean:.2f} vs Spreadsheet={spreadsheet_triton:.2f}  (diff={triton_mean-spreadsheet_triton:+.2f})")
        print(f"  CK:     Measured={ck_mean:.2f} vs Spreadsheet={spreadsheet_ck:.2f}  (diff={ck_mean-spreadsheet_ck:+.2f})")
        
        # Note: Spreadsheet includes 2x Torch copy kernels (3.96 + 3.96 = 7.92 us)
        measured_no_torch = hip_mean + triton_mean + ck_mean
        spreadsheet_no_torch = spreadsheet_hip + spreadsheet_triton + spreadsheet_ck
        
        print(f"\\n  Without Torch copies:")
        print(f"  Measured: {measured_no_torch:.2f} us")
        print(f"  Spreadsheet: {spreadsheet_no_torch:.2f} us")
        print(f"  Difference: {measured_no_torch - spreadsheet_no_torch:+.2f} us")

print("\\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
'''


def main():
    print("=" * 80)
    print("VERIFYING BASELINE AGAINST SPREADSHEET")
    print("=" * 80)
    
    script_path = WORKSPACE / "verify.py"
    script_path.write_text(VERIFY_SCRIPT)
    
    cmd = [
        "docker", "run", "--rm",
        "--device=/dev/kfd", "--device=/dev/dri", "--ipc=host",
        "-e", "HIP_VISIBLE_DEVICES=3",
        "-v", f"{WORKSPACE}:/workspace",
        "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x",
        "python3", "/workspace/verify.py"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])


if __name__ == "__main__":
    main()
