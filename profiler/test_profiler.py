#!/usr/bin/env python3
"""
Test the ROCm Profiler with sample Triton, HIP, and CK kernels.

This script demonstrates how to use the profiler utility for different
kernel types and how to interpret the results.
"""

import subprocess
from pathlib import Path
import json
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

WORKSPACE = Path("/home/sapmajum/kernel_optimization_framework/profiler/test_workspace")
WORKSPACE.mkdir(exist_ok=True)


def create_triton_vector_add_test() -> str:
    """Create a simple Triton vector add kernel test."""
    return '''#!/usr/bin/env python3
"""Test: Triton Vector Add Kernel"""
import torch
import triton
import triton.language as tl
import numpy as np
import json
import os
os.environ['TRITON_PRINT_AUTOTUNING'] = '0'

device = "cuda"

@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

# Test configuration
N = 1024 * 1024  # 1M elements
BLOCK = 1024

# Allocate tensors
x = torch.randn(N, device=device, dtype=torch.float32)
y = torch.randn(N, device=device, dtype=torch.float32)
output = torch.empty_like(x)

# Grid
grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

# Warmup
print("Warming up...")
for _ in range(100):
    vector_add_kernel[grid](x, y, output, N, BLOCK_SIZE=BLOCK)
torch.cuda.synchronize()

# Benchmark
print("Benchmarking...")
times = []
for _ in range(500):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    vector_add_kernel[grid](x, y, output, N, BLOCK_SIZE=BLOCK)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) * 1000)  # us

# Calculate metrics
mean_us = np.mean(times)
std_us = np.std(times)
bytes_transferred = 3 * N * 4  # 3 vectors * N elements * 4 bytes
flops = N  # N additions
bandwidth_gbps = bytes_transferred / (mean_us * 1e3)  # GB/s
arithmetic_intensity = flops / bytes_transferred  # FLOP/byte

results = {
    "kernel_name": "vector_add_triton",
    "kernel_type": "triton",
    "mean_us": mean_us,
    "std_us": std_us,
    "n_elements": N,
    "bytes_transferred": bytes_transferred,
    "flops": flops,
    "bandwidth_gbps": bandwidth_gbps,
    "arithmetic_intensity": arithmetic_intensity,
    "grid_size": triton.cdiv(N, BLOCK),
    "block_size": BLOCK,
}

print(f"\\nResults:")
print(f"  Latency: {mean_us:.2f} ± {std_us:.2f} us")
print(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")
print(f"  AI: {arithmetic_intensity:.4f} FLOP/byte")

with open("/workspace/triton_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\\nResults saved to triton_results.json")
'''


def create_triton_matmul_test() -> str:
    """Create a Triton matrix multiplication kernel test."""
    return '''#!/usr/bin/env python3
"""Test: Triton Matrix Multiplication Kernel"""
import torch
import triton
import triton.language as tl
import numpy as np
import json
import os
os.environ['TRITON_PRINT_AUTOTUNING'] = '0'

device = "cuda"

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a, b.T)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)

# Test configuration
M, N, K = 1024, 1024, 1024
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

# Allocate tensors
a = torch.randn((M, K), device=device, dtype=torch.float32)
b = torch.randn((K, N), device=device, dtype=torch.float32)
c = torch.empty((M, N), device=device, dtype=torch.float32)

# Grid
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

# Warmup
print("Warming up...")
for _ in range(50):
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
torch.cuda.synchronize()

# Benchmark
print("Benchmarking...")
times = []
for _ in range(200):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) * 1000)  # us

# Calculate metrics
mean_us = np.mean(times)
std_us = np.std(times)
flops = 2 * M * N * K  # 2*M*N*K FLOPs for matmul
bytes_transferred = (M*K + K*N + M*N) * 4  # A, B, C matrices
tflops = (flops / mean_us) / 1e6
bandwidth_gbps = bytes_transferred / (mean_us * 1e3)
arithmetic_intensity = flops / bytes_transferred

results = {
    "kernel_name": "matmul_triton",
    "kernel_type": "triton",
    "mean_us": mean_us,
    "std_us": std_us,
    "M": M, "N": N, "K": K,
    "bytes_transferred": bytes_transferred,
    "flops": flops,
    "tflops": tflops,
    "bandwidth_gbps": bandwidth_gbps,
    "arithmetic_intensity": arithmetic_intensity,
    "grid_size": grid,
    "block_size": (BLOCK_M, BLOCK_N, BLOCK_K),
}

print(f"\\nResults:")
print(f"  Latency: {mean_us:.2f} ± {std_us:.2f} us")
print(f"  TFLOP/s: {tflops:.2f}")
print(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")
print(f"  AI: {arithmetic_intensity:.2f} FLOP/byte")

with open("/workspace/matmul_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\\nResults saved to matmul_results.json")
'''


def create_topk_sort_test() -> str:
    """Create a test for the topk & sort pipeline (HIP + Triton + CK)."""
    return '''#!/usr/bin/env python3
"""Test: TopK & Sort Pipeline (HIP + Triton + CK kernels)"""
import torch
import triton
import triton.language as tl
import numpy as np
import json
import os
os.environ['TRITON_PRINT_AUTOTUNING'] = '0'

device = "cuda"

# Import aiter kernels
from aiter import biased_grouped_topk, moe_sorting_fwd

# Configuration
M = 4  # Batch size
K = 8  # TopK
S = 1  # Shared experts
NUM_EXPERTS = 256
TOTAL_EXPERTS = NUM_EXPERTS + S
NUM_GROUPS = 8
TOPK_GROUP = 4
UNIT_SIZE = 64
MAX_PAD = M * (K+S) + TOTAL_EXPERTS * UNIT_SIZE

print("=" * 70)
print("TOPK & SORT PIPELINE PROFILING")
print("=" * 70)

# Triton append kernel
@triton.jit
def fast_append(ids_in, w_in, ids_out, w_out, num_experts, scale, K: tl.constexpr, S: tl.constexpr, M: tl.constexpr):
    for token in range(M):
        in_off = token * K
        out_off = token * (K + S)
        offs = tl.arange(0, K)
        ids = tl.load(ids_in + in_off + offs)
        ws = tl.load(w_in + in_off + offs)
        tl.store(ids_out + out_off + offs, ids)
        tl.store(w_out + out_off + offs, ws)
        tl.store(ids_out + out_off + K, num_experts)
        tl.store(w_out + out_off + K, scale)

# Allocate tensors
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

def run_pipeline():
    biased_grouped_topk(gating, bias.to(gating.dtype), topk_w, topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    fast_append[(1,)](topk_ids, topk_w, ids_out, w_out, NUM_EXPERTS, 1.0, K, S, M)
    moe_sorting_fwd(ids_out, w_out, sorted_ids, sorted_w, sorted_exp, num_valid, moe_buf, TOTAL_EXPERTS, UNIT_SIZE)

# Warmup
print("Warming up (500 iterations)...")
for _ in range(500):
    run_pipeline()
torch.cuda.synchronize()

# Benchmark each component separately
print("\\nProfiling components...")

results = {"components": {}}

# 1. HIP biased_grouped_topk
hip_times = []
for _ in range(200):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    biased_grouped_topk(gating, bias.to(gating.dtype), topk_w, topk_ids, NUM_GROUPS, TOPK_GROUP, True, 1.0)
    end.record()
    torch.cuda.synchronize()
    hip_times.append(start.elapsed_time(end) * 1000)
results["components"]["hip_topk"] = {"mean_us": np.mean(hip_times), "std_us": np.std(hip_times), "type": "hip"}

# 2. Triton fast_append
triton_times = []
for _ in range(200):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fast_append[(1,)](topk_ids, topk_w, ids_out, w_out, NUM_EXPERTS, 1.0, K, S, M)
    end.record()
    torch.cuda.synchronize()
    triton_times.append(start.elapsed_time(end) * 1000)
results["components"]["triton_append"] = {"mean_us": np.mean(triton_times), "std_us": np.std(triton_times), "type": "triton"}

# 3. CK moe_sorting
ck_times = []
for _ in range(200):
    num_valid.zero_()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    moe_sorting_fwd(ids_out, w_out, sorted_ids, sorted_w, sorted_exp, num_valid, moe_buf, TOTAL_EXPERTS, UNIT_SIZE)
    end.record()
    torch.cuda.synchronize()
    ck_times.append(start.elapsed_time(end) * 1000)
results["components"]["ck_sorting"] = {"mean_us": np.mean(ck_times), "std_us": np.std(ck_times), "type": "ck"}

# 4. Full pipeline
full_times = []
for _ in range(200):
    num_valid.zero_()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    run_pipeline()
    end.record()
    torch.cuda.synchronize()
    full_times.append(start.elapsed_time(end) * 1000)
results["full_pipeline"] = {"mean_us": np.mean(full_times), "std_us": np.std(full_times)}

# Summary
print("\\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"\\n{'Component':<25} {'Type':<10} {'Latency (us)':<20}")
print("-" * 60)
for name, data in results["components"].items():
    print(f"{name:<25} {data['type']:<10} {data['mean_us']:.2f} ± {data['std_us']:.2f}")
print("-" * 60)
print(f"{'Full Pipeline':<25} {'mixed':<10} {results['full_pipeline']['mean_us']:.2f} ± {results['full_pipeline']['std_us']:.2f}")

# Calculate percentages
total = results['full_pipeline']['mean_us']
print("\\nComponent breakdown:")
for name, data in results["components"].items():
    pct = data['mean_us'] / total * 100
    print(f"  {name}: {pct:.1f}%")

# Save results
with open("/workspace/topk_sort_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\\nResults saved to topk_sort_results.json")
'''


def run_test(name: str, script: str) -> dict:
    """Run a test script in Docker and return results."""
    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print("="*70)
    
    # Write script
    script_path = WORKSPACE / f"{name}.py"
    script_path.write_text(script)
    
    # Run in Docker
    cmd = [
        "docker", "run", "--rm",
        "--device=/dev/kfd", "--device=/dev/dri", "--ipc=host",
        "-e", "HIP_VISIBLE_DEVICES=3",
        "-v", f"{WORKSPACE}:/workspace",
        "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x",
        "python3", f"/workspace/{name}.py"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-1000:])
    
    # Check for results
    results = {}
    for json_file in WORKSPACE.glob("*.json"):
        with open(json_file) as f:
            results[json_file.stem] = json.load(f)
            
    return results


def main():
    print("="*70)
    print("ROCm Profiler Test Suite")
    print("="*70)
    
    all_results = {}
    
    # Test 1: Triton vector add (memory-bound)
    all_results.update(run_test("triton_vec_add", create_triton_vector_add_test()))
    
    # Test 2: Triton matmul (compute-bound)
    all_results.update(run_test("triton_matmul", create_triton_matmul_test()))
    
    # Test 3: TopK & Sort pipeline (mixed HIP + Triton + CK)
    all_results.update(run_test("topk_sort", create_topk_sort_test()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, data in all_results.items():
        print(f"\n{name}:")
        if isinstance(data, dict):
            if "mean_us" in data:
                print(f"  Latency: {data['mean_us']:.2f} us")
            if "tflops" in data:
                print(f"  TFLOP/s: {data['tflops']:.2f}")
            if "bandwidth_gbps" in data:
                print(f"  Bandwidth: {data['bandwidth_gbps']:.2f} GB/s")
            if "arithmetic_intensity" in data:
                print(f"  AI: {data['arithmetic_intensity']:.4f} FLOP/byte")
            if "components" in data:
                print("  Components:")
                for cname, cdata in data["components"].items():
                    print(f"    {cname}: {cdata['mean_us']:.2f} us ({cdata['type']})")
    
    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    results = main()

