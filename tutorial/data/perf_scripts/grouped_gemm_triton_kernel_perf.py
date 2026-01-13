
import torch
import triton
import triton.language as tl
import time

# Import the optimized kernel wrapper
# Adjust the import path if necessary based on where this script runs relative to the kernel file
# For the GEAK tutorial harness, we usually assume the kernel file is available or we import the class dynamically.
# Here we assume the file 'grouped_gemm_triton_kernel.py' is in the python path or same dir.
try:
    from grouped_gemm_triton_kernel import OptimizedGroupedGEMM
except ImportError:
    # If running from tutorial/data/perf_scripts/ and kernel is in tutorial/data/kernels/
    # We might need sys.path hack, but the harness usually handles this.
    # For now, we assume the harness sets up the path.
    pass

def benchmark_op():
    device = 'cuda'
    
    # ------------------------------------------------
    # 1. Setup Data
    # ------------------------------------------------
    MIN_M = 128
    MAX_M = 2048
    K = 4096
    N = 4096
    NUM_GROUPS = 8
    
    # Generate random M sizes for the groups
    torch.manual_seed(0)
    M_splits = torch.randint(MIN_M, MAX_M, (NUM_GROUPS,)).tolist()
    total_M = sum(M_splits)
    
    # Weights
    B_list = [torch.randn((K, N), device=device, dtype=torch.bfloat16) for _ in range(NUM_GROUPS)]
    
    # Inputs
    A_concat = torch.randn((total_M, K), device=device, dtype=torch.bfloat16)
    
    # ------------------------------------------------
    # 2. Setup Models
    # ------------------------------------------------
    # A) Optimized Triton FP8 Kernel
    try:
        opt_model = OptimizedGroupedGEMM(B_list, device=device)
    except NameError:
        print("Skipping benchmark: OptimizedGroupedGEMM not found.")
        return
        
    # B) PyTorch Reference (Simulated Sequential Loop)
    # Just to have a baseline (though standard torch matmul isn't strictly fair comparison to fused grouped gemm)
    
    # Warmup
    for _ in range(10):
        _ = opt_model(A_concat, M_splits)
        
    # ------------------------------------------------
    # 3. Measure Triton Performance
    # ------------------------------------------------
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(100):
        _ = opt_model(A_concat, M_splits)
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_ms / 100
    
    # ------------------------------------------------
    # 4. Report FLOPs
    # ------------------------------------------------
    # FLOPs = 2 * sum(M * N * K) for each group
    total_flops = 2 * sum([m * N * K for m in M_splits])
    tflops = (total_flops / (avg_time_ms * 1e-3)) / 1e12
    
    print(f"Average execution time: {avg_time_ms:.3f} ms")
    print(f"Throughput: {tflops:.2f} TFLOPS")

if __name__ == "__main__":
    benchmark_op()
