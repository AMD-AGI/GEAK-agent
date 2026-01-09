import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch.module.grouped_linear import GroupedLinear
from transformer_engine.common.recipe import Float8CurrentScaling, Format


def get_fp8_dtype():
    """Return appropriate FP8 dtype based on platform (CUDA vs ROCm)."""
    if torch.version.hip is not None:
        return torch.float8_e4m3fnuz
    else:
        return torch.float8_e4m3fn


def benchmark_grouped_gemm_fp8():
    groups = 12
    input_dim = 208896
    hidden_dim = 3840

    device = "cuda"
    # Use bfloat16 as the params dtype; FP8 conversion happens internally via fp8_autocast
    params_dtype = torch.bfloat16

    # Create TransformerEngine GroupedLinear module
    mod = GroupedLinear(
        num_gemms=groups,
        in_features=hidden_dim,
        out_features=hidden_dim,
        bias=False,
        device=device,
        params_dtype=params_dtype,
    )

    # FP8 recipe using current scaling (per-tensor dynamic scaling)
    fp8_recipe = Float8CurrentScaling(fp8_format=Format.E4M3)

    # Calculate total FLOPs (constant across all configurations)
    # Total FLOPs = input_dim × hidden_dim × hidden_dim × 2 (multiply-add)
    total_flops = input_dim * hidden_dim * hidden_dim * 2
    
    # FP8 peak TFLOPS (adjust based on your GPU - this is for H100 FP8)
    # H100 SXM5: ~1979 TFLOPS FP8 Tensor Core
    # MI300 and MI325: 2614.9 TFLOPS FP8 per GPU.
    peak_tflops = 2614.9 if torch.version.hip is not None else 1979.0
    print(f"Peak FP8 TFLOPS: {peak_tflops}")

    # Store results across different split distributions
    num_split_configs = 50
    results = []

    for config_idx in range(num_split_configs):
        # Create Zipf-like distribution for input splits
        # Vary the exponent to get different levels of skew
        zipf_exponent = 0.5 + (config_idx / num_split_configs) * 1.0  # Range from 0.5 to 1.5
        ranks = torch.arange(1, groups + 1, dtype=torch.float32).cuda()
        input_splits = (1.0 / torch.pow(ranks, zipf_exponent))
        
        # Scale and convert to integers
        input_splits = (input_splits * 1000).int() + 1  # Add 1 to ensure minimum of 1

        # Normalize to sum to input_dim
        input_splits = (input_splits.float() / input_splits.sum() * input_dim).int()
        
        # Align each split to 256 elements (required for grouped GEMM efficiency)
        alignment = 256
        input_splits = ((input_splits + alignment - 1) // alignment) * alignment
        
        # Adjust to ensure exact sum equals input_dim
        current_sum = input_splits.sum()
        if current_sum != input_dim:
            diff = input_dim - current_sum
            # Distribute the difference to the largest splits
            if diff > 0:
                # Need to add more - add alignment blocks to largest splits
                num_blocks_to_add = (diff + alignment - 1) // alignment
                sorted_indices = torch.argsort(input_splits, descending=True)
                for i in range(num_blocks_to_add):
                    input_splits[sorted_indices[i % groups]] += alignment
            else:
                # Need to remove - remove alignment blocks from largest splits
                num_blocks_to_remove = (-diff + alignment - 1) // alignment
                sorted_indices = torch.argsort(input_splits, descending=True)
                for i in range(num_blocks_to_remove):
                    idx = sorted_indices[i % groups]
                    if input_splits[idx] > alignment:  # Ensure we don't go below minimum
                        input_splits[idx] -= alignment
            
            # Final adjustment: add/subtract the remainder to/from the last element
            input_splits[-1] += input_dim - input_splits.sum()

        # Ensure all splits are still aligned and positive
        assert (input_splits % alignment == 0).all(), "Not all splits are 256-aligned"
        assert input_splits.min() > 0, "Some splits are non-positive"

        print(f"Config {config_idx+1}/{num_split_configs}: min={input_splits.min().item()}, max={input_splits.max().item()}")

        # Calculate standard deviation of input splits
        splits_std = torch.std(input_splits.float()).item()
        assert input_splits.sum().item() == input_dim, "Input splits do not sum to input_dim"

        # Convert splits to list for GroupedLinear
        m_splits = input_splits.tolist()

        # Warmup with FP8
        for _ in range(5):
            x = torch.randn((input_dim, hidden_dim), device=device, dtype=params_dtype)
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                output = mod(x, m_splits)
        torch.cuda.synchronize()

        # Timed benchmark (forward pass only)
        num_iters = 20
        gemm_times = []

        for _ in range(num_iters):
            x = torch.randn((input_dim, hidden_dim), device=device, dtype=params_dtype)
            x = x.contiguous()  # Ensure proper strides
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                output = mod(x, m_splits)
            end_event.record()
            torch.cuda.synchronize()
            gemm_times.append(start_event.elapsed_time(end_event))

        # Calculate mean timing
        mean_gemm = sum(gemm_times) / len(gemm_times)
        
        # Calculate MFU (Model FLOPs Utilization)
        time_seconds = mean_gemm / 1000.0  # Convert ms to seconds
        achieved_tflops = (total_flops / time_seconds) / 1e12
        mfu_percent = (achieved_tflops / peak_tflops) * 100.0
        
        # Store results
        results.append({
            'std': splits_std,
            'gemm_mean': mean_gemm,
            'achieved_tflops': achieved_tflops,
            'mfu_percent': mfu_percent,
        })

    # Sort results by std and print
    sorted_results = sorted(results, key=lambda x: x['std'])
    print("\n" + "=" * 80)
    print("FP8 Grouped GEMM Results (sorted by standard deviation):")
    print("=" * 80)
    print(f"{'Rank':<6} {'Std':<12} {'Time(ms)':<12} {'TFLOPS':<12} {'MFU%':<10}")
    print("=" * 80)
    for i, result in enumerate(sorted_results):
        print(f"{i+1:<6} {result['std']:<12.2f} {result['gemm_mean']:<12.3f} "
              f"{result['achieved_tflops']:<12.2f} {result['mfu_percent']:<10.2f}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics:")
    print("=" * 80)
    times = [r['gemm_mean'] for r in sorted_results]
    tflops = [r['achieved_tflops'] for r in sorted_results]
    mfus = [r['mfu_percent'] for r in sorted_results]
    
    print(f"Time (ms):    min={min(times):.3f}, max={max(times):.3f}, avg={sum(times)/len(times):.3f}")
    print(f"TFLOPS:       min={min(tflops):.2f}, max={max(tflops):.2f}, avg={sum(tflops)/len(tflops):.2f}")
    print(f"MFU%:         min={min(mfus):.2f}, max={max(mfus):.2f}, avg={sum(mfus)/len(mfus):.2f}")


if __name__ == "__main__":
    benchmark_grouped_gemm_fp8()
