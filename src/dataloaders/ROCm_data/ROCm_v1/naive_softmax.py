# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
#Imports 
import argparse
import torch
import sys
import pytest

import triton
import triton.language as tl


@triton.jit
def softmax_kernel_naive(in_ptr, output_ptr, row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    in_max = -float('inf')
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_range = tl.arange(0, BLOCK_SIZE)
        col_mask = col_range + offset < n_cols
        in_data = tl.load(in_ptr + pid * row_stride + col_range + offset, mask=col_mask, other=-float('inf'))
        in_max = tl.maximum(in_max, tl.max(in_data, axis=-1))
    
    in_exp_sum = 0.0
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_range = tl.arange(0, BLOCK_SIZE)
        col_mask = col_range + offset < n_cols
        in_data = tl.load(in_ptr + pid * row_stride + col_range + offset, mask=col_mask, other=-float('inf'))
        in_exp_sum = in_exp_sum + tl.sum(tl.exp(in_data - in_max), axis=-1)
    
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_range = tl.arange(0, BLOCK_SIZE)
        col_mask = col_range + offset < n_cols
        in_data = tl.load(in_ptr + pid * row_stride + col_range + offset, mask=col_mask)
        in_exp = tl.exp(in_data - in_max)
        tl.store(output_ptr + pid * row_stride + col_range + offset, in_exp / in_exp_sum, mask=col_mask)



##################################################################################################################################################  


import numpy as np
import random
import torch 
import os
from tb_eval.perf.ROCm.performance_utils_pytest import PytestBenchmarker, do_bench_config, save_all_benchmark_results
from typing import Dict


result_gold = {}

######################################## HELPERS for Eval ######################################## 
CONFIG = {
  "llama3": {
    "8B": {
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "hidden_size": 4096,
      "intermediate_size": 14336,
      "vocab_size": 128256
    },
    "70B": {
      "num_attention_heads": 64,
      "num_key_value_heads": 8,
      "hidden_size": 8192,
      "intermediate_size": 28672,
      "vocab_size": 128256
    },
    "405B": {
      "num_attention_heads": 128,
      "num_key_value_heads": 8,
      "hidden_size": 16384,
      "intermediate_size": 53248,
      "vocab_size": 128256
    }
  },
  "mistral": {
    "7B": {
      "hidden_size": 4096,
      "intermediate_size": 14336,
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "vocab_size": 32000
    },
    "22B": {
      "hidden_size": 6144,
      "intermediate_size": 16384,
      "num_attention_heads": 48,
      "num_key_value_heads": 8,
      "vocab_size": 32000
    }

  }
}

  
def get_model_configs(config_path='model_configs.json', model_families=["llama3"], model="all"):  
    """  
    Load model names from the configuration file.  
  
    Args:  
        config_path (str): User-provided path to the configuration JSON file.  
        model_families (list): List of model family names to retrieve.  
  
    Returns:  
        dict: A dictionary of available models and their configurations for the specified families.  
    """  
    configs = CONFIG.copy()
  
    # Extract models and their configurations for the specified families  
    filtered_configs = {}  
  
    for family in model_families:  
        if family in configs:  
            # Check if model filtering is required  
            if model == "all":  
                # Include all models in the family  
                for model_size, model_configs in configs[family].items():  
                    filtered_configs[f"{family}-{model_size}"] = model_configs  
            else:  
                # Parse the model string (e.g., llama3_8B or llama3-8B)  
                delimiter = "_" if "_" in model else "-"  
                model_parts = model.split(delimiter)  
  
                # Check if the family and size match  
                if len(model_parts) == 2 and model_parts[0] == family:  
                    model_size = model_parts[1]  
                    if model_size in configs[family]:  
                        filtered_configs[f"{family}-{model_size}"] = configs[family][model_size]  
  
    if not filtered_configs:  
        print(f"Warning: No models selected for families: {model_families} with filter: '{model}'")  
  
    return filtered_configs  
  
  
def get_available_models(config_file='model_configs.json', model_families=["llama3"]):  
    """  
    Load model names from the configuration file.  
  
    Args:  
        config_file (str): Path to the configuration JSON file.  
        model_families (list): List of model family names to retrieve.  
  
    Returns:  
        list: A list of available models for the specified families.  
    """  
    configs = CONFIG.copy()
  
    models = [f"{family}-{model}" for family in model_families if family in configs for model in configs[family]]  
  
    return models  


def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility across multiple libraries and configure PyTorch for deterministic behavior.

    Args:
        seed (int): The seed value to set. Default is 42.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)
    # Set seed for NumPy
    np.random.seed(seed)
    # Set seed for PyTorch on CPU
    torch.manual_seed(seed)
    # Set seed for PyTorch on all GPUs (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variable for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)

######################################## HELPERS for Eval ######################################## 
  

def softmax(x):
    n_rows, n_cols = x.shape

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = 256
    y = torch.empty_like(x)

    num_programs = n_rows

    grid = lambda meta: (num_programs, )
    softmax_kernel_naive[grid](
        x,
        y,
        x.stride(0),
        n_cols,
        BLOCK_SIZE,
    )

    return y


def run_softmax(M, N):
    print(f"Running Softmax on shape ({M},{N})")
    set_seed()
    x = torch.randn(M, N, device='cuda')
    y_triton = softmax(x)

    return y_triton


#pytest
@pytest.mark.parametrize('M, N', [(1823, 781), (1, 1), (128, 1), (1, 128), (8192, 8192), (4096, 8192), (359, 1),
                                  (1, 359), (1, 131072), (1, 89999)])
def test_softmax(M, N, request):
    set_seed()
    x = torch.randn(M, N, device='cuda')
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = y_triton.clone().detach().cpu()
    ###################################################################
    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)


#Benchmark
arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}

# --- Define TFLOPS and GB/s calculators for Softmax Forward ---
def calculate_softmax_fwd_gbps(params: dict, ms: float) -> float:
    M, N = params['M'], params['N']
    dtype_str = params.get('dtype_str', 'fp16')
    if dtype_str == 'fp32': current_dtype = torch.float32
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16
    else: current_dtype = torch.float16
    element_size = torch.tensor([], dtype=current_dtype).element_size()
    
    # Read x (M,N), Write y (M,N)
    # Intermediate m (M) and row_sum (M) are usually kept in registers/SRAM per row,
    # not necessarily global memory traffic unless spilled, which is hard to model simply.
    # For online softmax, data is read twice effectively (once for max/sum, once for normalization).
    bytes_read_x_pass1 = M * N * element_size
    bytes_read_x_pass2 = M * N * element_size # Or just once if fully fused and data stays in cache
    bytes_write_y = M * N * element_size

    # A common simplification for bandwidth: 2*M*N (one read, one write)
    # More accurate for online: read_pass1 + read_pass2 + write
    # Let's use 2 reads, 1 write for online softmax
    total_bytes = bytes_read_x_pass1 + bytes_read_x_pass2 + bytes_write_y
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

def calculate_softmax_fwd_tflops(params: dict, ms: float) -> float:
    M, N = params['M'], params['N']
    # FLOPs for Softmax forward (per row):
    # 1. Find max: N-1 comparisons (approx N ops)
    # 2. Subtract max: N subtractions (N ops)
    # 3. Exp: N exponentials (N * ~5-10 ops, say N*5)
    # 4. Sum exps: N-1 additions (approx N ops)
    # 5. Divide by sum: N divisions (N ops)
    # Total per row approx: N + N + 5N + N + N = 9N ops
    flops_per_row = 9 * N 
    total_flops = M * flops_per_row
    tflops = total_flops / (ms / 1000) / 1e12
    return tflops

# --- Name for the benchmark output JSON file ---
OP_NAME_FOR_BENCHMARK = "softmax_triton_perf"

# --- Pytest test_softmax function MODIFIED for performance benchmarking ---
# Original parametrization is kept.
SOFTMAX_SHAPES_FOR_PERF = [
    (2048, 2048), (4096, 4096), (8192, 8192), # Square
    (1, 32000), (1, 131072),                 # Typical vocab sizes (batch 1)
    (1024, 8192), (512, 32000),              # Batch > 1
    # (1,4), (1823, 781) # Smaller/odd shapes
]
SOFTMAX_DTYPES_FOR_PERF = ['fp16', 'bf16', 'fp32']


@pytest.mark.parametrize('M, N', SOFTMAX_SHAPES_FOR_PERF)
@pytest.mark.parametrize('dtype_str', SOFTMAX_DTYPES_FOR_PERF)
def test_performance(M, N, dtype_str, request): # Renamed from test_softmax
    set_seed()
    
    if dtype_str == 'fp32': current_dtype = torch.float32
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16
    else: current_dtype = torch.float16

    x = torch.randn(M, N, device='cuda', dtype=current_dtype)

    # --- Create op_lambda for benchmarking ---
    op_lambda = lambda: softmax(x)

    # --- Benchmarking ---
    bench_config = do_bench_config(warm_up=25, repetition=100)
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)

    # Determine BLOCK_SIZE as it's done in the softmax_triton_wrapper for logging
    # This is for logging consistency, the actual kernel uses autotuned block size.
    # The BLOCK_SIZE passed to kernel is a key for autotuning.
    MAX_FUSED_SIZE_log = 65536 // x.element_size()
    BLOCK_SIZE_log = min(MAX_FUSED_SIZE_log, triton.next_power_of_2(N if N > 0 else 1))
    if BLOCK_SIZE_log == 0: BLOCK_SIZE_log = 1


    current_params_for_logs_and_calc = {
        "M": M, "N": N, "dtype_str": dtype_str,
        "LOGGED_BLOCK_SIZE_heuristic": BLOCK_SIZE_log # Log the heuristic block size
    }

    benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
                              gbps_calculator=calculate_softmax_fwd_gbps,
                              tflops_calculator=calculate_softmax_fwd_tflops)


def model_benchmark_configs(args):
    config_file = args.model_configs
    configs = get_model_configs(config_path=config_file, model_families=["llama3"], model=args.model)

    x_vals_list = []
    batch_size = args.b if args.b else 1

    for model_name, config in configs.items():
        seq_len = args.sq if args.sq else 4096
        x_vals_list.append((model_name, batch_size * seq_len, config["vocab_size"]))

    return x_vals_list


def run_benchmark(args):
    config = []
    if (args.M_benchmark):
        val = args.M_start
        x_vals_list = []
        while val <= args.M_end:
            x_vals_list.append(val)
            val *= args.M_step
        mn_args = {'N': args.N_start}
        plot_name = str("softmax-performance_" + args.dtype + "_N" + str(args.N_start) + "_M" + str(args.M_start) +
                        "-" + str(args.M_end) + "-" + str(args.M_step))
        x_names = ['M']
    else:
        x_vals_list = [i for i in range(args.N_start, args.N_end, args.N_step)]
        mn_args = {'M': args.M_start}
        plot_name = str("softmax-performance_" + args.dtype + "_M" + str(args.M_start) + "_N" + str(args.N_start) +
                        "-" + str(args.N_end) + "-" + str(args.N_step))
        x_names = ['N']

    if args.model:
        assert not args.M_benchmark, \
            "Trying to provide both -model benchmark and M_benchmark is not supported!"
        x_names = ['model', 'M', 'N']
        mn_args = {}
        plot_name = str("softmax-performance_" + args.dtype)
        x_vals_list = model_benchmark_configs(args)

    dtype = arg_to_torch_dtype[args.dtype]

    print(plot_name)
    config.append(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=[
                "Triton",
                "Torch",
            ],
            styles=[('blue', '-'), ('green', '-')],
            ylabel="GB/s",
            plot_name=plot_name,
            args=mn_args,
        ))

    @triton.testing.perf_report(config)
    def benchmark(M, N, provider, model=None):
        x = torch.randn(M, N, device='cuda', dtype=dtype)
        stream = torch.cuda.Stream()
        torch.cuda.set_stream(stream)
        if provider == 'torch':
            ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
        if provider == 'triton':
            ms = triton.testing.do_bench(lambda: softmax(x))
        gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms)

    benchmark.run(save_path=".", show_plots=True, print_data=True)

######################################## HELPERS for Eval ########################################     
# --- Pytest hook to save the dictionary at the end of the session ---  
def test_save_results():  
    """  
    Called after whole test run finished, right before returning the exit status to the system.  
    """
    print('Inside session finish...')
    if "_CALL_SUCCESS_" not in result_gold:
        result_gold['_CALL_SUCCESS_'] = torch.tensor([[0.0]])
    OUTPUT_FILENAME = __file__.replace('.','_') + '.pt'
    print(f"\nSaving all y_triton results to {OUTPUT_FILENAME}...")  
    # Ensure the directory for the output file exists if it's in a subdirectory  
    output_dir = os.path.dirname(OUTPUT_FILENAME)  
    if output_dir and not os.path.exists(output_dir):  
        os.makedirs(output_dir, exist_ok=True)  
    torch.save(result_gold, OUTPUT_FILENAME)       
    print(f"Successfully saved {len(result_gold)} y_triton tensors to {OUTPUT_FILENAME}.")  

def test_save_performance_results():
    """
    Called after the test_performance function finishes.
    This is a separate hook to ensure performance results are saved.
    """
    print('\nPytest session finishing... Saving benchmark results...')

    output_directory = os.path.join(os.path.dirname(__file__), "perf")  # Save in a "perf" subdirectory next to the test file
    os.makedirs(output_directory, exist_ok=True)
    
    save_all_benchmark_results(output_directory)
    print(f"All benchmark results attempted to save to: {output_directory}")

######################################## HELPERS for Eval ######################################## 

def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark Softmax",
        allow_abbrev=False,
    )
    parser.add_argument('-model_configs', type=str, default="model_configs.json", help="Model config json file.")

    available_models = get_available_models(model_families=["llama3"])  # Dynamically load model names
    model_help = (
        "Model name to benchmark. Select from: [" + ", ".join(available_models) +
        "]. Use 'all' to benchmark all models. Not providing runs the default benchmark script with custom configs.")
    parser.add_argument('-model', type=str, default=None, help=model_help)
    parser.add_argument('-b', type=int, default=0, help="Batch size used together with model.")
    parser.add_argument('-sq', type=int, default=0, help="Sequence length used together with model.")
    parser.add_argument('-M', "--M_start", default="1", type=int)
    parser.add_argument('-Ms', "--M_step", default="2", type=int)
    parser.add_argument('-Me', "--M_end", default="512", type=int)
    parser.add_argument('-Mb', "--M_benchmark", default=False, type=bool)

    parser.add_argument('-N', "--N_start", default="1024", type=int)
    parser.add_argument('-Ns', "--N_step", default="2048", type=int)
    parser.add_argument('-Ne', "--N_end", default="65536", type=int)

    parser.add_argument('-d', "--dtype", default="fp16")
    parser.add_argument('-nb', "--no_benchmark", default=False, type=bool)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.no_benchmark:
        run_softmax(args.M_start, args.N_start)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
