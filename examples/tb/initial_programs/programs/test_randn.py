# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
######################################## Imports ######################################## 
import numpy as np
import pytest
import torch

import triton
import triton.language as tl
######################################## Imports ######################################## 

#####################################
# Triton Kernels for randint
#####################################

BLOCK: tl.constexpr = 1024

@triton.jit
def randn_kernel_runtime_seed(X, N, seed, dtype: tl.constexpr):
    pid = tl.program_id(0).to(dtype)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    rand = tl.rand(seed, offset)
    tl.store(X + offset, rand, mask=offset < N)

@triton.jit
def randn_kernel_const_seed(X, N, seed: tl.constexpr, dtype: tl.constexpr):
    pid = tl.program_id(0).to(dtype)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    rand = tl.rand(seed, offset)
    tl.store(X + offset, rand, mask=offset < N)

##################################################################################################################################################  

import numpy as np
import random
import torch 
import os
import pytest
from numpy.random import RandomState
import triton
import scipy.stats
import triton.language as tl

result_gold = {}
from tb_eval.perf.ROCm.performance_utils_pytest import PytestBenchmarker, do_bench_config, save_all_benchmark_results
from typing import Dict
BLOCK: tl.constexpr = 1024
######################################## HELPERS for Eval ######################################## 
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

#####################################
# Reference Philox Implementation
#####################################


class PhiloxConfig:

    def __init__(self, PHILOX_ROUND_A, PHILOX_ROUND_B, PHILOX_KEY_A, PHILOX_KEY_B, DTYPE):
        self.PHILOX_ROUND_A = np.array(PHILOX_ROUND_A, dtype=DTYPE)
        self.PHILOX_ROUND_B = np.array(PHILOX_ROUND_B, dtype=DTYPE)
        self.PHILOX_KEY_A = np.array(PHILOX_KEY_A, dtype=DTYPE)
        self.PHILOX_KEY_B = np.array(PHILOX_KEY_B, dtype=DTYPE)
        self.DTYPE = DTYPE


# This is better for GPU
PHILOX_32 = PhiloxConfig(
    PHILOX_KEY_A=0x9E3779B9,
    PHILOX_KEY_B=0xBB67AE85,
    PHILOX_ROUND_A=0xD2511F53,
    PHILOX_ROUND_B=0xCD9E8D57,
    DTYPE=np.uint32,
)

# This is what numpy implements
PHILOX_64 = PhiloxConfig(
    PHILOX_KEY_A=0x9E3779B97F4A7C15,
    PHILOX_KEY_B=0xBB67AE8584CAA73B,
    PHILOX_ROUND_A=0xD2E7470EE14C6C93,
    PHILOX_ROUND_B=0xCA5A826395121157,
    DTYPE=np.uint64,
)


class CustomPhilox4x:

    def __init__(self, seed, config):
        self._config = config
        seed = self._into_pieces(seed)
        self._key = np.array(seed[:2], dtype=self._dtype)
        self._counter = np.array((0, 0) + seed[2:], dtype=self._dtype)

    @property
    def _dtype(self):
        return self._config.DTYPE

    def _into_pieces(self, n, pad=4):
        res = []
        bits = np.dtype(self._dtype).itemsize * 8
        while len(res) < pad:
            res.append(np.array((n & ((1 << bits) - 1)), dtype=self._dtype))
            n >>= bits
        assert n == 0
        return tuple(res)

    def _multiply_low_high(self, a, b):
        low = a * b
        high = int(a) * int(b)
        high = np.array(high >> (np.dtype(self._dtype).itemsize * 8), dtype=self._dtype)
        return low, high

    def _single_round(self, counter, key):
        lo0, hi0 = self._multiply_low_high(self._config.PHILOX_ROUND_A, counter[0])
        lo1, hi1 = self._multiply_low_high(self._config.PHILOX_ROUND_B, counter[2])
        ret0 = hi1 ^ counter[1] ^ key[0]
        ret1 = lo1
        ret2 = hi0 ^ counter[3] ^ key[1]
        ret3 = lo0
        return np.array([ret0, ret1, ret2, ret3], dtype=self._dtype)

    def _raise_key(self, key):
        pk = [self._config.PHILOX_KEY_A, self._config.PHILOX_KEY_B]
        return key + np.array(pk, dtype=self._dtype)

    def random_raw(self):
        counter = self._counter
        key = self._key
        for _ in range(10):
            counter = self._single_round(counter, key)
            key = self._raise_key(key)
        self.advance(1)
        return counter

    def advance(self, n_steps):
        self._counter[0] += n_steps
        assert self._counter[0] < 2**32, "FIXME: doesn't work for large offsets"


class CustomPhilox(CustomPhilox4x):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = []

    def random_raw(self):
        if len(self.buffer) == 0:
            self.buffer = list(super().random_raw())[::-1]
        return int(self.buffer.pop())


@pytest.mark.interpreter
@pytest.mark.parametrize('size, seed, dtype, const_seed', [(size, seed, dtype, const_seed)
                                                           for size in [100000]
                                                           for seed in [0, 42, 124, 54]
                                                           for dtype in ['int32', 'int64']
                                                           for const_seed in [True, False]])
def test_rand(size, seed, dtype, const_seed,  request, device='cuda'):


    # triton result
    set_seed(seed)

    x = torch.empty(size, dtype=torch.float32, device=device)
    N = x.numel()
    grid = (triton.cdiv(N, BLOCK), )
    if const_seed:
        randn_kernel_const_seed[grid](x, N, seed=seed, dtype=getattr(tl, dtype))
    else:
        randn_kernel_runtime_seed[grid](x, N, seed, dtype=getattr(tl, dtype))

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = x.clone().detach().cpu()
    ################################################################### 

    assert all((x >= 0) & (x <= 1))
    assert scipy.stats.kstest(x.tolist(), 'uniform', args=(0, 1)).statistic < 0.01



# tl.rand() should never produce >=1.0


def randn_triton_wrapper(X_buffer, N_elements, seed_val, 
                         is_const_seed: bool, tl_dtype_for_kernel_arg, # This is the tl.dtype 
                         num_warps_launch):
    grid = (triton.cdiv(N_elements, PYTHON_BLOCK_SIZE_CONST),)
    
    if is_const_seed:
        randn_kernel_const_seed[grid](
            X_buffer, N_elements, 
            seed=seed_val, 
            dtype=tl_dtype_for_kernel_arg, # *** CORRECTED: Use 'dtype' as keyword arg ***
            # num_warps=num_warps_launch
        )
    else:
        randn_kernel_runtime_seed[grid](
            X_buffer, N_elements, 
            seed_val, 
            dtype=tl_dtype_for_kernel_arg, # *** CORRECTED: Use 'dtype' as keyword arg ***
            # # num_warps=num_warps_launch
        )
    return X_buffer

# --- TFLOPS and GB/s calculators (Unchanged from previous correct version) ---
def calculate_randn_tflops(params: dict, ms: float) -> float:
    N = params['N_elements']
    ops_per_rand = 75 
    flops = N * ops_per_rand
    tflops = flops / (ms / 1000) / 1e12
    return tflops

def calculate_randn_gbps(params: dict, ms: float) -> float:
    N = params['N_elements']
    element_size = torch.tensor([], dtype=torch.float32).element_size() # tl.rand output is float
    bytes_x_write = N * element_size
    total_bytes = bytes_x_write
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

OP_NAME_FOR_BENCHMARK = "triton_rand_perf"

# --- Pytest parametrize for performance testing (Unchanged from previous correct version) ---
KERNEL_SUB_N_VALS_FOR_PERF = [2**i for i in range(10, 21)] # 1024 to 1,048,576 (original had KERNEL_SUB_... this should be RAND_...)
RAND_SIZES_FOR_PERF = [2**i for i in range(10, 23)] # Up to 4M elements, adjust if too slow/large
RAND_SEEDS_FOR_PERF = [0, 42] 
RAND_OFFSET_DTYPES_FOR_PERF = ['int32', 'int64'] 
RAND_CONST_SEED_BOOL_FOR_PERF = [True, False]
# NUM_WARPS_FOR_PERF = [4, 8] 
PYTHON_BLOCK_SIZE_CONST = 1024

@pytest.mark.parametrize("size_val", RAND_SIZES_FOR_PERF)
@pytest.mark.parametrize("seed_val", RAND_SEEDS_FOR_PERF)
@pytest.mark.parametrize("offset_tl_dtype_str", RAND_OFFSET_DTYPES_FOR_PERF)
@pytest.mark.parametrize("const_seed_bool", RAND_CONST_SEED_BOOL_FOR_PERF)
# @pytest.mark.parametrize("num_warps_val", NUM_WARPS_FOR_PERF) # Can add back if desired
def test_performance(size_val, seed_val, offset_tl_dtype_str, const_seed_bool, request, device='cuda'):
    # num_warps_val = 4 # Or from parametrize
    set_seed() 
    
    x_output_buffer = torch.empty(size_val, dtype=torch.float32, device=device)
    N_elements = x_output_buffer.numel()
    
    if offset_tl_dtype_str == 'int64':
        tl_dtype_for_offset_arg = tl.int64
    else: 
        tl_dtype_for_offset_arg = tl.int32
        
    op_lambda = lambda: randn_triton_wrapper(
        x_output_buffer, N_elements, seed_val,
        const_seed_bool, tl_dtype_for_offset_arg, # Pass the tl.dtype object
        num_warps_launch=4 
    )

    bench_config = do_bench_config(warm_up=25, repetition=100) 
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)
    current_params_for_logs_and_calc = {
        "N_elements": N_elements, 
        "seed": seed_val,
        "offset_dtype": offset_tl_dtype_str, # Log the string
        "const_seed": const_seed_bool,
        "output_dtype_str": "float32",
        "num_warps": 4 # Log the fixed num_warps
    }
    
    perf_result = benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
                                            gbps_calculator=calculate_randn_gbps,
                                            tflops_calculator=calculate_randn_tflops)


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