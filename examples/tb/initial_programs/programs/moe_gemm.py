# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
import triton  
import triton.language as tl  


@triton.jit  
def moe_gemm_kernel(  
    A,  
    B,  
    Out,  
    A_scale,  
    B_scale,  
    stride_am,  
    stride_ak,  
    stride_be,  
    stride_bn,  
    stride_bk,  
    stride_cm,  
    stride_cn,  
    stride_bse,  
    stride_bsn,  
    top_k: tl.constexpr,  
    topk_weights_ptr,  
    sorted_token_ids_ptr,  
    expert_ids_ptr,  
    EM: tl.constexpr,  
    N: tl.constexpr,  
    K: tl.constexpr,  
    EVEN_K: tl.constexpr,  
    MUL_ROUTED_WEIGHT: tl.constexpr,  
    use_fp8_w8a8: tl.constexpr,  
    use_int8_w8a16: tl.constexpr,  
    use_int8_w8a8: tl.constexpr,  
    BLOCK_SIZE_M: tl.constexpr,  
    BLOCK_SIZE_N: tl.constexpr,  
    BLOCK_SIZE_K: tl.constexpr,  
    GROUP_SIZE_M: tl.constexpr,  
):  
    """  
    Implements the fused computation for a Mixture of Experts (MOE) using  
    token and expert matrices.  
  
    Key Parameters:  
    - A: The input tensor representing tokens with shape (*, K), where '*' can  
        be any shape representing batches and K is the feature dimension of  
        each token.  
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is  
        the number of experts, K is the input feature dimension, and N is  
        the output feature dimension.  
    - C: The output cache tensor with shape (M, topk, N), where M is the  
        total number of tokens post padding, topk is the number of times  
        each token is repeated, and N is the output feature dimension.  
    - sorted_token_ids: A tensor containing the sorted indices of tokens,  
        repeated topk times and arranged by the expert index they are  
        assigned to.  
    - expert_ids: A tensor containing the indices of the expert for each  
        block. It determines which expert matrix from B should be used for  
        each block in A.  
    This kernel performs the multiplication of a token by its corresponding  
    expert matrix as determined by `expert_ids`. The sorting of  
    `sorted_token_ids` by expert index and padding ensures divisibility by  
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix  
    multiplication across different blocks processed by the same expert.  
    """  
    pid = tl.program_id(axis=0)  
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)  
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  
    num_pid_in_group = GROUP_SIZE_M * num_pid_n  
    group_id = pid // num_pid_in_group  
    first_pid_m = group_id * GROUP_SIZE_M  
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)  
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)  
    pid_n = (pid % num_pid_in_group) // group_size_m  
  
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)  
  
    # Here we assume that valid tokens are in the range [0, M).  
    token_mask = (offs_token >= 0) & (offs_token < EM)  
  
    off_experts = tl.load(expert_ids_ptr + pid_m)  
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N  
    offs_k = tl.arange(0, BLOCK_SIZE_K)  
    a_ptrs = A + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)  
    b_ptrs = B + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)  
  
    if use_int8_w8a16:  
        b_scale_ptrs = B_scale + off_experts * stride_bse + offs_bn[None, :] * stride_bsn  
        b_scale = tl.load(b_scale_ptrs)  
  
    if use_fp8_w8a8 or use_int8_w8a8:  
        a_scale = tl.load(A_scale)  
        b_scale = tl.load(B_scale + off_experts)  
  
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  
  
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):  
        # Masking ensures we don't load from invalid tokens or indices  
        if EVEN_K:  
            a = tl.load(a_ptrs, mask=(token_mask[:, None]), other=0.0)  
            b = tl.load(b_ptrs)  
        else:  
            a = tl.load(a_ptrs, mask=(token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K)), other=0.0)  
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K), other=0.0)  
  
        if use_int8_w8a16:  
            accumulator = tl.dot(a, b.to(a.dtype), acc=accumulator)  
        elif use_fp8_w8a8 or use_int8_w8a8:  
            accumulator += tl.dot(a, b)  
        else:  
            accumulator = tl.dot(a, b, acc=accumulator)  
  
        a_ptrs += BLOCK_SIZE_K * stride_ak  
        b_ptrs += BLOCK_SIZE_K * stride_bk  
  
    if MUL_ROUTED_WEIGHT:  
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)  
        accumulator = accumulator * moe_weight[:, None]  
  
    if use_int8_w8a16:  
        accumulator = (accumulator * b_scale).to(Out.dtype.element_ty)  
    elif use_fp8_w8a8 or use_int8_w8a8:  
        accumulator = (accumulator * a_scale * b_scale).to(Out.dtype.element_ty)  
    else:  
        accumulator = accumulator.to(Out.dtype.element_ty)  
  
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  
    out_ptrs = Out + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]  
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)  
    tl.store(out_ptrs, accumulator.to(Out.dtype.element_ty), mask=c_mask)


##################################################################################################################################################

  
import triton # Required for triton.testing utilities and launching kernel  
import torch  
import pytest  
from typing import Any, Dict, Optional  
import os  
import json  
import functools  
import argparse  
import sys  
import triton.language as tl # Required for tl.constexpr in quantize_input and other places  


from tb_eval.perf.ROCm.performance_utils_pytest import PytestBenchmarker, do_bench_config, save_all_benchmark_results
from typing import Dict
######################################## HELPERS for Eval ######################################## 
import numpy as np
import random
import torch 

result_gold = {}
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


M_THRESHOLD_SMALL = 256  
M_THRESHOLD_MEDIUM = 1024  
  
dtype_max = {  
    dtype: (torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)).max  
    for dtype in [  
        torch.float8_e5m2fnuz,  
        torch.float8_e4m3fnuz,  
        torch.int8,  
    ]  
}  
  
supported_fp8 = [torch.float8_e4m3fnuz, torch.float8_e5m2fnuz]  
  
  
class MetaData():  
    use_fp8_w8a8 = False  
    use_int8_w8a16 = False  
    use_int8_w8a8 = False  
  
    def __init__(self, top_k, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, config):  
        self.top_k = top_k  
        self.topk_weights = topk_weights  
        self.topk_ids = topk_ids  
        self.sorted_token_ids = sorted_token_ids  
        self.expert_ids = expert_ids  
        self.num_tokens_post_padded = num_tokens_post_padded  
        self.config = config  
  
    def set_use_fp8_w8a8(self, a_descale, b_descale, fp8_type):  
        self.use_fp8_w8a8 = True  
        self.a_descale = a_descale  
        self.b_descale = b_descale  
        self.fp8_type = fp8_type  
  
    def set_use_int8_w8a16(self, b_descale):  
        self.use_int8_w8a16 = True  
        self.b_descale = b_descale  
        self.a_descale = None  
  
    def set_use_int8_w8a8(self, a_descale, b_descale):  
        self.use_int8_w8a8 = True  
        self.a_descale = a_descale  
        self.b_descale = b_descale  
  
    def check_args(self, a, b, o):  
        assert a.shape[-1] == b.shape[-1] and b.shape[1] == o.shape[-1]  
  
        assert not (self.use_fp8_w8a8 and self.use_int8_w8a16 and self.use_int8_w8a8)  
        if self.use_fp8_w8a8:  
            assert self.fp8_type in supported_fp8, f"fp8 type {self.fp8_type} not supported"  
  
  
def _moe_align_block_size(topk_ids: torch.Tensor, num_experts: int, top_k: int, block_size: int,  
                          sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor,  
                          num_tokens_post_pad: torch.Tensor) -> None:  
    M, top_k = topk_ids.shape  
  
    expert_to_tokens = [[] for _ in range(num_experts)]  
    # For each token, for each selected expert, we append (token_id, expert)  
    for token_id in range(M):  
        for j in range(top_k):  
            e_id = topk_ids[token_id, j].item()  
            expert_to_tokens[e_id].append(token_id * top_k + j)  
  
    # Reorder tokens block by block, padding if needed  
    reordered_token_ids = []  
    reordered_expert_ids = []  
  
    for e_id in range(num_experts):  
        tokens_for_expert = expert_to_tokens[e_id]  
        num_tokens = len(tokens_for_expert)  
  
        n_blocks = ((num_tokens + block_size - 1) // block_size)  
        # If not a multiple of block_size, pad up to the next multiple  
        padded_size = n_blocks * block_size  
  
        # Reorder all actual tokens for expert e_id  
        reordered_token_ids.extend(tokens_for_expert)  
        # reordered_expert_ids.extend([e_id]*num_tokens)  
        reordered_expert_ids.extend([e_id] * n_blocks)  
  
        # Pad with dummy token_id = -1 (or any sentinel), if needed  
        if padded_size > num_tokens:  
            pad_count = padded_size - num_tokens  
            reordered_token_ids.extend([-1] * pad_count)  
  
    token_length = len(reordered_token_ids)  
    expert_length = len(reordered_expert_ids)  
  
    sorted_token_ids[:token_length] = torch.tensor(reordered_token_ids, dtype=sorted_token_ids.dtype,  
                                                   device=sorted_token_ids.device)  
    expert_ids[:expert_length] = torch.tensor(reordered_expert_ids, dtype=expert_ids.dtype, device=expert_ids.device)  
  
    # Fill remainder with -1 if these arrays are bigger than total_length  
    if token_length < sorted_token_ids.numel():  
        sorted_token_ids[token_length:] = -1  
    if expert_length < expert_ids.numel():  
        expert_ids[expert_length:] = -1  
  
    num_tokens_post_pad.fill_(token_length)  
  
  
def moe_align_block_size(topk_ids: torch.Tensor, block_size: int,  
                         num_experts: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  
    """  
    Aligns the token distribution across experts to be compatible with block size for matrix multiplication.  
  
    Parameters:  
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the top-k expert indices for each token.  
    - block_size: The block size used in block matrix multiplication.  
    - num_experts: The total number of experts.  
  
    Returns:  
    - sorted_token_ids: A tensor containing the sorted token indices according to their allocated expert.  
    - expert_ids: A tensor indicating the assigned expert index for each block.  
    - num_tokens_post_padded: The total number of tokens after padding, ensuring divisibility by block_size.  
  
    This function pads the number of tokens that each expert needs to process so that it is divisible by block_size.  
    Padding ensures that during block matrix multiplication, the dimensions align correctly.  
  
    Example:  
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]], block_size = 4, and num_experts = 4:  
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts, with each expert needing to process 3 tokens.  
    - As block_size is 4, we pad 1 token for each expert.  
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].  
    - Then append padding tokens [12, 12, 12, 12] for each block.  
    - After sorting by expert index, we obtain token_ids [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].  
        Tokens 12 are non-existent (padding) and are ignored in the subsequent matrix multiplication.  
    - The padding ensures that the total number of tokens is now divisible by block_size for proper block matrix operations.  
    """  
    top_k = topk_ids.shape[1]  
    sorted_ids = torch.empty((topk_ids.numel() + num_experts * (block_size - 1), ), dtype=torch.int32,  
                             device=topk_ids.device)  
    expert_ids = torch.empty((topk_ids.numel() + num_experts, ), dtype=torch.int32, device=topk_ids.device)  
    sorted_ids.fill_(topk_ids.numel())  
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)  
    _moe_align_block_size(topk_ids, num_experts, top_k, block_size, sorted_ids, expert_ids, num_tokens_post_pad)  
  
    return sorted_ids, expert_ids, num_tokens_post_pad  
  
  
def get_config_dtype_str(dtype: torch.dtype, use_int8_w8a16: Optional[bool] = False,  
                         use_int8_w8a8: Optional[bool] = False, use_fp8_w8a8: Optional[bool] = False):  
    if use_fp8_w8a8:  
        return "fp8_w8a8"  
    elif use_int8_w8a16:  
        return "int8_w8a16"  
    elif use_int8_w8a8:  
        return "int8_w8a8"  
    elif dtype == torch.float:  
        # avoiding cases where kernel fails when float32 MoE  
        # use fp16/bfloat16 configs  
        return "float32"  
    return None  
  
  
def get_config_file_name(dtype: Optional[str]) -> str:  
    device_name = torch.cuda.get_device_name(0).replace(" ", "_")  
    dtype_selector = "" if not dtype else f",dtype={dtype}"  
    return f"device_name={device_name}{dtype_selector}.json"  
  
  
@functools.lru_cache  
def get_moe_configs(dtype: Optional[str]) -> Optional[Dict[int, Any]]:  
    """  
    Return optimized configurations for the fused MoE kernel.  
  
    The return value will be a dictionary that maps an irregular grid of  
    batch sizes to configurations of the fused_moe kernel. To evaluate the  
    kernel on a given batch size bs, the closest batch size in the grid should  
    be picked and the associated configuration chosen to invoke the kernel.  
    """  
    # First look up if an optimized configuration is available in the configs  
    # directory  
    json_file_name = get_config_file_name(dtype)  
  
    config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name)  
    if os.path.exists(config_file_path):  
        with open(config_file_path) as f:  
            # If a configuration has been found, return it  
            return {key: val for key, val in json.load(f).items()}  
  
    # If no optimized configuration is available, we will use the default  
    # configuration  
    return None  
  
  
def get_default_config(  
    M: int,  
    E: int,  
    is_marlin: bool,  
) -> Dict[str, int]:  
    # config = {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}  
    config = {}
    # A heuristic: fused marlin works faster with this config for small M  
    if M <= E or (is_marlin and M <= 32):  
        # config = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}  
        config = {}
    return config  
  
  
def try_get_optimal_moe_config(  
    E: int,  
    dtype: Optional[str],  
    M: int,  
    is_marlin: bool = False,  
):  
    configs = get_moe_configs(dtype)  
  
    if configs:  
        if configs: # This inner 'if configs:' is redundant  
            if M < M_THRESHOLD_SMALL:  
                config = configs["small_M"]  
            elif M < M_THRESHOLD_MEDIUM:  
                config = configs["medium_M"]  
            else:  
                config = configs["large_M"]  

            ## NOTE: removing autotunable parameters
            if config:
                block_keys = []
                for k, v in config.items():
                    if k.startswith("BLOCK_SIZE") or k.startswith("GROUP_SIZE"):  
                        block_keys.append(k)  
                if len(block_keys):
                    for k in block_keys:  
                        del config[k]  # Remove block size keys from config
    else:  
        # Else use the default config  
        config = get_default_config(M, E, is_marlin)  
  
    return config  
  
  
def moe_gemm(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, metadata: MetaData) -> None:  
    # TODO shard M dim  
    metadata.check_args(a, b, c)  
  
    num_tokens_post_padded, topk_weights, sorted_token_ids, expert_ids, config = metadata.num_tokens_post_padded, metadata.topk_weights, metadata.sorted_token_ids, metadata.expert_ids, metadata.config  
  
    use_fp8_w8a8, use_int8_w8a16, use_int8_w8a8 = metadata.use_fp8_w8a8, metadata.use_int8_w8a16, metadata.use_int8_w8a8  
    a_descale, b_descale = None, None  
    stride_bse = None  
    stride_bsn = None  
    if use_fp8_w8a8 or use_int8_w8a16 or use_int8_w8a8:  
        a_descale, b_descale = metadata.a_descale, metadata.b_descale  
        if use_int8_w8a16:  
            stride_bse = b_descale.stride(0)  
            stride_bsn = b_descale.stride(1)  
  
    top_k = metadata.top_k  
  
    EM = num_tokens_post_padded.item()  
    _, N, K = b.shape  
    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )  
  
    # EVEN_K = K % config["BLOCK_SIZE_K"] == 0  
    EVEN_K = K % config.get("BLOCK_SIZE_K", 1) == 0        
    # This is where the kernel defined above is called  
    # We need to ensure moe_gemm_kernel is in scope, which it is if defined globally or imported.  
    # For this re-structuring, it's assumed the kernel from the <triton-kernel-code> block is accessible.  
    # If running this directly, you might need to ensure the kernel definition is executed first.  
    moe_gemm_kernel[grid](a, b, c, a_descale,  
                          b_descale, a.stride(0), a.stride(1), b.stride(0), b.stride(1), b.stride(2), c.stride(1),  
                          c.stride(2), stride_bse, stride_bsn, top_k, topk_weights, sorted_token_ids, expert_ids, EM, N,  
                          K, EVEN_K, MUL_ROUTED_WEIGHT=topk_weights is not None, use_fp8_w8a8=use_fp8_w8a8,  
                          use_int8_w8a16=use_int8_w8a16, use_int8_w8a8=use_int8_w8a8, **config)  
    return c  
  
  
def quantize_tensor(tensor: torch.Tensor, dtype, dim=()) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  
    quantize_dim = [i for i in range(tensor.dim()) if i not in dim]  
    max_vals = tensor.abs().amax(dim=quantize_dim, keepdim=True)  
    max_repr_val = dtype_max[dtype]  
    # Avoid division by zero  
    max_vals[max_vals == 0] = 1e-8  
  
    # Compute scale factors for each channel  
    scale: torch.Tensor = max_repr_val / max_vals.to(torch.float32)  
  
    # Quantize the tensor  
    tensor = tensor * scale  
    if dtype == torch.int8:  
        tensor = tensor.round_()  
    tensor.clamp_(-max_repr_val, max_repr_val)  
    tensor_quantized = tensor.to(dtype)  
  
    scale = scale.squeeze(dim=quantize_dim)  
  
    return tensor_quantized, scale, 1 / scale  
  
  
def quantize_input(a, b, use_fp8_w8a8: tl.constexpr, use_int8_w8a16: tl.constexpr, use_int8_w8a8: tl.constexpr,  
                   metatdata: MetaData, fp8_type=None):  
    assert not (use_fp8_w8a8 and use_int8_w8a16 and use_int8_w8a8)  
    assert not (use_fp8_w8a8 and fp8_type is None)  
  
    if use_fp8_w8a8:  
        a_quantized, _, a_descale = quantize_tensor(a, dtype=fp8_type)  
        b_quantized, _, b_descale = quantize_tensor(b, dim=(0, ), dtype=fp8_type)  
        metatdata.set_use_fp8_w8a8(a_descale, b_descale, fp8_type)  
        return a_quantized, b_quantized  
  
    if use_int8_w8a8:  
        a_quantized, _, a_descale = quantize_tensor(a, dtype=torch.int8)  
        b_quantized, _, b_descale = quantize_tensor(b, dim=(0, ), dtype=torch.int8)  
        metatdata.set_use_int8_w8a8(a_descale, b_descale)  
        return a_quantized, b_quantized  
  
    if use_int8_w8a16:  
        b_quantized, _, b_descale = quantize_tensor(b, dim=(0, 1), dtype=torch.int8)  
        metatdata.set_use_int8_w8a16(b_descale)  
        return a, b_quantized  
  
  
def input_helper(M: int, N: int, K: int, top_k: int, E: int, routed_weight: bool, use_fp8_w8a8: bool,  
                 use_int8_w8a16: bool, use_int8_w8a8: bool, fp8_type, dtype):  

    set_seed()
    a = torch.randn((M, K), dtype=dtype, device='cuda')  
    b = torch.randn((E, N, K), dtype=dtype, device='cuda')  
    c = torch.zeros((M, top_k, N), dtype=dtype, device='cuda')  
  
    values = torch.randn(M, E, device='cuda')  
  
    softmax_vals = torch.softmax(values, dim=1)  
    topk_weights, topk_ids = torch.topk(softmax_vals, k=top_k, dim=1)  
  
    config_dtype = get_config_dtype_str(use_fp8_w8a8=use_fp8_w8a8, use_int8_w8a16=use_int8_w8a16,  
                                        use_int8_w8a8=use_int8_w8a8, dtype=dtype)  
    get_config_func = functools.partial(  
        try_get_optimal_moe_config,  
        E,  
        config_dtype,  
    )  
    config = get_config_func(M)  
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, config['BLOCK_SIZE_M'], E)  
  
    metadata = MetaData(top_k, topk_weights if routed_weight else None, topk_ids, sorted_token_ids, expert_ids,  
                        num_tokens_post_padded, config)  
  
    if use_fp8_w8a8 or use_int8_w8a16 or use_int8_w8a8:  
        a, b = quantize_input(a, b, use_fp8_w8a8, use_int8_w8a16, use_int8_w8a8, metadata, fp8_type)  
  
    return a, b, c, metadata  
  
  
@pytest.mark.parametrize("M, N, K, top_k, E", [  
    (64, 14336, 4096, 2, 8),  
    (16, 14336, 1, 2, 4),  
    # (1, 14336, 128, 2, 4),  
    # (3, 14336, 128, 2, 4),  
    # (16, 14336, 128, 1, 4),  
    # (16, 14336, 128, 1, 1),  
    # (64, 7186, 128, 2, 8),  
    # (64, 3584, 128, 2, 8),  
    # (64, 1792, 128, 2, 8),  
    (64, 64, 128, 2, 8),  
])  
@pytest.mark.parametrize('routed_weight', [True, False])  
def test_correctness(M: int, N: int, K: int, top_k: int, E: int, routed_weight: bool, request, dtype=torch.float16):  
    
    a, b, c, metadata = input_helper(M, N, K, top_k, E, routed_weight=routed_weight, use_fp8_w8a8=False,  
                                     use_int8_w8a16=False, use_int8_w8a8=False, fp8_type=None, dtype=dtype)  
  
    tri_out = moe_gemm(a, b, c, metadata)  

    
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = tri_out.clone().detach().cpu()
    ###################################################################
    
    topk_ids = metadata.topk_ids  
    topk_weights = metadata.topk_weights  
    ref_out = torch.empty_like(c)  
    # Repeat a -> (M, top_k, K)  
    a_expanded = a.unsqueeze(1).repeat(1, top_k, 1)  
    # (M, top_k, N, K)  
    b_indexed = b[topk_ids]  
    ref_out = torch.einsum("mek,menk->men", a_expanded, b_indexed)  
    if routed_weight:  
        ref_out *= topk_weights.unsqueeze(-1)  

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    # Validate correctness  
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=1e-2)  
  
  
@pytest.mark.parametrize("M, N, K, top_k, E", [  
    (64, 14336, 4096, 2, 8),  
    (16, 14336, 1, 2, 4),  
    (1, 14336, 128, 2, 4),  
    (16, 14336, 128, 1, 4),  
    (16, 14336, 128, 1, 1),  
    (64, 7186, 128, 2, 8),  
    (64, 3584, 128, 2, 8),  
    (64, 1792, 128, 2, 8),  
    (64, 64, 128, 2, 8),  
])  
@pytest.mark.parametrize('routed_weight', [True, False])  
@pytest.mark.parametrize('use_fp8_w8a8', [True])  
@pytest.mark.parametrize('fp8_type', [torch.float8_e4m3fnuz, torch.float8_e5m2fnuz])  
def test_correctness_fp8(M: int, N: int, K: int, top_k: int, E: int, routed_weight: bool, use_fp8_w8a8, fp8_type, request, dtype=torch.float16): 

    a, b, c, metadata = input_helper(M, N, K, top_k, E, routed_weight=routed_weight, use_fp8_w8a8=use_fp8_w8a8,  
                                     use_int8_w8a16=False, fp8_type=fp8_type, use_int8_w8a8=False, dtype=dtype)  
  
    tri_out = moe_gemm(a, b, c, metadata)

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = tri_out.clone().detach().cpu()
    ###################################################################

    topk_ids = metadata.topk_ids  
    topk_weights = metadata.topk_weights  
    ref_out = torch.empty_like(c)  
    # Repeat a -> (M, top_k, K)  
    a_expanded = a.unsqueeze(1).repeat(1, top_k, 1)  
    # (M, top_k, N, K)  
    b_indexed = b.half()[topk_ids]  
    ref_out = torch.einsum("mek,menk->men", a_expanded.float(), b_indexed.float())  
  
    if routed_weight:  
        ref_out *= topk_weights.unsqueeze(-1)  
  
    ref_out = ref_out * metadata.b_descale[topk_ids].unsqueeze(-1)  
    ref_out = ref_out * metadata.a_descale  
    ref_out = ref_out.to(dtype)  
  
    # Validate correctness  
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=1e-2)  
  
  
@pytest.mark.parametrize("M, N, K, top_k, E", [  
    (64, 14336, 4096, 2, 8),  
    (16, 14336, 1, 2, 4),  
    (1, 14336, 128, 2, 4),  
    (16, 14336, 128, 1, 4),  
    (16, 14336, 128, 1, 1),  
    (64, 7186, 128, 2, 8),  
    (64, 3584, 128, 2, 8),  
    (64, 1792, 128, 2, 8),  
    (64, 64, 128, 2, 8),  
])  
@pytest.mark.parametrize('routed_weight', [True, False])  
@pytest.mark.parametrize('use_int8_w8a16', [True])  
def test_correctness_int8_w8a16(M: int, N: int, K: int, top_k: int, E: int, routed_weight: bool, use_int8_w8a16, request,  
                                dtype=torch.float16):  
    a, b, c, metadata = input_helper(M, N, K, top_k, E, routed_weight=routed_weight, use_fp8_w8a8=False,  
                                     use_int8_w8a16=use_int8_w8a16, use_int8_w8a8=False, fp8_type=None, dtype=dtype)  
  
    tri_out = moe_gemm(a, b, c, metadata)  
    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = tri_out.clone().detach().cpu()
    ###################################################################
    
    topk_ids = metadata.topk_ids  
    topk_weights = metadata.topk_weights  
    ref_out = torch.empty_like(c)  
    # Repeat a -> (M, top_k, K)  
    a_expanded = a.unsqueeze(1).repeat(1, top_k, 1)  
    # (M, top_k, N, K)  
    b_indexed = b[topk_ids]  
    ref_out = torch.einsum("mek,menk->men", a_expanded.float(), b_indexed.float())  
    if routed_weight:  
        ref_out *= topk_weights.unsqueeze(-1)  
  
    ref_out = ref_out * metadata.b_descale[topk_ids, :]  
    ref_out = ref_out.to(dtype)  
  
    # Validate correctness  
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=1e-2)  
  
  
@pytest.mark.parametrize("M, N, K, top_k, E", [  
    (64, 14336, 4096, 2, 8),  
    (16, 14336, 1, 2, 4),  
    (1, 14336, 128, 2, 4),  
    (16, 14336, 128, 1, 4),  
    (16, 14336, 128, 1, 1),  
    (64, 7186, 128, 2, 8),  
    (64, 3584, 128, 2, 8),  
    (64, 1792, 128, 2, 8),  
    (64, 64, 128, 2, 8),  
])  
@pytest.mark.parametrize('routed_weight', [True, False])  
@pytest.mark.parametrize('use_int8_w8a8', [True])  
def test_correctness_int8_w8a8(M: int, N: int, K: int, top_k: int, E: int, routed_weight: bool, use_int8_w8a8, request,  
                               dtype=torch.float16):    
    a, b, c, metadata = input_helper(M, N, K, top_k, E, routed_weight=routed_weight, use_fp8_w8a8=False,  
                                     use_int8_w8a16=False, use_int8_w8a8=use_int8_w8a8, fp8_type=None, dtype=dtype)  
  
    tri_out = moe_gemm(a, b, c, metadata)  
    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = tri_out.clone().detach().cpu()
    ###################################################################
    
    topk_ids = metadata.topk_ids  
    topk_weights = metadata.topk_weights  
    ref_out = torch.empty_like(c)  
    # Repeat a -> (M, top_k, K)  
    a_expanded = a.unsqueeze(1).repeat(1, top_k, 1)  
    # (M, top_k, N, K)  
    b_indexed = b[topk_ids]  
    ref_out = torch.einsum("mek,menk->men", a_expanded.float(), b_indexed.float())  
    if routed_weight:  
        ref_out *= topk_weights.unsqueeze(-1)  
  
    ref_out = ref_out * metadata.b_descale[topk_ids].unsqueeze(-1)  
    ref_out = ref_out * metadata.a_descale  
    ref_out = ref_out.to(dtype)  
  
    
    # Validate correctness  
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=1e-2)  

# --- Define TFLOPS and GB/s calculators for MoE GEMM ---
def calculate_moe_gemm_tflops(params: dict, ms: float) -> float:
    M = params['M_orig'] # Original number of tokens before top_k expansion
    N = params['N']
    K = params['K']
    top_k = params['top_k']
    # Each of M tokens interacts with top_k experts.
    # For each such interaction, it's an M_slice * K @ K * N GEMM.
    # Effective operations: M * top_k * (2 * K * N)
    flops = M * top_k * (2 * K * N)
    if params.get('routed_weight', False):
        flops += M * top_k * N # Element-wise multiplication by routing weights
    tflops = flops / (ms / 1000) / 1e12
    return tflops

def calculate_moe_gemm_gbps(params: dict, ms: float) -> float:
    M_orig = params['M_orig']
    N = params['N']
    K = params['K']
    E = params['E']
    top_k = params['top_k']
    dtype_str = params.get('dtype_str', 'fp16') # Default if not specified

    if dtype_str == 'fp32': current_dtype = torch.float32
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16
    else: current_dtype = torch.float16
    element_size = torch.tensor([], dtype=current_dtype).element_size()

    # Memory I/O:
    # Read A: (M_orig, K)
    # Read B (all expert weights): (E, N, K) - worst case, all loaded
    # Read routing info: topk_ids (M_orig, top_k), expert_ids (complex, related to padded M), topk_weights (M_orig, top_k)
    # Write C: (M_orig, top_k, N) -> effectively M_orig * top_k rows of size N are written.

    bytes_a = M_orig * K * element_size
    bytes_b = E * N * K * element_size # All expert weights
    # Output c_for_kernel is (EM_padded, N). EM_padded is roughly M_orig * top_k.
    # Let's use M_orig * top_k for a cleaner estimate of useful data written.
    bytes_c_written = M_orig * top_k * N * element_size

    # Routing info bytes are usually smaller and sometimes omitted for simplicity,
    # but can be significant for small M, N, K.
    # topk_ids: M_orig * top_k * 4 (int32)
    # expert_ids: (num_blocks_for_kernel) * 4 (int32) - harder to estimate precisely without full alignment logic
    # topk_weights: M_orig * top_k * element_size (if routed_weight)
    bytes_routing = M_orig * top_k * 4 # topk_ids
    if params.get('routed_weight', False):
        bytes_routing += M_orig * top_k * element_size

    # For MoE, a common simplification is A + B_active + C
    # B_active = M_orig * top_k * K * N (effectively) but data comes from (E,N,K)
    # Let's use: Read A, Read all B, Write C_useful
    total_bytes = bytes_a + bytes_b + bytes_c_written # Add bytes_routing if significant
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

# --- Name for the benchmark output JSON file ---
OP_NAME_FOR_BENCHMARK = "moe_gemm_triton_perf"

# --- Pytest parametrize for performance testing (based on test_correctness) ---
MOE_GEMM_PARAMS_FOR_PERF = [
    # M, N, K, top_k, E
    (64, 14336, 4096, 2, 8),
    (256, 7168, 4096, 2, 8), # Example medium size
    (1024, 14336, 4096, 2, 8), # Example larger M
    (16, 1024, 512, 1, 4),   # Smaller K, N
]
MOE_DTYPES_FOR_PERF = ['fp16'] #['fp16', 'bf16'] # Reduced set for faster perf testing
# Quantization modes are complex to integrate here simply, focus on main dtypes first.

@pytest.mark.parametrize("M_orig, N, K, top_k, E", MOE_GEMM_PARAMS_FOR_PERF)
@pytest.mark.parametrize('routed_weight', [True, False])
@pytest.mark.parametrize('dtype_str', MOE_DTYPES_FOR_PERF)
def test_performance(M_orig, N, K, top_k, E, routed_weight, dtype_str, request):
    # Determine torch dtype
    set_seed()
    if dtype_str == 'fp32': current_dtype = torch.float32
    elif dtype_str == 'bf16': current_dtype = torch.bfloat16
    else: current_dtype = torch.float16

    # --- Input Setup using input_helper ---
    # For performance, we are not testing quantization variants like fp8/int8 here for simplicity.
    # That would require more parameters for use_fp8_w8a8, etc.
    a, b, c_for_kernel, metadata = input_helper(
        M=M_orig, N=N, K=K, top_k=top_k, E=E,
        routed_weight=routed_weight,
        use_fp8_w8a8=False, use_int8_w8a16=False, use_int8_w8a8=False,
        fp8_type=None, dtype=current_dtype
    )

    # --- Create op_lambda for benchmarking ---
    op_lambda = lambda: moe_gemm(a, b, c_for_kernel, metadata)

    # --- Benchmarking ---
    bench_config = do_bench_config(warm_up=10, repetition=50) # MoE can be slower
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)

    current_params_for_logs_and_calc = {
        "M_orig": M_orig, "N": N, "K": K, "top_k": top_k, "E": E,
        "routed_weight": routed_weight, "dtype_str": dtype_str,
        # Include relevant metadata.config if it affects performance
        "BLOCK_SIZE_M": metadata.config['BLOCK_SIZE_M'],
        "BLOCK_SIZE_N": metadata.config['BLOCK_SIZE_N'],
        "BLOCK_SIZE_K": metadata.config['BLOCK_SIZE_K'],
        "GROUP_SIZE_M": metadata.config['GROUP_SIZE_M'],
    }

    benchmarker.run_benchmark(current_params_dict=current_params_for_logs_and_calc,
                              gbps_calculator=calculate_moe_gemm_gbps,
                              tflops_calculator=calculate_moe_gemm_tflops)
    
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
    print(f"\nSaving all tri_out results to {OUTPUT_FILENAME}...")  
    # Ensure the directory for the output file exists if it's in a subdirectory  
    output_dir = os.path.dirname(OUTPUT_FILENAME)  
    if output_dir and not os.path.exists(output_dir):  
        os.makedirs(output_dir, exist_ok=True)  
    torch.save(result_gold, OUTPUT_FILENAME)       
    print(f"Successfully saved {len(result_gold)} tri_out tensors to {OUTPUT_FILENAME}.")  

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


def get_configs():  
    configs = [  
        {"M": 64, "N": 256, "K": 128, "E": 8, "top_k": 2},  
        {"M": 64, "N": 1792, "K": 1024, "E": 8, "top_k": 2},  
        {"M": 64, "N": 7168, "K": 4096, "E": 8, "top_k": 2},  
        {"M": 128, "N": 7168, "K": 4096, "E": 8, "top_k": 2},  
        {"M": 1024, "N": 7168, "K": 4096, "E": 8, "top_k": 2},  
        {"M": 4096, "N": 7168, "K": 4096, "E": 8, "top_k": 2},  
        {"M": 64, "N": 14336, "K": 4096, "E": 8, "top_k": 2},  
        {"M": 128, "N": 14336, "K": 4096, "E": 8, "top_k": 2},  
        {"M": 256, "N": 14336, "K": 4096, "E": 8, "top_k": 2},  
        {"M": 512, "N": 14336, "K": 4096, "E": 8, "top_k": 2},  
        {"M": 1024, "N": 14336, "K": 4096, "E": 8, "top_k": 2},  
        {"M": 2048, "N": 14336, "K": 4096, "E": 8, "top_k": 2},  
        {"M": 4096, "N": 14336, "K": 4096, "E": 8, "top_k": 2},  
    ]  
    return configs  
  
  
def model_benchmark_configs(args):  
    config_file = args.model_configs  
    configs = get_model_configs(config_path=config_file, model_families=["mistral"], model=args.model)  
    moe_configs = []  
    M = args.M if args.M else 4096  # check size  
    # M, K, N, E, top_k  
  
    for model_name, config in configs.items():  
        N1 = config["intermediate_size"]  
        K1 = config["hidden_size"]  
  
        N2 = config["hidden_size"]  
        K2 = config["intermediate_size"] // 2  
  
        E = 8  
        top_k = 2  
        # The first moe layer  
        moe_configs.append((model_name, M, N1, K1, E, top_k))  
        # The second moe layer  
        moe_configs.append((model_name, M * top_k, N2, K2, E, 1))  
  
    return moe_configs  
  
  
def run_benchmark(custom, args):  
    routed_weight = args.routed_weight  
    use_int8_w8a16 = args.int8_w8a16  
    use_fp8_w8a8 = args.fp8_w8a8  
    use_int8_w8a8 = args.int8_w8a8  
    dtype = arg_to_torch_dtype[args.dtype]  
    fp8_type = arg_to_torch_dtype[args.fp8_type]  
  
    x_names = ['M', 'N', 'K', 'E', 'top_k']  
    if custom:
        assert args.M and args.N and args.K and args.E and args.top_k, \
            "Please provide M, N, K, E, top_k for custom runs."
        x_vals_list = [(args.M, args.N, args.K, args.E, args.top_k)] 
    else:  
        if args.model:  
            x_vals_list = model_benchmark_configs(args)  
            x_names = ['model', 'M', 'N', 'K', 'E', 'top_k']  
        else:  
            configs = get_configs()  
            x_vals_list = [(cfg['M'], cfg['N'], cfg['K'], cfg['E'], cfg['top_k']) for cfg in configs]  
  
    line_names = ['Time (ms)', 'TFLOPS', 'Bandwidth (GB/s)']  
    line_vals = ['time', 'tflops', 'bandwidth']  
  
    benchmark = triton.testing.Benchmark(  
        x_names=x_names, x_vals=x_vals_list, line_arg='metric', line_vals=line_vals, line_names=line_names,  
        styles=[('red', '-'), ('blue', '-'),  
                ('yellow', '-')], ylabel='ms / TFLOPS / GB/s', plot_name='moe-gemm-benchmark', args={  
                    'dtype': dtype, 'routed_weight': routed_weight, 'use_fp8_w8a8': use_fp8_w8a8, 'use_int8_w8a16':  
                    use_int8_w8a16, 'use_int8_w8a8': use_int8_w8a8, 'fp8_type': fp8_type  
                })  
  
    @triton.testing.perf_report([benchmark])  
    def bench_moe_gemm(M, N, K, E, top_k, dtype, routed_weight, metric, use_fp8_w8a8, use_int8_w8a16, use_int8_w8a8,  
                       fp8_type, model=None):  
        a, b, c, metadata = input_helper(M, N, K, top_k, E, routed_weight=routed_weight, use_fp8_w8a8=use_fp8_w8a8,  
                                         use_int8_w8a16=use_int8_w8a16, use_int8_w8a8=use_int8_w8a8, fp8_type=fp8_type,  
                                         dtype=dtype)  
  
        # (M, K) * (top_k, N, K) -> (M, top_k, N). 2 for multiplication and accumulation  
        flops = 2.0 * M * top_k * K * N  
        # The weight is applied on the gemm product which has the shape of (M, top_k, N)  
        if routed_weight:  
            flops += M * top_k * N  
  
        if use_fp8_w8a8:  
            a_bytes = b_bytes = torch.tensor([], dtype=fp8_type).element_size()  
            c_bytes = torch.tensor([], dtype=dtype).element_size()  
        if use_int8_w8a8: # This should be elif  
            a_bytes = b_bytes = torch.tensor([], dtype=torch.int8).element_size()  
            c_bytes = torch.tensor([], dtype=torch.int8).element_size() # This was torch.int8, should be dtype for c  
        elif use_int8_w8a16:  
            b_bytes = torch.tensor([], dtype=torch.int8).element_size()  
            a_bytes = c_bytes = torch.tensor([], dtype=dtype).element_size()  
        else:  
            a_bytes = b_bytes = c_bytes = torch.tensor([], dtype=dtype).element_size()  
  
        # (M, K) memory load for A (E,  N,  K) for B not (top_k,  N,  K) because we are in total bringing in all expert matrices into the chip from memory. It's just that not all multiply the same A.  
        mem_read = (M * K) * a_bytes + (E * N * K) * b_bytes  
        # Memory write for the gemm product  
        mem_write = (M * top_k * N) * c_bytes  
        mem = mem_read + mem_write  
        fn = lambda: moe_gemm(a, b, c, metadata)  
        ms = triton.testing.do_bench(fn)  
  
        bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s  
        tflops = flops / ms * 1e-9  

        ############ BENCHMARK LOGGING ############
        log_entry = f"ms: {ms} bandwidth: {bandwidth} tflops: {tflops}" + '\n'
        OUTPUT_FILENAME = __file__.replace('.','_') + '_benchmark.log'
        with open(OUTPUT_FILENAME, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        ############ BENCHMARK LOGGING ############
        
        # Return exactly one scalar depending on which metric is active  
        if metric == 'time':  
            return ms  
        elif metric == 'tflops':  
            return tflops  
        elif metric == 'bandwidth':  
            return bandwidth  
        else:  
            raise ValueError("Unknown metric: " + metric)  
  
    bench_moe_gemm.run(save_path=".", print_data=True)  
  
  
def parse_args():  
    parser = argparse.ArgumentParser(  
        prog="Benchmark MoE GEMM",  
        allow_abbrev=False,  
    )  
    parser.add_argument('-model_configs', type=str, default="model_configs.json", help="Model config json file.")  
    available_models = get_available_models(model_families=["mistral"])  # Dynamically load model names  
    model_help = ("Model name to benchmark. Select from: [" + ", ".join(available_models) +  
                  "]. Use 'all' to benchmark all models or leave blank for the default benchmark script.")  
    parser.add_argument('-model', type=str, default=None, help=model_help)  
    parser.add_argument("-M", type=int, default=0, help="M dimension")  
    parser.add_argument("-K", type=int, default=0, help="K dimension")  
    parser.add_argument("-N", type=int, default=0, help="N dimension")  
    parser.add_argument("-E", type=int, default=0, help="Number of experts")  
    parser.add_argument("-top_k", type=int, default=0, help="top_k experts per token")  
    parser.add_argument("-routed_weight", action='store_true', default=False)  
    parser.add_argument("-int8_w8a16", action='store_true', default=False)  
    parser.add_argument("-int8_w8a8", action='store_true', default=False)  
    parser.add_argument("-fp8_w8a8", action='store_true', default=False)  
    parser.add_argument("-dtype", default='fp16')  
    parser.add_argument("-fp8_type", default='e5m2fnuz')  
    args = parser.parse_args()  
    return args  
  
  
arg_to_torch_dtype = {  
    'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32, "e5m2fnuz": torch.float8_e5m2fnuz, "e4m3fnuz":  
    torch.float8_e4m3fnuz  
}  
  
  
def main():  
    args = parse_args()  
    custom_config = False  
    # If user provides all M,K,N,E,top_k we consider it custom  
    if args.M and args.K and args.N and args.E and args.top_k:  
        custom_config = True  
    run_benchmark(custom_config, args)

if __name__ == '__main__':  
    sys.exit(main())  
