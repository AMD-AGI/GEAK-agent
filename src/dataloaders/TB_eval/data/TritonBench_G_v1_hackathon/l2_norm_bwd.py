# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/thunlp/TritonBench - Apache License 2.0




##################################################################################################################################################


import torch

# Test the backward L2 normalization
def test_l2_norm_bwd():
    results = {}
    
    # Test case 1: Default case
    x = torch.randn(4, 8, device='cuda', dtype=torch.float32)
    dy = torch.randn(4, 8, device='cuda', dtype=torch.float32)
    dx = _l2_norm_bwd(x, dy)
    results['test_case_1'] = dx

    # Test case 2: Different shape
    x = torch.randn(2, 16, device='cuda', dtype=torch.float32)
    dy = torch.randn(2, 16, device='cuda', dtype=torch.float32)
    dx = _l2_norm_bwd(x, dy)
    results['test_case_2'] = dx

    # Test case 3: Larger tensor
    x = torch.randn(8, 8, device='cuda', dtype=torch.float32)
    dy = torch.randn(8, 8, device='cuda', dtype=torch.float32)
    dx = _l2_norm_bwd(x, dy)
    results['test_case_3'] = dx

    # Test case 4: Edge case with small tensor
    x = torch.randn(1, 8, device='cuda', dtype=torch.float32)
    dy = torch.randn(1, 8, device='cuda', dtype=torch.float32)
    dx = _l2_norm_bwd(x, dy)
    results['test_case_4'] = dx

    return results

# Run the tests
result_gold = test_l2_norm_bwd()
