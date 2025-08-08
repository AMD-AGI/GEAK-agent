# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/thunlp/TritonBench - Apache License 2.0



##################################################################################################################################################


# Test the forward function with different configurations
def test_swiglu_fwd():
    results = {}
    # Test case 1
    batch_size = 4
    ncols = 128
    xy = torch.randn(batch_size, 2 * ncols, device='cuda', dtype=torch.float32)
    out = _swiglu_fwd(xy)
    results['test_case_1'] = out.detach().cpu()

    # Test case 2
    batch_size = 8
    ncols = 256
    xy = torch.randn(batch_size, 2 * ncols, device='cuda', dtype=torch.float32)
    out = _swiglu_fwd(xy)
    results['test_case_2'] = out.detach().cpu()

    # Test case 3
    batch_size = 16
    ncols = 512
    xy = torch.randn(batch_size, 2 * ncols, device='cuda', dtype=torch.float32)
    out = _swiglu_fwd(xy)
    results['test_case_3'] = out.detach().cpu()

    # Test case 4
    batch_size = 32
    ncols = 1024
    xy = torch.randn(batch_size, 2 * ncols, device='cuda', dtype=torch.float32)
    out = _swiglu_fwd(xy)
    results['test_case_4'] = out.detach().cpu()

    return results

# Run the tests
result_gold = test_swiglu_fwd()
