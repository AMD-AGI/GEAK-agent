# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/thunlp/TritonBench - Apache License 2.0


##################################################################################################################################################


def test_cos_function():
    # Create test cases with various input sizes
    test_cases = {
        'test_case_1': torch.rand(1024, device='cuda') * 2 * math.pi,
        'test_case_2': torch.rand(2048, device='cuda') * 2 * math.pi,
        'test_case_3': torch.rand(4096, device='cuda') * 2 * math.pi,
        'test_case_4': torch.rand(8192, device='cuda') * 2 * math.pi
    }
    
    results = {}
    
    for case_name, input_tensor in test_cases.items():
        # Compute cosine using Triton
        B_triton = cos(input_tensor)
        results[case_name] = B_triton
    
    return results

# Run the test
result_gold = test_cos_function()
