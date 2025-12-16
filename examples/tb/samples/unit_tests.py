# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/algorithmicsuperintelligence/openevolve  - Apache License 2.0

import torch 

def get_input_tensors():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shapes = [
        (1, 1), (16, 16), (32, 32), (64, 64),
        (128, 128), (256, 256), (1024, 1024),
        (1, 1024), (1024, 1),   
        (37, 59),  # weird prime dims
    ]
    dtypes = [torch.float16, torch.float32, torch.bfloat16]
    tensors = []
    for shape in input_shapes:
        for dtype in dtypes:
            tensors.append(
                            (
                                torch.randn(shape, dtype=dtype, device=device),
                                torch.randn(shape, dtype=dtype, device=device)
                            )
                )
    return tensors
