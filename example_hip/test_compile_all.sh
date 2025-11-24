#!/bin/bash
echo "Testing compilation of all naive kernels..."
echo

for dir in gemm_naive softmax_naive reduction_naive attention_naive conv_naive layernorm_naive transpose_naive; do
    if [ -d "$dir" ]; then
        echo "Testing $dir..."
        cd "$dir"
        if make clean > /dev/null 2>&1 && make > /dev/null 2>&1; then
            echo "  ✓ $dir compiles successfully"
        else
            echo "  ✗ $dir failed to compile"
        fi
        cd ..
    fi
done
echo
echo "Done!"
