#!/bin/bash

echo "Checking for optimization-related comments in baseline kernels..."
echo

all_clean=true

for dir in gemm_naive reduction_naive softmax_naive conv_naive attention_naive layernorm_naive transpose_naive silu; do
    if [ -f "$dir/${dir}.hip" ] || [ -f "$dir/$(ls $dir/*.hip 2>/dev/null | head -1)" ]; then
        hip_file=$(find "$dir" -name "*.hip" | head -1)
        
        # Check for optimization-related keywords in comments
        if grep -iE '(optimize|optimization|improve|performance|efficient|inefficient|slow|fast|bad|good|naive|better|worse)' "$hip_file" | grep -E '(//|/\*|\*)' >/dev/null 2>&1; then
            echo "✗ $dir - Contains optimization comments"
            all_clean=false
        else
            echo "✓ $dir - No optimization comments"
        fi
    fi
done

echo
if [ "$all_clean" = true ]; then
    echo "✓ All kernels are clean - no optimization hints in comments"
else
    echo "⚠ Some kernels contain optimization hints"
fi
