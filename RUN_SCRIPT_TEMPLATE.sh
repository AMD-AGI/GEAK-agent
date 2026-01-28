#!/bin/bash
# {KERNEL_NAME} - GPU {GPU_DEVICE}
# Run inside Docker container: docker exec -it minikernel_sdubagun bash

export HIP_VISIBLE_DEVICES={GPU_DEVICE}
export PYTHONPATH="/home/sdubagun/work/repos/GEAK-agent:/home/sdubagun/work/repos/aiter"

# AMD LLM API Key for LLM-based optimization (Claude Opus 4.5)
# Set your key here or export it before running
export AMD_LLM_API_KEY="${AMD_LLM_API_KEY}"

cd /home/sdubagun/work/repos/GEAK-agent

echo "=== {KERNEL_NAME} Optimization ==="
echo "GPU: {GPU_DEVICE} (HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES)"
echo "Kernel: /home/sdubagun/work/repos/GEAK-agent/{WORKSPACE_DIR}/kernel.py"
echo ""

python -m mini_kernel.cli \
    /home/sdubagun/work/repos/GEAK-agent/{WORKSPACE_DIR}/kernel.py \
    --gpu {GPU_DEVICE} \
    --evolve \
    --no-docker \
    --iterations {ITERATIONS} \
    --work-dir /home/sdubagun/work/repos/GEAK-agent/{WORKSPACE_DIR}/optimization_output

echo ""
echo "=== {KERNEL_NAME} Complete ==="
