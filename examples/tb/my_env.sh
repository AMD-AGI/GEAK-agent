#!/bin/bash
# GEAK-OpenEvolve Environment Configuration
# Source this file before running optimization

# Automatically load .env file from workspace root
ENV_FILE="/wekafs/dougljia/.env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment from $ENV_FILE..."
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

# Map AMD_API_KEY to OPENAI_API_KEY (used by OpenEvolve)
if [ -n "$AMD_API_KEY" ]; then
    export OPENAI_API_KEY="$AMD_API_KEY"
    echo "OPENAI_API_KEY set from AMD_API_KEY"
fi

# ROCm Evaluator Configuration
export ROCM_GOLDEN_DATA_PATH="/wekafs/dougljia/GEAK-eval-OE/geak_eval/data/ROCm/data/performance/golden_results"

echo "GEAK-OpenEvolve environment configured!"
echo "ROCM_GOLDEN_DATA_PATH=$ROCM_GOLDEN_DATA_PATH"
