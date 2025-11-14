#!/bin/bash
# OpenEvolve Environment Configuration Template
# Source this file or copy these exports to your shell profile

# LLM API Configuration
# Required: Your LLM API key (e.g., OpenAI, Anthropic, or custom endpoint)
export OPENAI_API_KEY="your_api_key_here"

# ROCm Evaluator Configuration
# Path to golden results data for TB-eval-OE
export ROCM_GOLDEN_DATA_PATH="/path/to/TB-eval-OE/tb_eval/data/ROCm/data/performance/golden_results"

# Optional: GPU Architecture (for ROCm)
# export GPU_ARCHS="gfx950"

# Optional: Maximum parallel jobs for compilation
# export MAX_JOBS=16

