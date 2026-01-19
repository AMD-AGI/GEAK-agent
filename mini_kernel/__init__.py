"""
Mini-Kernel Agent - Autonomous GPU Kernel Optimization

Battle-tested on:
- TopK & Sort kernels (achieved 1.23x speedup)
- MLA (Multi-head Latent Attention) kernels

This package provides autonomous GPU kernel optimization using:
- Profiler-guided bottleneck analysis
- OpenEvolve evolutionary search
- Actual kernel code generation (not just wrappers!)
- Correctness verification

Usage:
    # CLI
    python -m mini_kernel path/to/kernel.py --gpu 0 --evolve
    
    # Or via wrapper script
    ./mini-kernel optimize path/to/kernel.py --gpu 0
"""

__version__ = "1.0.0"
__author__ = "Mini-Kernel Team"

# Core exports
from .cli import main

__all__ = [
    "__version__",
    "main",
]
