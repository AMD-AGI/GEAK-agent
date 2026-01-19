"""
ROCm Kernel Profiler - Generic profiling utility for the agent framework.

This module provides tools for profiling GPU kernels and analyzing
bottlenecks to guide optimization.

Example usage:
    from profiler import GenericProfiler, profile_module
    
    # Quick profiling
    analysis = profile_module(script_content, module_name="my_module")
    
    # Or use the class directly
    profiler = GenericProfiler(gpu_id=0)
    results = profiler.profile_script(script_content)
    analysis = profiler.analyze_results(results)
    profiler.print_analysis(analysis)
"""

from .generic_profiler import (
    GenericProfiler,
    profile_module,
    KernelMetrics,
    ModuleAnalysis,
    BottleneckType,
)

from .bottleneck_analyzer import (
    BottleneckAnalyzer,
    BottleneckReport,
    BottleneckType as BottleneckAnalyzerType,
)

from .roofline import RooflineModel

from .openevolve_integration import ProfilingFitness

__all__ = [
    # Main profiler
    "GenericProfiler",
    "profile_module",
    
    # Data classes
    "KernelMetrics",
    "ModuleAnalysis",
    "BottleneckReport",
    
    # Analysis
    "BottleneckType",
    "BottleneckAnalyzer",
    "RooflineModel",
    
    # Integration
    "ProfilingFitness",
]

__version__ = "1.0.0"
