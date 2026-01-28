"""
MCP Tools for mini-kernel Agent

Core MCP Tools:
- profiler-mcp: Hardware-level bottleneck analysis
- bench-mcp: Reliable performance measurement
- verify-mcp: Functional correctness validation
- metrix-mcp: Detailed hardware metrics via metrix
- openevolve-mcp: Evolutionary parameter and code search
- module-mcp: Multi-kernel discovery and fusion analysis

Each tool is invoked on demand, based on agent state and optimization progress.
"""

from .profiler import ProfilerTool
from .bench import BenchTool
from .verify import VerifyTool
from .metrix import MetrixTool

__all__ = [
    "ProfilerTool",
    "BenchTool", 
    "VerifyTool",
    "MetrixTool",
]


