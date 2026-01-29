"""
Automated Test Discovery MCP Server

Provides tools for automatically discovering tests and benchmarks for GPU kernels.
No configuration needed - uses content-based detection with auto-pattern learning.

Tools:
- discover_for_kernel: Find tests and benchmarks for a kernel file
- discover_kernels: Find all kernel files in a workspace
- analyze_file: Check if a file is a test, benchmark, or kernel
- get_test_command: Get the command to run a discovered test
"""

from .server import mcp

__all__ = ["mcp"]
