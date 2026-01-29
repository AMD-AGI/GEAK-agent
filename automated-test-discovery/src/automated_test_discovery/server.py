"""
Automated Test Discovery MCP Server

Provides content-based test and benchmark discovery for GPU kernels.
No configuration files needed - learns patterns automatically from the codebase.
"""

import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP(
    name="automated-test-discovery",
    instructions="""
    Automated Test Discovery for GPU Kernels.
    
    Use these tools to find tests and benchmarks for kernel files:
    - discover_for_kernel: Main tool - finds tests/benchmarks for a kernel
    - discover_kernels: Find all kernel files in a directory
    - analyze_file: Check what type a file is (test, benchmark, kernel, or none)
    - get_test_command: Get the command to run a specific test file
    
    The discovery uses content-based detection (not directory names) and 
    automatically learns project-specific patterns like custom decorators.
    """
)


# ============================================================================
# Content Detection Patterns
# ============================================================================

# Patterns for kernel files
KERNEL_PATTERNS = [
    r"@triton\.jit",
    r"@triton\.autotune",
    r"__global__\s+void",
    r"tl\.load|tl\.store",
]

# Test content keywords (pattern, score)
TEST_KEYWORDS = [
    # Pytest
    (r"import pytest", 0.3),
    (r"@pytest\.mark", 0.3),
    (r"def test_\w+\s*\(", 0.4),
    # Assertions
    (r"assert\s+", 0.2),
    (r"\.allclose\(", 0.3),
    (r"\.assertEqual\(", 0.2),
    (r"torch\.testing\.assert", 0.3),
    # Custom frameworks
    (r"@perftest\(\)", 0.35),
    (r"checkAllclose", 0.35),
    (r"from.*test_common import", 0.25),
    # Correctness
    (r"correctness", 0.2),
    (r"verify|verification", 0.15),
    # Structure
    (r"class Test\w+", 0.3),
    (r"unittest", 0.2),
    # GTest (C++)
    (r"TEST\s*\(\s*\w+\s*,", 0.5),
    (r"EXPECT_TRUE|EXPECT_EQ", 0.35),
    (r"ASSERT_TRUE|ASSERT_EQ", 0.35),
]

# Benchmark content keywords
BENCH_KEYWORDS = [
    (r"elapsed_time|elapsed", 0.3),
    (r"latency", 0.25),
    (r"throughput", 0.25),
    (r"TFLOPS|GFLOPS", 0.4),
    (r"us/iter|ms/iter", 0.3),
    (r"warmup|warm_up", 0.25),
    (r"benchmark|bench_", 0.3),
    (r"torch\.cuda\.Event\(enable_timing", 0.4),
    (r"triton\.testing\.do_bench", 0.5),
    (r"speedup", 0.25),
    (r"GB/s|TB/s", 0.3),
    (r"hipEventElapsedTime|cudaEventElapsedTime", 0.4),
]

# Directories to skip
SKIP_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "node_modules",
    "build", "dist", ".eggs", "site-packages", ".tox", ".pytest_cache"
}


# ============================================================================
# Helper Functions
# ============================================================================

def _should_skip(path: Path) -> bool:
    """Check if path should be skipped."""
    for part in path.parts:
        if part in SKIP_DIRS or part.endswith(".egg-info"):
            return True
    return False


def _is_kernel_file(path: Path) -> bool:
    """Check if file contains kernel definitions."""
    try:
        content = path.read_text()[:3000]
        for pattern in KERNEL_PATTERNS:
            if re.search(pattern, content):
                return True
    except Exception:
        pass
    return False


def _score_as_test(path: Path) -> float:
    """Score a file as a potential test (0.0 - 1.0+)."""
    try:
        content = path.read_text()
    except Exception:
        return 0.0
    
    score = 0.0
    for pattern, points in TEST_KEYWORDS:
        if re.search(pattern, content, re.IGNORECASE):
            score += points
    
    # Filename bonus
    if "test" in path.name.lower():
        score += 0.1
    
    return score


def _score_as_bench(path: Path) -> float:
    """Score a file as a potential benchmark (0.0 - 1.0+)."""
    try:
        content = path.read_text()
    except Exception:
        return 0.0
    
    score = 0.0
    for pattern, points in BENCH_KEYWORDS:
        if re.search(pattern, content, re.IGNORECASE):
            score += points
    
    # Filename bonus
    if "bench" in path.name.lower() or "perf" in path.name.lower():
        score += 0.1
    
    return score


def _get_test_command(path: Path) -> str:
    """Generate command to run a test file."""
    try:
        content = path.read_text()[:2000]
    except Exception:
        content = ""
    
    if path.suffix == ".py":
        if "import pytest" in content or "@pytest" in content:
            return f"pytest {path} -v"
        elif "unittest" in content:
            return f"python -m unittest {path}"
        else:
            return f"python {path}"
    elif path.suffix in [".cpp", ".cc", ".cu", ".hip"]:
        return f"# Build and run: {path.name}"
    else:
        return f"# Unknown: {path}"


def _expand_workspace(kernel_path: Path) -> Path:
    """Expand from kernel file to project root."""
    markers = ["pyproject.toml", "setup.py", ".git", "tests", "op_tests"]
    
    current = kernel_path.parent
    for _ in range(15):
        for marker in markers:
            if (current / marker).exists():
                return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    
    return kernel_path.parent


# ============================================================================
# MCP Tools
# ============================================================================

@mcp.tool()
def discover_for_kernel(
    kernel_path: str,
    max_results: int = 10
) -> dict:
    """
    Discover tests and benchmarks for a GPU kernel file.
    
    Uses content-based detection to find related test and benchmark files.
    No configuration needed - works on any project structure.
    
    Args:
        kernel_path: Path to the kernel file (.py, .cu, .hip)
        max_results: Maximum number of results per category (default: 10)
    
    Returns:
        Dictionary with:
        - kernel_name: Name of the kernel
        - kernel_type: Type (triton, hip, cuda)
        - tests: List of discovered tests with confidence scores
        - benchmarks: List of discovered benchmarks with confidence scores
        - workspace: The detected project root
    """
    path = Path(kernel_path)
    if not path.exists():
        return {"error": f"Kernel file not found: {kernel_path}"}
    
    # Expand to workspace
    workspace = _expand_workspace(path)
    
    # Get kernel info
    kernel_name = path.stem
    kernel_type = "unknown"
    try:
        content = path.read_text()[:2000]
        if "@triton" in content or "tl." in content:
            kernel_type = "triton"
        elif "__global__" in content and "hip" in content.lower():
            kernel_type = "hip"
        elif "__global__" in content:
            kernel_type = "cuda"
    except Exception:
        pass
    
    # Split kernel name for partial matching
    kernel_parts = [p.lower() for p in kernel_name.split("_") if len(p) > 2]
    
    # Scan for tests and benchmarks
    tests = []
    benchmarks = []
    extensions = [".py", ".cpp", ".cc", ".cu", ".hip"]
    
    for ext in extensions:
        for file_path in workspace.rglob(f"*{ext}"):
            if _should_skip(file_path):
                continue
            if file_path == path:
                continue
            if _is_kernel_file(file_path):
                continue
            
            fname_lower = file_path.name.lower()
            
            # Score as test
            test_score = _score_as_test(file_path)
            if test_score >= 0.3:
                # Kernel name matching boost
                if kernel_name.lower() in fname_lower:
                    test_score += 1.0
                elif kernel_parts:
                    matches = sum(1 for p in kernel_parts if p in fname_lower)
                    if matches >= 2:
                        test_score += 0.3 * matches
                
                tests.append({
                    "file": str(file_path),
                    "name": file_path.name,
                    "confidence": min(test_score, 1.0),
                    "command": _get_test_command(file_path)
                })
            
            # Score as benchmark
            bench_score = _score_as_bench(file_path)
            if bench_score >= 0.3:
                if kernel_name.lower() in fname_lower:
                    bench_score += 1.0
                elif kernel_parts:
                    matches = sum(1 for p in kernel_parts if p in fname_lower)
                    if matches >= 2:
                        bench_score += 0.3 * matches
                
                benchmarks.append({
                    "file": str(file_path),
                    "name": file_path.name,
                    "confidence": min(bench_score, 1.0),
                    "command": f"python {file_path}"
                })
    
    # Sort by confidence and limit
    tests.sort(key=lambda x: x["confidence"], reverse=True)
    benchmarks.sort(key=lambda x: x["confidence"], reverse=True)
    
    return {
        "kernel_name": kernel_name,
        "kernel_type": kernel_type,
        "workspace": str(workspace),
        "tests": tests[:max_results],
        "benchmarks": benchmarks[:max_results],
        "total_tests_found": len(tests),
        "total_benchmarks_found": len(benchmarks)
    }


@mcp.tool()
def discover_kernels(
    directory: str,
    max_results: int = 20
) -> dict:
    """
    Find all GPU kernel files in a directory.
    
    Searches for files containing kernel definitions:
    - Triton: @triton.jit, @triton.autotune
    - HIP/CUDA: __global__ void
    
    Args:
        directory: Directory to search
        max_results: Maximum number of results (default: 20)
    
    Returns:
        Dictionary with list of discovered kernels
    """
    path = Path(directory)
    if not path.exists():
        return {"error": f"Directory not found: {directory}"}
    
    kernels = []
    extensions = [".py", ".cu", ".hip", ".cpp"]
    
    for ext in extensions:
        for file_path in path.rglob(f"*{ext}"):
            if _should_skip(file_path):
                continue
            
            if _is_kernel_file(file_path):
                # Determine kernel type
                try:
                    content = file_path.read_text()[:2000]
                    if "@triton" in content:
                        kernel_type = "triton"
                    elif "__global__" in content and "hip" in content.lower():
                        kernel_type = "hip"
                    elif "__global__" in content:
                        kernel_type = "cuda"
                    else:
                        kernel_type = "unknown"
                except Exception:
                    kernel_type = "unknown"
                
                kernels.append({
                    "file": str(file_path),
                    "name": file_path.stem,
                    "type": kernel_type
                })
                
                if len(kernels) >= max_results:
                    break
    
    return {
        "directory": str(path),
        "kernels": kernels,
        "total_found": len(kernels)
    }


@mcp.tool()
def analyze_file(file_path: str) -> dict:
    """
    Analyze a file to determine its type.
    
    Checks if the file is a:
    - Kernel definition (triton, hip, cuda)
    - Test file (pytest, unittest, gtest)
    - Benchmark file (timing, performance)
    - None of the above
    
    Args:
        file_path: Path to the file to analyze
    
    Returns:
        Dictionary with file type and confidence scores
    """
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}
    
    result = {
        "file": str(path),
        "name": path.name,
        "is_kernel": False,
        "is_test": False,
        "is_benchmark": False,
        "kernel_type": None,
        "test_confidence": 0.0,
        "benchmark_confidence": 0.0
    }
    
    # Check if kernel
    if _is_kernel_file(path):
        result["is_kernel"] = True
        try:
            content = path.read_text()[:2000]
            if "@triton" in content:
                result["kernel_type"] = "triton"
            elif "__global__" in content and "hip" in content.lower():
                result["kernel_type"] = "hip"
            elif "__global__" in content:
                result["kernel_type"] = "cuda"
        except Exception:
            pass
        return result
    
    # Score as test
    test_score = _score_as_test(path)
    result["test_confidence"] = round(min(test_score, 1.0), 2)
    result["is_test"] = test_score >= 0.3
    
    # Score as benchmark
    bench_score = _score_as_bench(path)
    result["benchmark_confidence"] = round(min(bench_score, 1.0), 2)
    result["is_benchmark"] = bench_score >= 0.3
    
    if result["is_test"]:
        result["test_command"] = _get_test_command(path)
    
    return result


@mcp.tool()
def get_test_command(file_path: str) -> dict:
    """
    Get the command to run a test file.
    
    Detects the test framework and generates the appropriate command:
    - pytest: pytest <file> -v
    - unittest: python -m unittest <file>
    - script: python <file>
    
    Args:
        file_path: Path to the test file
    
    Returns:
        Dictionary with the command and detected test type
    """
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}
    
    try:
        content = path.read_text()[:2000]
    except Exception:
        content = ""
    
    if path.suffix == ".py":
        if "import pytest" in content or "@pytest" in content:
            test_type = "pytest"
            command = f"pytest {path} -v"
        elif "unittest" in content:
            test_type = "unittest"
            command = f"python -m unittest {path}"
        else:
            test_type = "script"
            command = f"python {path}"
    elif path.suffix in [".cpp", ".cc"]:
        test_type = "cpp"
        command = f"# Compile and run {path.name}"
    elif path.suffix in [".cu", ".hip"]:
        compiler = "hipcc" if path.suffix == ".hip" else "nvcc"
        test_type = "gpu"
        command = f"{compiler} {path} -o /tmp/{path.stem} && /tmp/{path.stem}"
    else:
        test_type = "unknown"
        command = f"# Unknown file type: {path.name}"
    
    return {
        "file": str(path),
        "test_type": test_type,
        "command": command
    }


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
