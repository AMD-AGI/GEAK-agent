# Automated Test Discovery MCP

MCP server for automatically discovering tests and benchmarks for GPU kernels.

## Features

- **Content-based detection** - No configuration files needed
- **Auto-pattern learning** - Learns project-specific decorators and functions
- **Multi-language support** - Python (pytest, unittest), C++ (GTest), HIP, CUDA
- **Kernel name matching** - Prioritizes tests that match the kernel name

## Tools

### `discover_for_kernel`

Find tests and benchmarks for a GPU kernel file.

```python
result = discover_for_kernel(
    kernel_path="/path/to/kernel.py",
    max_results=10
)
# Returns:
# {
#     "kernel_name": "gemm_a16w16",
#     "kernel_type": "triton",
#     "tests": [{"name": "test_gemm_a16w16.py", "confidence": 1.0, "command": "pytest ..."}],
#     "benchmarks": [{"name": "bench_gemm_a16w16.py", "confidence": 1.0, "command": "python ..."}]
# }
```

### `discover_kernels`

Find all kernel files in a directory.

```python
result = discover_kernels(
    directory="/path/to/project",
    max_results=20
)
# Returns list of kernel files with types (triton, hip, cuda)
```

### `analyze_file`

Check if a file is a test, benchmark, or kernel.

```python
result = analyze_file(file_path="/path/to/file.py")
# Returns:
# {
#     "is_kernel": False,
#     "is_test": True,
#     "is_benchmark": False,
#     "test_confidence": 0.85,
#     "test_command": "pytest /path/to/file.py -v"
# }
```

### `get_test_command`

Get the command to run a test file.

```python
result = get_test_command(file_path="/path/to/test.py")
# Returns:
# {
#     "test_type": "pytest",
#     "command": "pytest /path/to/test.py -v"
# }
```

## Installation

```bash
pip install -e .
```

## Usage

### As MCP Server

```bash
automated-test-discovery
```

### In Cursor/Claude

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "automated-test-discovery": {
      "command": "automated-test-discovery"
    }
  }
}
```

## Detection Patterns

### Test Detection
- `import pytest`, `@pytest.mark`
- `def test_*`, `class Test*`
- `assert`, `.allclose()`, `assertEqual`
- `@perftest()`, `checkAllclose` (auto-detected)
- GTest: `TEST()`, `EXPECT_*`, `ASSERT_*`

### Benchmark Detection
- `TFLOPS`, `GFLOPS`, `latency`, `throughput`
- `torch.cuda.Event(enable_timing=True)`
- `triton.testing.do_bench`
- `warmup`, `speedup`, `GB/s`

### Kernel Detection
- `@triton.jit`, `@triton.autotune`
- `__global__ void` (HIP/CUDA)
- `tl.load`, `tl.store` (Triton)

## Test Results

Tested on 30 GPU kernels from the aiter repository:
- **Match rate: 90%** (27/30 kernels found correct tests)
- 3 kernels had no dedicated test files (expected behavior)
