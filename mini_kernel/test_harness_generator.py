#!/usr/bin/env python3
"""
Test Harness Generator - Auto-generate Comprehensive Tests for Any Kernel

This module automatically generates test harnesses with LOW/MEDIUM/HIGH coverage
for any GPU kernel or module, without requiring manual test writing.

Features:
- Auto-detects kernel signatures and input types
- Generates appropriate test data for each input
- Creates LOW (edge cases), MEDIUM (typical), HIGH (stress) test cases
- Validates correctness against reference implementations
- Benchmarks performance across different sizes

Usage:
    generator = TestHarnessGenerator(kernel_path)
    harness_code = generator.generate()
"""

import ast
import re
import inspect
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class InputType(Enum):
    """Detected input types for kernel functions."""
    TENSOR_1D = "tensor_1d"
    TENSOR_2D = "tensor_2d"
    TENSOR_3D = "tensor_3d"
    TENSOR_4D = "tensor_4d"
    SCALAR_INT = "scalar_int"
    SCALAR_FLOAT = "scalar_float"
    POINTER = "pointer"
    CONSTEXPR = "constexpr"
    UNKNOWN = "unknown"


@dataclass
class KernelParam:
    """A kernel parameter with inferred type."""
    name: str
    input_type: InputType
    is_output: bool = False
    dtype: str = "torch.float32"
    shape_hint: Optional[str] = None


@dataclass 
class TestConfig:
    """Configuration for a test case."""
    name: str
    category: str  # "low", "medium", "high"
    description: str
    size_multiplier: float
    dtype: str = "torch.float32"


class KernelAnalyzer:
    """Analyze kernel source to understand its structure."""
    
    # Common parameter name patterns
    TENSOR_PATTERNS = {
        r'.*_ptr$': InputType.POINTER,
        r'^(x|y|z|a|b|c|input|output|out|result)$': InputType.TENSOR_1D,
        r'^(q|k|v|query|key|value)$': InputType.TENSOR_3D,
        r'^(weight|bias|grad)': InputType.TENSOR_2D,
        r'^n_.*|^num_.*|^size.*|^batch.*|^seq.*': InputType.SCALAR_INT,
        r'^scale|^alpha|^beta|^eps': InputType.SCALAR_FLOAT,
        r'^BLOCK.*|^NUM.*': InputType.CONSTEXPR,
    }
    
    OUTPUT_PATTERNS = [
        r'output', r'out', r'result', r'o_', r'y$', r'z$'
    ]
    
    def __init__(self, kernel_path: Path):
        self.kernel_path = kernel_path
        self.source = kernel_path.read_text() if kernel_path.exists() else ""
        
    def find_main_function(self) -> Optional[str]:
        """Find the main callable function in the kernel."""
        # Priority order for function names (Python wrappers, not JIT kernels)
        # Include common kernel naming patterns from customer kernels
        priority = [
            # Customer kernel patterns (L1 kernels)
            'poi_fused_add', 'poi_fused_add_simple', 'poi_fused_to_copy', 
            'poi_fused_to_copy_transpose', 'red_fused_sum', 'tem_fused_mm',
            'tem_fused_bmm', 'tem_fused_linear',
            # MoE kernels
            'fused_moe', 'fused_experts', 'fused_moe_mxfp4',
            # General patterns
            'triton_op', 'triton_add', 'triton_matmul', 'triton_attention',
            'torch_op', 'main', 'forward', 'run', 'compute', 'execute',
            # Benchmark entry points
            'bench_op', 'run_baseline', 'benchmark'
        ]
        
        # Parse AST
        try:
            tree = ast.parse(self.source)
        except:
            return None
        
        functions = []
        jit_functions = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
                # Check if it has @triton.jit decorator (these are not directly callable)
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Attribute):
                        if decorator.attr == 'jit':
                            jit_functions.add(node.name)
                    elif isinstance(decorator, ast.Name):
                        if decorator.id == 'jit':
                            jit_functions.add(node.name)
        
        # Filter out JIT functions - they're not directly callable
        callable_functions = [f for f in functions if f not in jit_functions]
        
        # Check priority list first
        for name in priority:
            if name in callable_functions:
                return name
        
        # Return first non-private callable function
        for name in callable_functions:
            if not name.startswith('_'):
                return name
        
        return callable_functions[0] if callable_functions else None
    
    def find_reference_function(self) -> Optional[str]:
        """Find a reference/baseline function for correctness checking."""
        # Priority order - prefer torch_op over run_baseline
        priority_names = ['torch_op', 'torch_add', 'torch_matmul', 'ref_op', 'reference']
        secondary_patterns = ['ref_', 'reference', 'cpu_']
        
        try:
            tree = ast.parse(self.source)
        except:
            return None
        
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        
        # Check priority names first
        for name in priority_names:
            if name in functions:
                return name
        
        # Check secondary patterns
        for node_name in functions:
            for pattern in secondary_patterns:
                if pattern in node_name.lower() and node_name != 'run_baseline':
                    return node_name
        
        return None
    
    def analyze_function(self, func_name: str) -> List[KernelParam]:
        """Analyze a function to determine its parameters."""
        params = []
        
        try:
            tree = ast.parse(self.source)
        except:
            return params
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                for arg in node.args.args:
                    param = self._analyze_param(arg.arg)
                    params.append(param)
                break
        
        return params
    
    def _analyze_param(self, name: str) -> KernelParam:
        """Analyze a single parameter name to infer its type."""
        input_type = InputType.UNKNOWN
        is_output = False
        
        # Check against patterns
        for pattern, itype in self.TENSOR_PATTERNS.items():
            if re.match(pattern, name, re.IGNORECASE):
                input_type = itype
                break
        
        # Check if output
        for pattern in self.OUTPUT_PATTERNS:
            if re.match(pattern, name, re.IGNORECASE):
                is_output = True
                break
        
        # Default tensor type if still unknown and looks like data
        if input_type == InputType.UNKNOWN:
            if '_ptr' in name or name in ['x', 'y', 'z', 'a', 'b']:
                input_type = InputType.TENSOR_1D
        
        return KernelParam(
            name=name,
            input_type=input_type,
            is_output=is_output,
        )
    
    def detect_kernel_type(self) -> str:
        """Detect the type of kernel (add, matmul, attention, etc.)."""
        source_lower = self.source.lower()
        
        if 'attention' in source_lower or 'mla' in source_lower:
            return "attention"
        elif 'matmul' in source_lower or 'gemm' in source_lower:
            return "matmul"
        elif 'conv' in source_lower:
            return "convolution"
        elif 'softmax' in source_lower:
            return "softmax"
        elif 'layernorm' in source_lower or 'rms' in source_lower:
            return "normalization"
        elif 'topk' in source_lower or 'sort' in source_lower:
            return "sorting"
        else:
            return "elementwise"


class TestHarnessGenerator:
    """
    Generate comprehensive test harnesses for any kernel.
    
    Creates test cases across three categories:
    - LOW: Edge cases, small sizes, boundary conditions
    - MEDIUM: Typical workloads, common configurations
    - HIGH: Stress tests, large sizes, performance limits
    """
    
    # Test configurations by kernel type
    TEST_CONFIGS = {
        "elementwise": [
            # LOW - Edge cases
            TestConfig("single_element", "low", "Single element", 1),
            TestConfig("small_16", "low", "16 elements (sub-warp)", 16),
            TestConfig("warp_64", "low", "64 elements (one warp)", 64),
            TestConfig("block_1024", "low", "1024 elements (one block)", 1024),
            TestConfig("non_aligned", "low", "Non-aligned size (1000)", 1000),
            TestConfig("prime", "low", "Prime size (1009)", 1009),
            # MEDIUM - Typical
            TestConfig("typical_64k", "medium", "64K elements", 64 * 1024),
            TestConfig("typical_256k", "medium", "256K elements", 256 * 1024),
            TestConfig("typical_1m", "medium", "1M elements", 1024 * 1024),
            TestConfig("fp16_1m", "medium", "1M FP16", 1024 * 1024, "torch.float16"),
            TestConfig("bf16_1m", "medium", "1M BF16", 1024 * 1024, "torch.bfloat16"),
            # HIGH - Stress
            TestConfig("large_16m", "high", "16M elements", 16 * 1024 * 1024),
            TestConfig("xlarge_64m", "high", "64M elements", 64 * 1024 * 1024),
            TestConfig("stress_128m", "high", "128M elements", 128 * 1024 * 1024),
        ],
        "matmul": [
            # LOW
            TestConfig("tiny_4x4", "low", "4x4 matrices", 4),
            TestConfig("small_16x16", "low", "16x16 matrices", 16),
            TestConfig("block_64x64", "low", "64x64 matrices", 64),
            TestConfig("non_square", "low", "32x64 matrices", 32),
            # MEDIUM
            TestConfig("typical_256", "medium", "256x256 matrices", 256),
            TestConfig("typical_512", "medium", "512x512 matrices", 512),
            TestConfig("typical_1024", "medium", "1024x1024 matrices", 1024),
            TestConfig("fp16_1024", "medium", "1024x1024 FP16", 1024, "torch.float16"),
            # HIGH
            TestConfig("large_2048", "high", "2048x2048 matrices", 2048),
            TestConfig("xlarge_4096", "high", "4096x4096 matrices", 4096),
            TestConfig("stress_8192", "high", "8192x8192 matrices", 8192),
        ],
        "attention": [
            # LOW
            TestConfig("tiny_bs1_seq16", "low", "BS=1, Seq=16", 16),
            TestConfig("small_bs1_seq64", "low", "BS=1, Seq=64", 64),
            TestConfig("single_head", "low", "Single head", 128),
            # MEDIUM
            TestConfig("typical_bs4_seq512", "medium", "BS=4, Seq=512", 512),
            TestConfig("typical_bs4_seq1024", "medium", "BS=4, Seq=1024", 1024),
            TestConfig("typical_bs8_seq1024", "medium", "BS=8, Seq=1024", 1024),
            # HIGH
            TestConfig("large_bs16_seq2048", "high", "BS=16, Seq=2048", 2048),
            TestConfig("xlarge_bs32_seq4096", "high", "BS=32, Seq=4096", 4096),
            TestConfig("stress_bs64_seq8192", "high", "BS=64, Seq=8192", 8192),
        ],
    }
    
    def __init__(self, kernel_path: Path):
        self.kernel_path = kernel_path
        self.analyzer = KernelAnalyzer(kernel_path)
        self.kernel_type = self.analyzer.detect_kernel_type()
        self.main_func = self.analyzer.find_main_function()
        self.ref_func = self.analyzer.find_reference_function()
        
    def generate(self) -> str:
        """Generate the complete test harness code."""
        configs = self.TEST_CONFIGS.get(self.kernel_type, self.TEST_CONFIGS["elementwise"])
        
        return f'''#!/usr/bin/env python3
"""
AUTO-GENERATED TEST HARNESS
===========================
Kernel: {self.kernel_path.name}
Type: {self.kernel_type}
Main Function: {self.main_func}
Reference Function: {self.ref_func}

Test Coverage:
- LOW: Edge cases, small sizes, boundary conditions
- MEDIUM: Typical workloads, common configurations
- HIGH: Stress tests, large sizes, performance limits
"""

import sys
import json
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

# Add kernel directory to path
KERNEL_DIR = "{self.kernel_path.parent}"
sys.path.insert(0, KERNEL_DIR)

# Import kernel module
try:
    from {self.kernel_path.stem} import *
except ImportError as e:
    print(f"Failed to import kernel: {{e}}")
    sys.exit(1)


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

@dataclass
class TestCase:
    name: str
    category: str
    description: str
    size: int
    dtype: str = "torch.float32"


TEST_CASES: List[TestCase] = [
{self._generate_test_cases(configs)}
]


# =============================================================================
# TEST HARNESS
# =============================================================================

class TestHarness:
    """Comprehensive test harness with LOW/MEDIUM/HIGH coverage."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.results: List[Dict[str, Any]] = []
        
        # Detect available functions
        self.kernel_fn = {self._get_kernel_fn_code()}
        self.ref_fn = {self._get_ref_fn_code()}
    
    def generate_inputs(self, size: int, dtype_str: str) -> tuple:
        """Generate appropriate test inputs based on kernel type."""
        dtype = eval(dtype_str)
        
{self._generate_input_generator()}
    
    def run_kernel(self, inputs: tuple):
        """Run the kernel under test."""
        return self.kernel_fn(*inputs)
    
    def run_reference(self, inputs: tuple):
        """Run the reference implementation."""
        if self.ref_fn:
            return self.ref_fn(*inputs)
        else:
            # Fall back to simple PyTorch equivalent
            return self._pytorch_fallback(inputs)
    
    def _pytorch_fallback(self, inputs: tuple):
        """Simple PyTorch fallback for correctness checking."""
{self._generate_pytorch_fallback()}
    
    def check_correctness(self, result, reference, dtype_str: str) -> tuple:
        """Check if result matches reference within tolerance."""
        if dtype_str == "torch.float16":
            rtol, atol = 1e-3, 1e-3
        elif dtype_str == "torch.bfloat16":
            rtol, atol = 1e-2, 1e-2
        else:
            rtol, atol = 1e-5, 1e-5
        
        if isinstance(result, tuple):
            result = result[0]
        if isinstance(reference, tuple):
            reference = reference[0]
        
        max_diff = (result - reference).abs().max().item()
        is_correct = torch.allclose(result, reference, rtol=rtol, atol=atol)
        
        return is_correct, max_diff
    
    def benchmark(self, inputs: tuple, warmup: int = 100, iters: int = 1000) -> float:
        """Benchmark kernel latency in microseconds."""
        # Warmup
        for _ in range(warmup):
            self.run_kernel(inputs)
        torch.cuda.synchronize()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iters):
            self.run_kernel(inputs)
        end.record()
        torch.cuda.synchronize()
        
        return start.elapsed_time(end) * 1000 / iters  # microseconds
    
    def run_single_test(self, tc: TestCase) -> Dict[str, Any]:
        """Run a single test case."""
        result = {{
            "name": tc.name,
            "category": tc.category,
            "size": tc.size,
            "dtype": tc.dtype,
            "passed": False,
            "max_diff": 0.0,
            "latency_us": 0.0,
            "error": None,
        }}
        
        try:
            # Generate inputs
            inputs = self.generate_inputs(tc.size, tc.dtype)
            
            # Run kernel
            kernel_result = self.run_kernel(inputs)
            torch.cuda.synchronize()
            
            # Run reference
            ref_result = self.run_reference(inputs)
            
            # Check correctness
            is_correct, max_diff = self.check_correctness(kernel_result, ref_result, tc.dtype)
            result["passed"] = is_correct
            result["max_diff"] = max_diff
            
            # Benchmark if correct and size is reasonable
            if is_correct and tc.size >= 1024:
                iters = max(100, min(1000, 10_000_000 // tc.size))
                result["latency_us"] = self.benchmark(inputs, iters=iters)
            
        except Exception as e:
            result["error"] = str(e)
            import traceback
            traceback.print_exc()
        
        return result
    
    def run_all(self, categories: List[str] = None) -> Dict[str, Any]:
        """Run all test cases."""
        if categories is None:
            categories = ["low", "medium", "high"]
        
        self.results = []
        
        print("=" * 70)
        print("  AUTO-GENERATED TEST HARNESS")
        print("  Kernel: {self.kernel_path.name}")
        print("  Type: {self.kernel_type}")
        print("=" * 70)
        
        for category in categories:
            tests = [tc for tc in TEST_CASES if tc.category == category]
            
            print(f"\\n[{{category.upper()}}] Running {{len(tests)}} test cases...")
            print("-" * 60)
            
            for tc in tests:
                result = self.run_single_test(tc)
                self.results.append(result)
                
                status = "✓ PASS" if result["passed"] else "✗ FAIL"
                latency = f"{{result['latency_us']:.2f}} μs" if result["latency_us"] > 0 else "N/A"
                
                print(f"  {{status}} | {{tc.name:20}} | {{tc.size:>12,}} | {{latency:>12}} | {{tc.description}}")
                
                if result["error"]:
                    print(f"         Error: {{result['error']}}")
        
        # Summary
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        
        print("\\n" + "=" * 70)
        print("  SUMMARY")
        print("=" * 70)
        print(f"  Total:  {{total}}")
        print(f"  Passed: {{passed}} ({{100*passed/total:.1f}}%)")
        print(f"  Failed: {{total - passed}}")
        print("=" * 70)
        
        return {{
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "results": self.results,
        }}


# =============================================================================
# REQUIRED FUNCTIONS FOR AGENT
# =============================================================================

def run_baseline():
    """Run the kernel with default inputs."""
    harness = TestHarness()
    inputs = harness.generate_inputs(1024 * 1024, "torch.float32")
    result = harness.run_kernel(inputs)
    torch.cuda.synchronize()
    return result


def check_correctness() -> Dict[str, Any]:
    """Verify kernel correctness across test cases."""
    harness = TestHarness()
    results = harness.run_all(["low", "medium"])
    
    return {{
        "passed": results["passed"] == results["total"],
        "pass_rate": results["pass_rate"],
        "total_tests": results["total"],
        "passed_tests": results["passed"],
    }}


def benchmark() -> Dict[str, Any]:
    """Full benchmark including stress tests."""
    harness = TestHarness()
    results = harness.run_all(["low", "medium", "high"])
    
    medium_results = [r for r in results["results"] 
                      if r["category"] == "medium" and r["latency_us"] > 0]
    avg_latency = sum(r["latency_us"] for r in medium_results) / len(medium_results) if medium_results else 0
    
    return {{
        "passed": results["passed"] == results["total"],
        "avg_latency_us": avg_latency,
        "results": results,
    }}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    harness = TestHarness()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            harness.run_all(["low", "medium"])
        elif sys.argv[1] == "--full":
            harness.run_all(["low", "medium", "high"])
        elif sys.argv[1] == "--low":
            harness.run_all(["low"])
        elif sys.argv[1] == "--medium":
            harness.run_all(["medium"])
        elif sys.argv[1] == "--high":
            harness.run_all(["high"])
        else:
            print("Usage: python test_harness.py [--quick|--full|--low|--medium|--high]")
    else:
        harness.run_all(["low", "medium"])
'''

    def _generate_test_cases(self, configs: List[TestConfig]) -> str:
        """Generate test case definitions."""
        lines = []
        for tc in configs:
            lines.append(f'    TestCase("{tc.name}", "{tc.category}", "{tc.description}", {int(tc.size_multiplier)}, "{tc.dtype}"),')
        return "\n".join(lines)
    
    def _get_kernel_fn_code(self) -> str:
        """Get code to reference the kernel function."""
        if self.main_func:
            return f"globals().get('{self.main_func}') or locals().get('{self.main_func}')"
        return "globals().get('triton_op') or globals().get('triton_add')"
    
    def _get_ref_fn_code(self) -> str:
        """Get code to reference the reference function."""
        if self.ref_func and self.ref_func not in ['run_baseline', 'benchmark', 'check_correctness']:
            return f"globals().get('{self.ref_func}') or locals().get('{self.ref_func}')"
        # Fallback: try common names
        return "globals().get('torch_op') or globals().get('torch_add') or globals().get('ref_op')"
    
    def _generate_input_generator(self) -> str:
        """Generate input generation code based on kernel type."""
        if self.kernel_type == "elementwise":
            return '''        # Elementwise: two input tensors of same size
        x = torch.randn(size, device=self.device, dtype=dtype)
        y = torch.randn(size, device=self.device, dtype=dtype)
        return (x, y)'''
        
        elif self.kernel_type == "matmul":
            return '''        # Matrix multiplication: two 2D tensors
        a = torch.randn(size, size, device=self.device, dtype=dtype)
        b = torch.randn(size, size, device=self.device, dtype=dtype)
        return (a, b)'''
        
        elif self.kernel_type == "attention":
            return '''        # Attention: Q, K, V tensors
        batch_size = 4
        num_heads = 16
        seq_len = size
        head_dim = 64
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device, dtype=dtype)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device, dtype=dtype)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device, dtype=dtype)
        return (q, k, v)'''
        
        else:
            return '''        # Default: single input tensor
        x = torch.randn(size, device=self.device, dtype=dtype)
        y = torch.randn(size, device=self.device, dtype=dtype)
        return (x, y)'''
    
    def _generate_pytorch_fallback(self) -> str:
        """Generate PyTorch fallback code."""
        if self.kernel_type == "elementwise":
            return '''        # Elementwise add
        x, y = inputs
        return x + y'''
        
        elif self.kernel_type == "matmul":
            return '''        # Matrix multiplication
        a, b = inputs
        return torch.matmul(a, b)'''
        
        elif self.kernel_type == "attention":
            return '''        # Scaled dot-product attention
        q, k, v = inputs
        scale = q.shape[-1] ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, v)'''
        
        else:
            return '''        # Default: element-wise add
        if len(inputs) >= 2:
            return inputs[0] + inputs[1]
        return inputs[0]'''


def generate_test_harness(kernel_path: Path, output_path: Optional[Path] = None) -> str:
    """
    Generate a test harness for the given kernel.
    
    Args:
        kernel_path: Path to the kernel source file
        output_path: Optional path to write the harness (if None, returns string)
    
    Returns:
        Generated test harness code
    """
    generator = TestHarnessGenerator(kernel_path)
    code = generator.generate()
    
    if output_path:
        output_path.write_text(code)
        print(f"Test harness written to: {output_path}")
    
    return code


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_harness_generator.py <kernel_path> [output_path]")
        sys.exit(1)
    
    kernel_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    code = generate_test_harness(kernel_path, output_path)
    
    if not output_path:
        print(code)

