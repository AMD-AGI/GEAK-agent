#!/usr/bin/env python3
"""
Simple Vector Add Kernel - Example for Mini-Kernel Agent

This is a basic Triton kernel that adds two vectors.
Use this as a starting point to understand how the agent optimizes kernels.

Usage:
    ./mini-kernel optimize examples/add_kernel/kernel.py --gpu 0
"""

import torch
import triton
import triton.language as tl
from dataclasses import dataclass
from typing import List, Dict, Any
import json


# =============================================================================
# TRITON KERNEL
# =============================================================================

@triton.jit
def add_kernel(
    x_ptr,      # Pointer to first input vector
    y_ptr,      # Pointer to second input vector
    output_ptr, # Pointer to output vector
    n_elements, # Number of elements
    BLOCK_SIZE: tl.constexpr,  # Block size (tunable parameter)
):
    """
    Simple vector addition kernel.
    
    Each program instance processes BLOCK_SIZE elements.
    """
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary handling
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute addition
    output = x + y
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """
    Add two tensors using the Triton kernel.
    
    Args:
        x: First input tensor
        y: Second input tensor
        block_size: Block size for the kernel (tunable)
        
    Returns:
        Sum of x and y
    """
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape
    
    x = x.contiguous()
    y = y.contiguous()
    
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    add_kernel[grid](
        x, y, output, n_elements,
        BLOCK_SIZE=block_size,
    )
    
    return output


# Export for the agent
triton_op = triton_add


def torch_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    return x + y


# Reference for correctness checking
torch_op = torch_add


# =============================================================================
# TEST HARNESS - Comprehensive Test Coverage
# =============================================================================

@dataclass
class TestCase:
    """A single test case definition."""
    name: str
    size: int
    dtype: torch.dtype
    description: str
    category: str  # "low", "medium", "high"


class TestHarness:
    """
    Comprehensive Test Harness for Kernel Validation
    
    Covers:
    - LOW: Edge cases, small sizes, boundary conditions
    - MEDIUM: Typical workloads, common configurations  
    - HIGH: Stress tests, large sizes, performance limits
    """
    
    # Test case definitions
    TEST_CASES: List[TestCase] = [
        # === LOW: Edge Cases & Small Sizes ===
        TestCase("tiny", 1, torch.float32, "Single element", "low"),
        TestCase("small", 16, torch.float32, "Sub-warp size", "low"),
        TestCase("warp", 64, torch.float32, "One warp", "low"),
        TestCase("block", 1024, torch.float32, "One block", "low"),
        TestCase("non_aligned", 1000, torch.float32, "Non-aligned size", "low"),
        TestCase("prime", 1009, torch.float32, "Prime number size", "low"),
        
        # === MEDIUM: Typical Workloads ===
        TestCase("typical_small", 64 * 1024, torch.float32, "64K elements", "medium"),
        TestCase("typical_med", 256 * 1024, torch.float32, "256K elements", "medium"),
        TestCase("typical_large", 1024 * 1024, torch.float32, "1M elements", "medium"),
        TestCase("fp16", 1024 * 1024, torch.float16, "FP16 precision", "medium"),
        TestCase("bf16", 1024 * 1024, torch.bfloat16, "BF16 precision", "medium"),
        
        # === HIGH: Stress Tests ===
        TestCase("large", 16 * 1024 * 1024, torch.float32, "16M elements", "high"),
        TestCase("xlarge", 64 * 1024 * 1024, torch.float32, "64M elements", "high"),
        TestCase("stress", 128 * 1024 * 1024, torch.float32, "128M elements", "high"),
    ]
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.results: List[Dict[str, Any]] = []
    
    def run_single_test(self, tc: TestCase) -> Dict[str, Any]:
        """Run a single test case."""
        result = {
            "name": tc.name,
            "category": tc.category,
            "size": tc.size,
            "dtype": str(tc.dtype),
            "passed": False,
            "max_diff": 0.0,
            "latency_us": 0.0,
            "error": None,
        }
        
        try:
            # Create test data
            x = torch.randn(tc.size, device=self.device, dtype=tc.dtype)
            y = torch.randn(tc.size, device=self.device, dtype=tc.dtype)
            
            # Run Triton kernel
            triton_result = triton_add(x, y)
            torch.cuda.synchronize()
            
            # Run reference
            torch_result = torch_add(x, y)
            
            # Check correctness
            if tc.dtype == torch.float16:
                rtol, atol = 1e-3, 1e-3
            elif tc.dtype == torch.bfloat16:
                rtol, atol = 1e-2, 1e-2
            else:
                rtol, atol = 1e-5, 1e-5
            
            max_diff = (triton_result - torch_result).abs().max().item()
            is_correct = torch.allclose(triton_result, torch_result, rtol=rtol, atol=atol)
            
            result["passed"] = is_correct
            result["max_diff"] = max_diff
            
            # Benchmark if correct
            if is_correct and tc.size >= 1024:
                # Warmup
                for _ in range(10):
                    _ = triton_add(x, y)
                torch.cuda.synchronize()
                
                # Time
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                iters = max(100, min(1000, 10_000_000 // tc.size))
                start.record()
                for _ in range(iters):
                    _ = triton_add(x, y)
                end.record()
                torch.cuda.synchronize()
                
                result["latency_us"] = start.elapsed_time(end) * 1000 / iters
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def run_all(self, categories: List[str] = None) -> Dict[str, Any]:
        """
        Run all test cases.
        
        Args:
            categories: List of categories to run ("low", "medium", "high")
                       If None, runs all categories.
        """
        if categories is None:
            categories = ["low", "medium", "high"]
        
        self.results = []
        
        print("=" * 70)
        print("  TEST HARNESS - Comprehensive Kernel Validation")
        print("=" * 70)
        print(f"  Categories: {', '.join(categories)}")
        print("=" * 70)
        
        for category in categories:
            tests = [tc for tc in self.TEST_CASES if tc.category == category]
            
            print(f"\n[{category.upper()}] Running {len(tests)} test cases...")
            print("-" * 50)
            
            for tc in tests:
                result = self.run_single_test(tc)
                self.results.append(result)
                
                status = "✓ PASS" if result["passed"] else "✗ FAIL"
                latency = f"{result['latency_us']:.2f} μs" if result["latency_us"] > 0 else "N/A"
                
                print(f"  {status} | {tc.name:15} | {tc.size:>12,} | {latency:>12} | {tc.description}")
                
                if result["error"]:
                    print(f"         Error: {result['error']}")
        
        # Summary
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed
        
        print("\n" + "=" * 70)
        print("  SUMMARY")
        print("=" * 70)
        print(f"  Total:  {total}")
        print(f"  Passed: {passed} ({100*passed/total:.1f}%)")
        print(f"  Failed: {failed}")
        
        # Category breakdown
        for cat in categories:
            cat_results = [r for r in self.results if r["category"] == cat]
            cat_passed = sum(1 for r in cat_results if r["passed"])
            print(f"  {cat.upper():8} {cat_passed}/{len(cat_results)} passed")
        
        print("=" * 70)
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0,
            "results": self.results,
        }
    
    def run_quick(self) -> Dict[str, Any]:
        """Run quick validation (low + medium only)."""
        return self.run_all(["low", "medium"])
    
    def run_full(self) -> Dict[str, Any]:
        """Run full validation including stress tests."""
        return self.run_all(["low", "medium", "high"])


# =============================================================================
# REQUIRED FUNCTIONS FOR AGENT
# =============================================================================

def run_baseline():
    """Run the kernel (called by agent's test harness)."""
    size = 1024 * 1024  # 1M elements
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    y = torch.randn(size, device='cuda', dtype=torch.float32)
    
    output = triton_add(x, y)
    torch.cuda.synchronize()
    
    return output


def check_correctness() -> Dict[str, Any]:
    """Verify kernel produces correct results."""
    harness = TestHarness()
    
    # Run low and medium tests for quick correctness check
    results = harness.run_all(["low", "medium"])
    
    return {
        "passed": results["passed"] == results["total"],
        "pass_rate": results["pass_rate"],
        "total_tests": results["total"],
        "passed_tests": results["passed"],
    }


def benchmark() -> Dict[str, Any]:
    """Run full benchmark suite."""
    harness = TestHarness()
    results = harness.run_full()
    
    # Calculate average latency for medium tests
    medium_results = [r for r in results["results"] 
                      if r["category"] == "medium" and r["latency_us"] > 0]
    
    avg_latency = sum(r["latency_us"] for r in medium_results) / len(medium_results) if medium_results else 0
    
    return {
        "passed": results["passed"] == results["total"],
        "avg_latency_us": avg_latency,
        "results": results,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  Vector Add Kernel - Example with Comprehensive Test Harness".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Parse command line
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            harness = TestHarness()
            harness.run_quick()
        elif sys.argv[1] == "--full":
            harness = TestHarness()
            harness.run_full()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python kernel.py [--quick|--full]")
    else:
        # Default: run quick tests
        harness = TestHarness()
        harness.run_quick()
    
    print()
    print("To optimize this kernel, run:")
    print("  ./mini-kernel optimize examples/add_kernel/kernel.py --gpu 0")
    print()
