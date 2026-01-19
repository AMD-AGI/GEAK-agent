"""
verify-mcp: Functional Correctness Validation

Role: Ensure functional correctness is preserved.
- Baseline vs optimized output comparison
- Configurable absolute and relative tolerances
- Support for non-deterministic outputs (sorting, etc.)
- Detailed mismatch reporting

Used after: every optimization step before checkpointing.
"""

import subprocess
import tempfile
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set


@dataclass
class TensorComparison:
    """Result of comparing a single tensor."""
    name: str
    passed: bool
    status: str  # "passed", "shape_mismatch", "value_mismatch", "missing", "passed_relaxed"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifyResult:
    """Result from verification."""
    passed: bool
    comparisons: List[TensorComparison]
    mismatches: List[str]
    total_tensors: int
    passed_tensors: int
    error: Optional[str] = None


class VerifyTool:
    """
    MCP Tool for functional correctness validation.
    
    Compares outputs from optimized code against baseline
    reference outputs to ensure correctness is preserved.
    """
    
    # Tool metadata for MCP
    TOOL_NAME = "verify"
    TOOL_DESCRIPTION = """Verify functional correctness of optimized kernel.
    
Compares optimized outputs against baseline reference outputs.
Returns:
- Pass/fail status
- Per-tensor comparison results
- Detailed mismatch information"""
    
    TOOL_SCHEMA = {
        "type": "object",
        "properties": {
            "baseline_code": {
                "type": "string",
                "description": "The baseline kernel code"
            },
            "optimized_code": {
                "type": "string",
                "description": "The optimized kernel code to verify"
            },
            "output_tensors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Names of output tensors to compare"
            },
            "rtol": {
                "type": "number",
                "description": "Relative tolerance for float comparison",
                "default": 0.001
            },
            "atol": {
                "type": "number",
                "description": "Absolute tolerance for float comparison",
                "default": 0.001
            },
            "non_deterministic_outputs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tensor names with non-deterministic ordering (relaxed comparison)"
            }
        },
        "required": ["baseline_code", "optimized_code", "output_tensors"]
    }
    
    def __init__(self, docker_image: str = None, gpu_device: str = "3"):
        self.docker_image = docker_image or "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x"
        self.gpu_device = gpu_device
        self.work_dir = Path(tempfile.mkdtemp(prefix="verify_"))
    
    def verify(self, baseline_code: str,
               optimized_code: str,
               output_tensors: List[str],
               rtol: float = 0.001,
               atol: float = 0.001,
               non_deterministic_outputs: Optional[Set[str]] = None) -> VerifyResult:
        """
        Verify optimized code against baseline.
        
        This is the main entry point for the MCP tool.
        """
        non_deterministic = set(non_deterministic_outputs or [])
        
        # Generate verification script
        script = self._generate_verify_script(
            baseline_code, optimized_code, output_tensors,
            rtol, atol, non_deterministic
        )
        
        script_path = self.work_dir / "verify_kernel.py"
        script_path.write_text(script)
        
        # Run verification
        result = self._run_verify(script_path)
        
        if "error" in result:
            return VerifyResult(
                passed=False,
                comparisons=[],
                mismatches=[],
                total_tensors=len(output_tensors),
                passed_tensors=0,
                error=result["error"],
            )
        
        # Parse results
        comparisons = []
        mismatches = []
        passed_count = 0
        
        for check in result.get("checks", []):
            name = check["name"]
            status = check["status"]
            passed = status in ["passed", "passed_relaxed"]
            
            comparison = TensorComparison(
                name=name,
                passed=passed,
                status=status,
                details=check,
            )
            comparisons.append(comparison)
            
            if passed:
                passed_count += 1
            else:
                mismatches.append(name)
        
        return VerifyResult(
            passed=result.get("passed", False),
            comparisons=comparisons,
            mismatches=mismatches,
            total_tensors=len(output_tensors),
            passed_tensors=passed_count,
        )
    
    def _generate_verify_script(self, baseline_code: str,
                                optimized_code: str,
                                output_tensors: List[str],
                                rtol: float,
                                atol: float,
                                non_deterministic: Set[str]) -> str:
        """Generate the verification script."""
        tensors_list = json.dumps(output_tensors)
        non_det_list = json.dumps(list(non_deterministic))
        
        return f'''#!/usr/bin/env python3
"""Auto-generated verification script"""
import torch
import json

torch.manual_seed(42)
device = "cuda"

OUTPUT_TENSORS = {tensors_list}
NON_DETERMINISTIC = set({non_det_list})
RTOL = {rtol}
ATOL = {atol}

# ============================================================
# BASELINE CODE
# ============================================================
print("Running baseline...")
{baseline_code}

# Capture baseline outputs
baseline_outputs = {{}}
for name in OUTPUT_TENSORS:
    try:
        tensor = eval(name)
        baseline_outputs[name] = tensor.clone()
    except NameError:
        print(f"Warning: {{name}} not found in baseline")

torch.cuda.synchronize()

# Reset state for optimized run
torch.manual_seed(42)

# ============================================================
# OPTIMIZED CODE
# ============================================================
print("Running optimized...")
{optimized_code}

torch.cuda.synchronize()

# ============================================================
# COMPARISON
# ============================================================
print("Comparing outputs...")

results = {{
    "passed": True,
    "checks": [],
}}

for name in OUTPUT_TENSORS:
    if name not in baseline_outputs:
        results["checks"].append({{"name": name, "status": "missing_baseline"}})
        continue
    
    try:
        optimized_tensor = eval(name)
    except NameError:
        results["checks"].append({{"name": name, "status": "missing_optimized"}})
        results["passed"] = False
        continue
    
    baseline_tensor = baseline_outputs[name]
    
    # Shape check
    if optimized_tensor.shape != baseline_tensor.shape:
        results["checks"].append({{
            "name": name,
            "status": "shape_mismatch",
            "expected": list(baseline_tensor.shape),
            "got": list(optimized_tensor.shape),
        }})
        results["passed"] = False
        continue
    
    # Non-deterministic outputs get relaxed checking
    if name in NON_DETERMINISTIC:
        # Just check basic sanity
        if optimized_tensor.dtype in [torch.float32, torch.bfloat16, torch.float16]:
            has_nan = torch.isnan(optimized_tensor).any().item()
            has_inf = torch.isinf(optimized_tensor).any().item()
            if has_nan or has_inf:
                results["checks"].append({{"name": name, "status": "has_nan_or_inf"}})
                results["passed"] = False
            else:
                results["checks"].append({{"name": name, "status": "passed_relaxed"}})
        else:
            results["checks"].append({{"name": name, "status": "passed_relaxed"}})
        continue
    
    # Value comparison
    if optimized_tensor.dtype in [torch.int32, torch.int64, torch.int16, torch.int8]:
        # Integer: exact match
        if torch.equal(optimized_tensor, baseline_tensor):
            results["checks"].append({{"name": name, "status": "passed"}})
        else:
            diff_count = (optimized_tensor != baseline_tensor).sum().item()
            results["checks"].append({{
                "name": name,
                "status": "value_mismatch",
                "diff_count": diff_count,
                "total": optimized_tensor.numel(),
            }})
            results["passed"] = False
    else:
        # Float: allclose
        if torch.allclose(optimized_tensor, baseline_tensor, rtol=RTOL, atol=ATOL):
            results["checks"].append({{"name": name, "status": "passed"}})
        else:
            max_diff = (optimized_tensor.float() - baseline_tensor.float()).abs().max().item()
            results["checks"].append({{
                "name": name,
                "status": "value_mismatch",
                "max_diff": max_diff,
                "rtol": RTOL,
                "atol": ATOL,
            }})
            results["passed"] = False

print(f"Verification: {{'PASSED' if results['passed'] else 'FAILED'}}")

with open("/workspace/verify_result.json", "w") as f:
    json.dump(results, f, indent=2)
'''
    
    def _run_verify(self, script_path: Path) -> Dict[str, Any]:
        """Run verification in Docker."""
        cmd = [
            "docker", "run", "--rm",
            "--device=/dev/kfd", "--device=/dev/dri",
            "--ipc=host", "--group-add", "video",
            "-e", f"HIP_VISIBLE_DEVICES={self.gpu_device}",
            "-v", f"{self.work_dir}:/workspace",
            self.docker_image,
            "python3", f"/workspace/{script_path.name}"
        ]
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            
            # Load results
            result_path = self.work_dir / "verify_result.json"
            if result_path.exists():
                with open(result_path) as f:
                    return json.load(f)
            
            return {"error": result.stderr}
            
        except subprocess.TimeoutExpired:
            return {"error": "timeout"}
        except Exception as e:
            return {"error": str(e)}
    
    # MCP Tool interface methods
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return MCP tool definition."""
        return {
            "name": self.TOOL_NAME,
            "description": self.TOOL_DESCRIPTION,
            "inputSchema": self.TOOL_SCHEMA,
        }
    
    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given arguments."""
        result = self.verify(
            baseline_code=arguments["baseline_code"],
            optimized_code=arguments["optimized_code"],
            output_tensors=arguments["output_tensors"],
            rtol=arguments.get("rtol", 0.001),
            atol=arguments.get("atol", 0.001),
            non_deterministic_outputs=set(arguments.get("non_deterministic_outputs", [])),
        )
        
        return {
            "passed": result.passed,
            "total_tensors": result.total_tensors,
            "passed_tensors": result.passed_tensors,
            "mismatches": result.mismatches,
            "comparisons": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "status": c.status,
                }
                for c in result.comparisons
            ],
            "error": result.error,
        }


