"""Discovery Pipeline - Runs before the agent loop.

This module handles:
1. Kernel discovery - Find kernel files in the workspace
2. Test discovery - Find existing tests or prompt to create (content-based)
3. Benchmark discovery - Find performance benchmarks (content-based)
4. User confirmation - Interactive prompts to confirm/edit
5. LLM-assisted analysis - Optional LLM for smarter discovery

Discovery Modes:
- User provides: --test "pytest..." --bench "python..."
- Agent discovers: Search repo for test files, confirm with user
- Agent creates: (TODO) Help create tests if none exist

Content-based detection keywords:
- Tests: assert, pytest, allclose, correctness, assertEqual, expect
- Benchmarks: elapsed_time, latency, throughput, TFLOPS, benchmark, warmup
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Try to import LLM support
try:
    import anthropic
    HAS_LLM = True
except ImportError:
    HAS_LLM = False


@dataclass
class KernelInfo:
    """Information about a discovered kernel."""
    file_path: Path
    kernel_name: str
    kernel_type: str  # triton, hip, cuda
    function_names: list[str] = field(default_factory=list)
    has_jit_decorator: bool = False
    has_autotune: bool = False


@dataclass
class TestInfo:
    """Information about a discovered test."""
    file_path: Path
    test_type: str  # pytest, script, makefile
    command: str  # Command to run the test
    confidence: float  # 0-1, how confident we are this is the right test


@dataclass
class BenchmarkInfo:
    """Information about a discovered benchmark."""
    file_path: Path
    bench_type: str  # pytest, script, custom
    command: str
    confidence: float


@dataclass
class DiscoveryResult:
    """Result of the discovery pipeline."""
    kernels: list[KernelInfo] = field(default_factory=list)
    tests: list[TestInfo] = field(default_factory=list)
    benchmarks: list[BenchmarkInfo] = field(default_factory=list)
    workspace_path: Path = None
    needs_user_confirmation: bool = True
    user_provided_test: Optional[str] = None
    user_provided_bench: Optional[str] = None


class DiscoveryPipeline:
    """
    Discovery pipeline that runs before the agent loop.
    
    Progressive discovery:
    1. Check if user provided explicit commands
    2. If not, search for test/bench files (CONTENT-BASED)
    3. Present findings to user for confirmation
    4. Offer to create tests if none found (TODO)
    5. Optionally use LLM for smarter analysis
    """
    
    # Patterns for finding kernel files
    KERNEL_PATTERNS = [
        r"@triton\.jit",
        r"@triton\.autotune",
        r"__global__\s+void",  # CUDA/HIP
        r"tl\.load|tl\.store",  # Triton ops
    ]
    
    # Content keywords for TEST detection (more reliable than filename)
    TEST_CONTENT_KEYWORDS = [
        # Pytest
        (r"import pytest", 0.3),
        (r"@pytest\.mark", 0.3),
        (r"def test_\w+\s*\(", 0.4),
        # Assertions
        (r"assert\s+", 0.2),
        (r"\.allclose\(", 0.3),
        (r"\.assertEqual\(", 0.2),
        (r"torch\.testing\.assert", 0.3),
        # Correctness checking
        (r"correctness", 0.2),
        (r"verify|verification", 0.15),
        (r"expected.*actual|actual.*expected", 0.2),
        (r"reference.*output|output.*reference", 0.2),
        # Test structure
        (r"class Test\w+", 0.3),
        (r"unittest", 0.2),
    ]
    
    # Content keywords for BENCHMARK detection
    BENCH_CONTENT_KEYWORDS = [
        # Timing
        (r"elapsed_time|elapsed", 0.3),
        (r"latency", 0.25),
        (r"throughput", 0.25),
        (r"TFLOPS|GFLOPS|TFLOPs|GFLOPs", 0.4),
        (r"us/iter|ms/iter|μs", 0.3),
        # Benchmarking patterns
        (r"warmup|warm_up|warm-up", 0.25),
        (r"benchmark|bench_", 0.3),
        (r"time\.time\(\)|time\.perf_counter", 0.2),
        (r"torch\.cuda\.Event\(enable_timing", 0.4),
        (r"start\.record\(\)|end\.record\(\)", 0.35),
        (r"triton\.testing\.do_bench", 0.5),
        # Performance reporting
        (r"speedup", 0.25),
        (r"GB/s|TB/s", 0.3),
        (r"bandwidth", 0.2),
        (r"median|percentile|p50|p99", 0.2),
    ]
    
    # Filename patterns (lower priority than content)
    TEST_FILE_PATTERNS = [
        "test_*.py",
        "*_test.py",
        "tests/*.py",
        "test/*.py",
    ]
    
    BENCH_FILE_PATTERNS = [
        "bench*.py",
        "*benchmark*.py",
        "*_perf.py",
        "perf_*.py",
    ]
    
    def __init__(self, workspace_path: Path = None, use_llm: bool = False):
        self.workspace = Path(workspace_path) if workspace_path else Path.cwd()
        self.result = DiscoveryResult(workspace_path=self.workspace)
        self.use_llm = use_llm and HAS_LLM
        self._llm_client = None
        
        if self.use_llm:
            self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM client for smart discovery."""
        api_key = os.environ.get("AMD_LLM_API_KEY") or os.environ.get("LLM_GATEWAY_KEY")
        if not api_key:
            self.use_llm = False
            return
        
        try:
            self._llm_client = anthropic.Anthropic(
                api_key="dummy",
                base_url="https://llm-api.amd.com/Anthropic",
                default_headers={
                    "Ocp-Apim-Subscription-Key": api_key,
                    "anthropic-version": "2023-10-16",
                },
            )
        except Exception:
            self.use_llm = False
    
    def _llm_analyze_file(self, file_path: Path, file_type: str) -> Optional[dict]:
        """Use LLM to analyze a file when content-based detection is uncertain."""
        if not self.use_llm or not self._llm_client:
            return None
        
        try:
            content = file_path.read_text()[:3000]  # Limit content size
        except Exception:
            return None
        
        prompt = f"""Analyze this Python file and determine if it's a {file_type}.

File: {file_path.name}
Content (first 3000 chars):
```python
{content}
```

Respond with JSON only:
{{
    "is_{file_type}": true/false,
    "confidence": 0.0-1.0,
    "reason": "brief explanation",
    "command": "how to run this file (e.g., 'pytest file.py' or 'python file.py')"
}}
"""
        
        try:
            response = self._llm_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            
            import json
            result_text = response.content[0].text
            # Extract JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            return json.loads(result_text.strip())
        except Exception:
            return None
    
    def run(
        self,
        kernel_path: Optional[Path] = None,
        test_command: Optional[str] = None,
        bench_command: Optional[str] = None,
        interactive: bool = True
    ) -> DiscoveryResult:
        """
        Run the full discovery pipeline.
        
        Args:
            kernel_path: Explicit path to kernel file/directory
            test_command: User-provided test command (skips discovery)
            bench_command: User-provided benchmark command (skips discovery)
            interactive: Whether to prompt user for confirmation
        
        Returns:
            DiscoveryResult with all discovered information
        """
        print("\n" + "=" * 60)
        print("  DISCOVERY PIPELINE")
        print("=" * 60)
        
        # Store user-provided commands
        self.result.user_provided_test = test_command
        self.result.user_provided_bench = bench_command
        
        # If kernel_path is a file, expand workspace to find related tests
        self._kernel_file = None
        if kernel_path and kernel_path.is_file():
            self._kernel_file = kernel_path
            # Search for workspace root (look for common markers)
            self._expand_workspace_for_file(kernel_path)
        
        # Step 1: Discover kernels
        self._discover_kernels(kernel_path)
        
        # Step 2: Discover tests (unless user provided)
        if test_command:
            self.result.tests.append(TestInfo(
                file_path=Path("user-provided"),
                test_type="user",
                command=test_command,
                confidence=1.0
            ))
            self.result.needs_user_confirmation = False
        else:
            self._discover_tests()
        
        # Step 3: Discover benchmarks (unless user provided)
        if bench_command:
            self.result.benchmarks.append(BenchmarkInfo(
                file_path=Path("user-provided"),
                bench_type="user",
                command=bench_command,
                confidence=1.0
            ))
        else:
            self._discover_benchmarks()
        
        # Step 4: Display findings
        self._display_findings()
        
        # Step 5: User confirmation (if interactive)
        if interactive and self.result.needs_user_confirmation:
            self._prompt_user_confirmation()
        
        return self.result
    
    def _expand_workspace_for_file(self, kernel_file: Path):
        """Expand workspace when given a single file to find related tests."""
        # Look for common project root markers
        markers = ["pyproject.toml", "setup.py", "setup.cfg", ".git", "op_tests", "tests"]
        
        current = kernel_file.parent
        for _ in range(10):  # Max 10 levels up
            for marker in markers:
                if (current / marker).exists():
                    self.workspace = current
                    self.result.workspace_path = current
                    print(f"      Expanded workspace to: {current}")
                    return
            
            parent = current.parent
            if parent == current:  # Reached root
                break
            current = parent
    
    def _discover_kernels(self, kernel_path: Optional[Path] = None):
        """Discover kernel files in the workspace."""
        print("\n[1/4] Discovering kernels...")
        
        search_path = kernel_path or self.workspace
        
        if search_path.is_file():
            # Single file provided
            kernel_info = self._analyze_kernel_file(search_path)
            if kernel_info:
                self.result.kernels.append(kernel_info)
        else:
            # Search directory
            for py_file in search_path.rglob("*.py"):
                # Skip test files and common non-kernel dirs
                if self._should_skip_file(py_file):
                    continue
                
                kernel_info = self._analyze_kernel_file(py_file)
                if kernel_info:
                    self.result.kernels.append(kernel_info)
        
        print(f"      Found {len(self.result.kernels)} kernel(s)")
    
    def _analyze_kernel_file(self, file_path: Path) -> Optional[KernelInfo]:
        """Analyze a file to determine if it contains kernels."""
        try:
            content = file_path.read_text()
        except Exception:
            return None
        
        # Check for kernel patterns
        kernel_type = None
        has_jit = False
        has_autotune = False
        
        if "@triton.jit" in content or "tl.load" in content:
            kernel_type = "triton"
            has_jit = "@triton.jit" in content
            has_autotune = "@triton.autotune" in content
        elif "__global__" in content and ("__device__" in content or "hipLaunch" in content):
            kernel_type = "hip"
        elif "__global__" in content and "cuda" in content.lower():
            kernel_type = "cuda"
        
        if not kernel_type:
            return None
        
        # Extract function names
        function_names = []
        
        # Find @triton.jit decorated functions
        jit_pattern = r"@triton\.jit\s*\n\s*def\s+(\w+)"
        for match in re.finditer(jit_pattern, content):
            function_names.append(match.group(1))
        
        # Find wrapper functions (triton_op, run_baseline, etc.)
        wrapper_pattern = r"def\s+(triton_op|run_baseline|torch_op|forward|main)\s*\("
        for match in re.finditer(wrapper_pattern, content):
            if match.group(1) not in function_names:
                function_names.append(match.group(1))
        
        # Extract kernel name from first JIT function or file name
        kernel_name = function_names[0] if function_names else file_path.stem
        
        return KernelInfo(
            file_path=file_path,
            kernel_name=kernel_name,
            kernel_type=kernel_type,
            function_names=function_names,
            has_jit_decorator=has_jit,
            has_autotune=has_autotune
        )
    
    def _discover_tests(self):
        """Discover test files in the workspace (content-based)."""
        print("\n[2/4] Discovering tests (content-based)...")
        
        seen_paths = set()
        
        # Priority 1: Files matching kernel name
        if self.result.kernels:
            for kernel in self.result.kernels:
                kernel_name = kernel.kernel_name
                for py_file in self.workspace.rglob(f"*{kernel_name}*.py"):
                    if self._should_skip_file(py_file) or py_file in seen_paths:
                        continue
                    test_info = self._analyze_test_file(py_file)
                    if test_info:
                        self.result.tests.append(test_info)
                        seen_paths.add(py_file)
        
        # Priority 2: Files in test directories or with test in name
        test_dirs = ["test", "tests", "op_tests", "unit_tests"]
        for py_file in self.workspace.rglob("*.py"):
            if self._should_skip_file(py_file) or py_file in seen_paths:
                continue
            
            # Check if in test directory or has test in name
            in_test_dir = any(d in py_file.parts for d in test_dirs)
            has_test_name = "test" in py_file.name.lower()
            
            if in_test_dir or has_test_name:
                test_info = self._analyze_test_file(py_file)
                if test_info:
                    self.result.tests.append(test_info)
                    seen_paths.add(py_file)
        
        # Sort by confidence
        self.result.tests.sort(key=lambda t: t.confidence, reverse=True)
        
        # Limit results
        if len(self.result.tests) > 10:
            print(f"      Found {len(self.result.tests)} potential test(s), showing top 10")
            self.result.tests = self.result.tests[:10]
        else:
            print(f"      Found {len(self.result.tests)} potential test(s)")
    
    def _analyze_test_file(self, file_path: Path) -> Optional[TestInfo]:
        """Analyze a file to determine if it's a test (content-based)."""
        try:
            content = file_path.read_text()
        except Exception:
            return None
        
        confidence = 0.0
        test_type = "script"
        
        # Content-based scoring
        for pattern, score in self.TEST_CONTENT_KEYWORDS:
            if re.search(pattern, content, re.IGNORECASE):
                confidence += score
        
        # Filename bonus (lower priority than content)
        if "test" in file_path.name.lower():
            confidence += 0.1
        
        # Kernel name match bonus
        for kernel in self.result.kernels:
            if kernel.kernel_name.lower() in file_path.name.lower():
                confidence += 0.15
                break
        
        # Must have minimum confidence to be considered a test
        if confidence < 0.3:
            return None
        
        # Determine test type
        if "import pytest" in content or "@pytest" in content:
            test_type = "pytest"
        elif "unittest" in content:
            test_type = "unittest"
        
        # Generate command
        if test_type == "pytest":
            command = f"pytest {file_path} -v"
        elif test_type == "unittest":
            command = f"python -m unittest {file_path}"
        else:
            command = f"python {file_path}"
        
        return TestInfo(
            file_path=file_path,
            test_type=test_type,
            command=command,
            confidence=min(confidence, 1.0)
        )
    
    def _discover_benchmarks(self):
        """Discover benchmark files in the workspace (content-based)."""
        print("\n[3/4] Discovering benchmarks (content-based)...")
        
        seen_paths = set()
        
        # Priority 1: Files matching kernel name with bench/perf keywords
        if self.result.kernels:
            for kernel in self.result.kernels:
                kernel_name = kernel.kernel_name
                for py_file in self.workspace.rglob(f"*{kernel_name}*.py"):
                    if self._should_skip_file(py_file) or py_file in seen_paths:
                        continue
                    bench_info = self._analyze_bench_file(py_file)
                    if bench_info:
                        self.result.benchmarks.append(bench_info)
                        seen_paths.add(py_file)
        
        # Priority 2: Files in benchmark directories or with bench/perf in name
        bench_dirs = ["bench", "benchmark", "benchmarks", "op_benchmarks", "perf"]
        for py_file in self.workspace.rglob("*.py"):
            if self._should_skip_file(py_file) or py_file in seen_paths:
                continue
            
            # Check if in bench directory or has bench/perf in name
            in_bench_dir = any(d in py_file.parts for d in bench_dirs)
            has_bench_name = "bench" in py_file.name.lower() or "perf" in py_file.name.lower()
            
            if in_bench_dir or has_bench_name:
                bench_info = self._analyze_bench_file(py_file)
                if bench_info:
                    self.result.benchmarks.append(bench_info)
                    seen_paths.add(py_file)
        
        # Sort by confidence
        self.result.benchmarks.sort(key=lambda b: b.confidence, reverse=True)
        
        # Limit results
        if len(self.result.benchmarks) > 10:
            print(f"      Found {len(self.result.benchmarks)} potential benchmark(s), showing top 10")
            self.result.benchmarks = self.result.benchmarks[:10]
        else:
            print(f"      Found {len(self.result.benchmarks)} potential benchmark(s)")
    
    def _analyze_bench_file(self, file_path: Path) -> Optional[BenchmarkInfo]:
        """Analyze a file to determine if it's a benchmark (content-based)."""
        try:
            content = file_path.read_text()
        except Exception:
            return None
        
        confidence = 0.0
        bench_type = "script"
        
        # Content-based scoring
        for pattern, score in self.BENCH_CONTENT_KEYWORDS:
            if re.search(pattern, content, re.IGNORECASE):
                confidence += score
        
        # Filename bonus (lower priority than content)
        if "bench" in file_path.name.lower() or "perf" in file_path.name.lower():
            confidence += 0.1
        
        # Kernel name match bonus
        for kernel in self.result.kernels:
            if kernel.kernel_name.lower() in file_path.name.lower():
                confidence += 0.15
                break
        
        # Must have minimum confidence to be considered a benchmark
        if confidence < 0.3:
            return None
        
        # Determine benchmark type
        if "pytest" in content and "benchmark" in content:
            bench_type = "pytest"
        elif "triton.testing.do_bench" in content:
            bench_type = "triton"
        
        command = f"python {file_path}"
        
        return BenchmarkInfo(
            file_path=file_path,
            bench_type=bench_type,
            command=command,
            confidence=min(confidence, 1.0)
        )
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if a file should be skipped during discovery."""
        skip_dirs = {
            "__pycache__", ".git", ".venv", "venv", "node_modules",
            "build", "dist", ".eggs", "*.egg-info"
        }
        
        for part in file_path.parts:
            if part in skip_dirs or part.endswith(".egg-info"):
                return True
        
        return False
    
    def _display_findings(self):
        """Display discovery findings to the user."""
        print("\n[4/4] Discovery complete!")
        print("\n" + "-" * 60)
        
        # Display kernels
        print("\n  KERNELS FOUND:")
        if self.result.kernels:
            for k in self.result.kernels:
                print(f"    - {k.kernel_name} ({k.kernel_type})")
                print(f"      File: {k.file_path}")
                if k.function_names:
                    print(f"      Functions: {', '.join(k.function_names)}")
        else:
            print("    (none found)")
        
        # Display tests
        print("\n  TESTS FOUND:")
        if self.result.tests:
            for t in self.result.tests:
                conf_pct = int(t.confidence * 100)
                print(f"    - {t.file_path.name} ({t.test_type}, {conf_pct}% confidence)")
                print(f"      Command: {t.command}")
        else:
            print("    (none found)")
        
        # Display benchmarks
        print("\n  BENCHMARKS FOUND:")
        if self.result.benchmarks:
            for b in self.result.benchmarks:
                conf_pct = int(b.confidence * 100)
                print(f"    - {b.file_path.name} ({b.bench_type}, {conf_pct}% confidence)")
                print(f"      Command: {b.command}")
        else:
            print("    (none found)")
        
        print("\n" + "-" * 60)
    
    def _prompt_user_confirmation(self):
        """Prompt user to confirm or modify discoveries."""
        if not self.result.tests and not self.result.benchmarks:
            print("\n  No tests or benchmarks found.")
            print("  Options:")
            print("    [c] Create tests (I'll help)")
            print("    [p] Provide test command manually")
            print("    [s] Search in different directory")
            print("    [q] Quit")
        else:
            print("\n  Is this correct?")
            print("    [y] Yes, proceed")
            print("    [e] Edit these paths")
            print("    [s] Search for more")
            print("    [m] Modify these tests")
            print("    [c] Create additional tests")
        
        # For now, just print the prompt - actual input handling
        # will be done by the agent or CLI
        print("\n  (Awaiting user input...)")
    
    def get_test_command(self) -> Optional[str]:
        """Get the best test command from discovery."""
        if self.result.user_provided_test:
            return self.result.user_provided_test
        if self.result.tests:
            return self.result.tests[0].command
        return None
    
    def get_bench_command(self) -> Optional[str]:
        """Get the best benchmark command from discovery."""
        if self.result.user_provided_bench:
            return self.result.user_provided_bench
        if self.result.benchmarks:
            return self.result.benchmarks[0].command
        return None
    
    def get_kernel_path(self) -> Optional[Path]:
        """Get the primary kernel path."""
        if self.result.kernels:
            return self.result.kernels[0].file_path
        return None
    
    def to_context(self) -> dict:
        """Convert discovery result to context for agent prompt."""
        return {
            "workspace": str(self.result.workspace_path),
            "kernels": [
                {
                    "name": k.kernel_name,
                    "file": str(k.file_path),
                    "type": k.kernel_type,
                    "functions": k.function_names
                }
                for k in self.result.kernels
            ],
            "test_command": self.get_test_command(),
            "bench_command": self.get_bench_command(),
            "has_tests": len(self.result.tests) > 0,
            "has_benchmarks": len(self.result.benchmarks) > 0,
        }


# Convenience function
def discover(
    workspace: Path = None,
    kernel_path: Path = None,
    test_command: str = None,
    bench_command: str = None,
    interactive: bool = True,
    use_llm: bool = False
) -> DiscoveryResult:
    """
    Run discovery pipeline.
    
    Args:
        workspace: Workspace directory to search
        kernel_path: Explicit kernel file/directory
        test_command: User-provided test command
        bench_command: User-provided benchmark command
        interactive: Whether to prompt for confirmation
        use_llm: Whether to use LLM for smart analysis
    
    Returns:
        DiscoveryResult with all discovered information
    """
    # If workspace is a file, treat it as kernel_path
    if workspace and Path(workspace).is_file():
        kernel_path = Path(workspace)
        workspace = kernel_path.parent
    
    pipeline = DiscoveryPipeline(workspace, use_llm=use_llm)
    return pipeline.run(
        kernel_path=kernel_path,
        test_command=test_command,
        bench_command=bench_command,
        interactive=interactive
    )


if __name__ == "__main__":
    # Test the discovery pipeline
    import sys
    
    if len(sys.argv) > 1:
        workspace = Path(sys.argv[1])
    else:
        workspace = Path.cwd()
    
    result = discover(workspace, interactive=True)
    
    print("\n\nDiscovery Result:")
    print(f"  Kernels: {len(result.kernels)}")
    print(f"  Tests: {len(result.tests)}")
    print(f"  Benchmarks: {len(result.benchmarks)}")
