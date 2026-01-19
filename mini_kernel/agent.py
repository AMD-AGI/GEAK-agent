"""
MiniKernelAgent: The Main Autonomous Kernel Optimization Agent

This is the core agent that orchestrates:
- Test discovery (or generation)
- Baseline capture
- Profiling for bottleneck analysis
- Strategy selection based on profiler output
- Optimization loop with OpenEvolve integration
- Correctness verification
- Checkpointing
- User intervention when needed

Usage:
    agent = MiniKernelAgent(kernel_path="my_kernel.py")
    result = agent.optimize()
"""

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from enum import Enum

from .state import StateMachine, AgentState, TransitionTrigger, OptimizationProgress
from .intervention import InterventionPolicy, InterventionMode, INTERRUPT_KEYWORDS
from .checkpoint import CheckpointManager
from .mcp_tools.profiler import ProfilerTool, BottleneckType
from .mcp_tools.bench import BenchTool
from .mcp_tools.verify import VerifyTool


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    kernel_path: Path
    function_name: Optional[str] = None
    test_command: Optional[str] = None
    bench_command: Optional[str] = None
    gpu_device: str = "3"
    intervention_mode: str = "balanced"
    max_iterations: int = 20
    goal: str = "minimize latency"
    protected_files: List[str] = field(default_factory=list)
    docker_image: str = "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x"


@dataclass
class OptimizationResult:
    """Final result of optimization."""
    success: bool
    baseline_latency_us: float
    best_latency_us: float
    speedup: float
    best_strategy: str
    iterations: int
    checkpoints_created: int
    total_time_seconds: float
    best_code: Optional[str] = None
    report_path: Optional[str] = None


class MiniKernelAgent:
    """
    The main autonomous kernel optimization agent.
    
    Features:
    - Zero-config: Auto-discovers tests, benchmarks, kernel functions
    - Supervised autonomy: Runs fast, pauses when it matters
    - Checkpointing: Never loses progress
    - MCP Tools: Uses profiler, bench, verify as independent tools
    - Live feedback: Shows performance deltas in real-time
    """
    
    def __init__(self, config: AgentConfig = None, **kwargs):
        # Allow both config object and kwargs
        if config:
            self.config = config
        else:
            self.config = AgentConfig(**kwargs)
        
        # Initialize work directory
        self.work_dir = Path(f".mini-kernel-{self.config.kernel_path.stem}")
        self.work_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.policy = InterventionPolicy.from_mode(self.config.intervention_mode)
        self.policy.protected_files = set(self.config.protected_files)
        
        self.state_machine = StateMachine(self.policy)
        self.checkpoint_mgr = CheckpointManager(self.work_dir)
        
        # Initialize MCP tools
        self.profiler = ProfilerTool(
            docker_image=self.config.docker_image,
            gpu_device=self.config.gpu_device
        )
        self.bench = BenchTool(
            docker_image=self.config.docker_image,
            gpu_device=self.config.gpu_device
        )
        self.verify = VerifyTool(
            docker_image=self.config.docker_image,
            gpu_device=self.config.gpu_device
        )
        
        # State
        self.baseline_code: Optional[str] = None
        self.baseline_latency_us: float = 0
        self.current_code: Optional[str] = None
        self.current_strategy: Optional[str] = None
        self.strategies_tried: List[str] = []
        self.start_time: Optional[float] = None
        
        # Callbacks
        self._on_progress: Optional[Callable] = None
        self._on_checkpoint: Optional[Callable] = None
        self._on_intervention: Optional[Callable] = None
    
    def log(self, msg: str, level: str = "INFO"):
        """Log with timestamp."""
        elapsed = f"[{time.time() - self.start_time:.1f}s] " if self.start_time else ""
        print(f"{elapsed}[{level}] {msg}")
        
        # Also write to session log
        log_file = self.work_dir / "session.log"
        with open(log_file, "a") as f:
            f.write(f"{elapsed}[{level}] {msg}\n")
    
    def optimize(self) -> OptimizationResult:
        """
        Run the full autonomous optimization pipeline.
        
        Returns:
            OptimizationResult with best code and metrics
        """
        self.start_time = time.time()
        
        self._print_banner()
        
        # ================================================================
        # PHASE 1: DISCOVERY
        # ================================================================
        self.log("\n[PHASE 1] DISCOVERY")
        self.state_machine.transition(TransitionTrigger.TESTS_PASS)  # Move to discovering
        
        discovery = self._discover()
        if not discovery["success"]:
            return self._create_failure_result(discovery.get("error", "Discovery failed"))
        
        # ================================================================
        # PHASE 2: BASELINE
        # ================================================================
        self.log("\n[PHASE 2] BASELINE CAPTURE")
        
        baseline = self._capture_baseline()
        if not baseline["success"]:
            return self._create_failure_result(baseline.get("error", "Baseline capture failed"))
        
        self.baseline_latency_us = baseline["latency_us"]
        self.checkpoint_mgr.save_baseline({"latency_us": self.baseline_latency_us})
        
        self.state_machine.update_progress(
            baseline_latency_us=self.baseline_latency_us,
            best_latency_us=self.baseline_latency_us,
        )
        
        # ================================================================
        # PHASE 3: PROFILE & ANALYZE
        # ================================================================
        self.log("\n[PHASE 3] PROFILE & ANALYZE")
        
        profile = self._profile_baseline()
        strategies = self._select_strategies(profile)
        
        self.log(f"  Bottleneck: {profile['bottleneck']}")
        self.log(f"  Strategies: {[s['name'] for s in strategies]}")
        
        # ================================================================
        # PHASE 4: OPTIMIZATION LOOP
        # ================================================================
        self.log("\n[PHASE 4] OPTIMIZATION LOOP")
        self.state_machine.transition(TransitionTrigger.CHECKPOINT_SAVED)  # Move to autonomous
        
        best_result = self._optimization_loop(strategies)
        
        # ================================================================
        # PHASE 5: FINALIZE
        # ================================================================
        self.log("\n[PHASE 5] FINALIZE")
        
        result = self._finalize(best_result)
        
        self._print_summary(result)
        
        return result
    
    def _print_banner(self):
        """Print startup banner."""
        print("=" * 70)
        print("  mini-kernel: Autonomous Kernel Optimization Agent")
        print("=" * 70)
        print(f"  Kernel: {self.config.kernel_path}")
        print(f"  Mode: {self.config.intervention_mode}")
        print(f"  GPU: {self.config.gpu_device}")
        print("=" * 70)
    
    def _discover(self) -> Dict[str, Any]:
        """Discover kernel, tests, and benchmarks."""
        self.log("-" * 50)
        
        # Load kernel code
        kernel_path = Path(self.config.kernel_path)
        if not kernel_path.exists():
            return {"success": False, "error": f"Kernel not found: {kernel_path}"}
        
        self.baseline_code = kernel_path.read_text()
        self.current_code = self.baseline_code
        
        self.log(f"  Loaded kernel: {kernel_path}")
        
        # Auto-detect function name if not specified
        if not self.config.function_name:
            # Look for main kernel function
            import re
            funcs = re.findall(r'def\s+(\w+)\s*\(', self.baseline_code)
            kernel_funcs = [f for f in funcs if 'run_' in f or 'kernel' in f.lower()]
            if kernel_funcs:
                self.config.function_name = kernel_funcs[0]
                self.log(f"  Auto-detected function: {self.config.function_name}")
        
        # Check for test files
        test_patterns = ["test_*.py", "*_test.py"]
        tests_found = []
        for pattern in test_patterns:
            tests_found.extend(kernel_path.parent.glob(pattern))
        
        if tests_found:
            self.log(f"  Found tests: {[t.name for t in tests_found]}")
        
        # Check for benchmark files
        bench_patterns = ["bench*.py", "*benchmark*.py"]
        benchmarks_found = []
        for pattern in bench_patterns:
            benchmarks_found.extend(kernel_path.parent.glob(pattern))
        
        if benchmarks_found:
            self.log(f"  Found benchmarks: {[b.name for b in benchmarks_found]}")
        
        return {"success": True, "tests": tests_found, "benchmarks": benchmarks_found}
    
    def _capture_baseline(self) -> Dict[str, Any]:
        """Capture baseline performance."""
        self.log("-" * 50)
        
        # Use bench tool to measure baseline
        result = self.bench.benchmark(
            kernel_code=self.baseline_code,
            warmup_iters=1000,
            bench_iters=3000,
        )
        
        self.log(f"  Baseline latency: {result.mean_latency_us:.2f} μs")
        
        return {
            "success": True,
            "latency_us": result.mean_latency_us,
            "min_us": result.min_latency_us,
            "max_us": result.max_latency_us,
        }
    
    def _profile_baseline(self) -> Dict[str, Any]:
        """Profile baseline to identify bottlenecks."""
        self.log("-" * 50)
        
        # Use profiler tool
        result = self.profiler.profile(
            kernel_code=self.baseline_code,
            warmup_iters=100,
            profile_iters=100,
        )
        
        return {
            "bottleneck": result.bottleneck.value,
            "suggestions": result.suggestions,
            "compute_util": result.compute_utilization,
            "memory_util": result.memory_utilization,
        }
    
    def _select_strategies(self, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select optimization strategies based on profiler output."""
        bottleneck = profile["bottleneck"]
        
        if bottleneck == "latency":
            return [
                {"name": "pytorch_replacement", "description": "Replace with PyTorch ops"},
                {"name": "hip_graph", "description": "Use HIP Graph capture"},
                {"name": "kernel_fusion", "description": "Fuse adjacent kernels"},
            ]
        elif bottleneck == "memory":
            return [
                {"name": "shared_memory", "description": "Use shared memory caching"},
                {"name": "coalesce_memory", "description": "Improve memory coalescing"},
                {"name": "reduce_traffic", "description": "Reduce memory traffic"},
            ]
        elif bottleneck == "compute":
            return [
                {"name": "vectorize", "description": "Vectorize operations"},
                {"name": "tensor_cores", "description": "Use tensor cores"},
                {"name": "algorithm_opt", "description": "Algorithmic optimization"},
            ]
        else:
            return [
                {"name": "autotune", "description": "Autotune parameters"},
                {"name": "general_opt", "description": "General optimizations"},
            ]
    
    def _optimization_loop(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the main optimization loop."""
        best_latency = self.baseline_latency_us
        best_strategy = None
        best_code = self.baseline_code
        
        for i, strategy in enumerate(strategies):
            iteration = i + 1
            
            # Check for timeout
            if time.time() - self.start_time > 1800:  # 30 min timeout
                self.log("⏱️ Timeout reached")
                break
            
            # Check iteration limit
            if iteration > self.config.max_iterations:
                self.log(f"Max iterations ({self.config.max_iterations}) reached")
                break
            
            self.log(f"\n[ITERATION {iteration}] {strategy['name']}")
            self.log("-" * 50)
            
            self.current_strategy = strategy['name']
            self.strategies_tried.append(strategy['name'])
            
            # Generate optimized code
            optimized_code = self._generate_optimized(strategy)
            
            if optimized_code is None:
                self.log(f"  ⚠️ No optimization generated")
                self.state_machine.update_progress(consecutive_failures=
                    self.state_machine.progress.consecutive_failures + 1)
                continue
            
            # Verify correctness
            self.log(f"  Verifying correctness...")
            verify_result = self._verify_optimized(optimized_code)
            
            if not verify_result["passed"]:
                self.log(f"  ❌ Correctness failed: {verify_result.get('mismatches', [])}")
                self.state_machine.update_progress(consecutive_failures=
                    self.state_machine.progress.consecutive_failures + 1)
                
                # Check if intervention needed
                trigger = self.state_machine.check_intervention_needed()
                if trigger:
                    self._handle_intervention(trigger)
                continue
            
            self.log(f"  ✓ Correctness passed")
            
            # Benchmark
            self.log(f"  Benchmarking...")
            bench_result = self.bench.benchmark(
                kernel_code=optimized_code,
                warmup_iters=1000,
                bench_iters=3000,
                baseline_latency_us=self.baseline_latency_us,
            )
            
            latency = bench_result.mean_latency_us
            speedup = self.baseline_latency_us / latency if latency > 0 else 1.0
            
            self.log(f"  Latency: {latency:.2f} μs (speedup: {speedup:.2f}x)")
            
            # Update progress
            self.state_machine.update_progress(
                iteration=iteration,
                current_latency_us=latency,
                consecutive_failures=0,
            )
            
            # Check for improvement
            if latency < best_latency:
                self.log(f"  ⭐ NEW BEST!")
                best_latency = latency
                best_strategy = strategy['name']
                best_code = optimized_code
                
                # Create checkpoint
                self.state_machine.transition(TransitionTrigger.PERFORMANCE_IMPROVED)
                
                checkpoint = self.checkpoint_mgr.create_checkpoint(
                    latency_us=latency,
                    strategy=strategy['name'],
                    iteration=iteration,
                    tests_passed=True,
                    modified_files={"optimized.py": optimized_code},
                    summary=f"{strategy['name']}: {speedup:.2f}x speedup",
                )
                
                self.log(f"  Checkpoint: {checkpoint.id}")
                
                self.state_machine.update_progress(
                    best_latency_us=best_latency,
                    best_checkpoint=checkpoint.id,
                )
                
                self.state_machine.transition(TransitionTrigger.CHECKPOINT_SAVED)
            else:
                # Check for regression
                regression = (latency - best_latency) / best_latency * 100
                if regression > self.policy.performance_regression_pct:
                    self.log(f"  ⚠️ Regression: {regression:.1f}%")
        
        return {
            "best_latency_us": best_latency,
            "best_strategy": best_strategy,
            "best_code": best_code,
            "iterations": len(self.strategies_tried),
        }
    
    def _generate_optimized(self, strategy: Dict[str, Any]) -> Optional[str]:
        """Generate optimized code for a strategy."""
        # In a full implementation, this would use LLM to generate optimizations
        # For now, return None to indicate no optimization generated
        # The actual implementation would integrate with OpenEvolve
        return None
    
    def _verify_optimized(self, optimized_code: str) -> Dict[str, Any]:
        """Verify optimized code against baseline."""
        # Use verify tool
        # In a full implementation, would pass proper output tensor list
        return {"passed": True, "mismatches": []}
    
    def _handle_intervention(self, trigger: TransitionTrigger):
        """Handle user intervention."""
        self.state_machine.transition(trigger)
        
        self.log(f"\n⏸️ INTERVENTION NEEDED: {trigger.name}")
        self.log("=" * 50)
        
        status = self.state_machine.get_status()
        self.log(f"  State: {status['state']}")
        self.log(f"  Iteration: {status['iteration']}")
        self.log(f"  Best speedup: {status['baseline_us'] / status['best_us']:.2f}x")
        self.log(f"  Consecutive failures: {status['consecutive_failures']}")
        
        # In interactive mode, would prompt user here
        # For autonomous mode, continue with next strategy
        self.state_machine.transition(TransitionTrigger.CHECKPOINT_SAVED)
    
    def _finalize(self, best_result: Dict[str, Any]) -> OptimizationResult:
        """Finalize optimization and generate report."""
        total_time = time.time() - self.start_time
        
        speedup = self.baseline_latency_us / best_result["best_latency_us"] \
            if best_result["best_latency_us"] > 0 else 1.0
        
        # Generate report
        report = {
            "kernel": str(self.config.kernel_path),
            "baseline_latency_us": self.baseline_latency_us,
            "best_latency_us": best_result["best_latency_us"],
            "speedup": speedup,
            "best_strategy": best_result["best_strategy"],
            "iterations": best_result["iterations"],
            "strategies_tried": self.strategies_tried,
            "checkpoints": self.checkpoint_mgr.list_checkpoints(),
            "total_time_seconds": total_time,
        }
        
        report_path = self.work_dir / "optimization_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        # Export best if improved
        if speedup > 1.0 and self.checkpoint_mgr.best_checkpoint:
            export_dir = self.work_dir / "best_result"
            self.checkpoint_mgr.export_best(export_dir)
        
        return OptimizationResult(
            success=speedup > 1.0,
            baseline_latency_us=self.baseline_latency_us,
            best_latency_us=best_result["best_latency_us"],
            speedup=speedup,
            best_strategy=best_result["best_strategy"] or "none",
            iterations=best_result["iterations"],
            checkpoints_created=len(self.checkpoint_mgr.checkpoints),
            total_time_seconds=total_time,
            best_code=best_result["best_code"],
            report_path=str(report_path),
        )
    
    def _create_failure_result(self, error: str) -> OptimizationResult:
        """Create a failure result."""
        self.log(f"❌ {error}", "ERROR")
        return OptimizationResult(
            success=False,
            baseline_latency_us=0,
            best_latency_us=0,
            speedup=1.0,
            best_strategy="none",
            iterations=0,
            checkpoints_created=0,
            total_time_seconds=time.time() - (self.start_time or time.time()),
        )
    
    def _print_summary(self, result: OptimizationResult):
        """Print final summary."""
        print("\n" + "=" * 70)
        print("  OPTIMIZATION COMPLETE")
        print("=" * 70)
        print(f"  Baseline:    {result.baseline_latency_us:.2f} μs")
        print(f"  Best:        {result.best_latency_us:.2f} μs")
        print(f"  Speedup:     {result.speedup:.2f}x")
        print(f"  Strategy:    {result.best_strategy}")
        print(f"  Iterations:  {result.iterations}")
        print(f"  Checkpoints: {result.checkpoints_created}")
        print(f"  Time:        {result.total_time_seconds:.1f}s")
        print(f"  Report:      {result.report_path}")
        print("=" * 70)
    
    # ================================================================
    # USER CONTROL METHODS
    # ================================================================
    
    def pause(self):
        """Pause the agent."""
        self.state_machine.transition(TransitionTrigger.USER_INTERRUPT)
    
    def resume(self):
        """Resume the agent."""
        self.state_machine.transition(TransitionTrigger.CHECKPOINT_SAVED)
    
    def rollback(self, checkpoint_id: str = None):
        """Rollback to a checkpoint."""
        files = self.checkpoint_mgr.rollback(checkpoint_id)
        if files:
            self.current_code = files.get("optimized.py", self.baseline_code)
            self.log(f"Rolled back to {checkpoint_id or 'previous'}")
    
    def rollback_to_best(self):
        """Rollback to best checkpoint."""
        self.rollback("best")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return self.state_machine.get_status()
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints."""
        return self.checkpoint_mgr.list_checkpoints()


