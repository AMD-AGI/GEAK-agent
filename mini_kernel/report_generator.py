#!/usr/bin/env python3
"""
Automatic Report Generator for Mini-Kernel Agent

Triggered automatically when all optimization strategies complete.
Generates comprehensive report including:
- Summary statistics
- What was tried
- What worked / what didn't
- Correctness test results
- Benchmark test results
- Recommendations
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class StrategyResult:
    """Result of a single strategy."""
    name: str
    latency_us: float
    speedup: float
    correct: bool
    error: Optional[str] = None
    correctness_details: Optional[Dict] = None
    benchmark_details: Optional[Dict] = None


@dataclass
class OptimizationReport:
    """Complete optimization report."""
    module_name: str
    timestamp: str
    duration_seconds: float
    
    # Baseline
    baseline_latency_us: float
    
    # Best result
    best_latency_us: float
    best_speedup: float
    best_strategy: str
    
    # All results
    strategies_tried: int
    strategies_passed: int
    strategies_failed: int
    results: List[StrategyResult]
    
    # Configuration
    gpu_device: str
    docker_image: str
    warmup_iters: int
    benchmark_iters: int


class ReportGenerator:
    """
    Generates comprehensive optimization reports.
    
    Automatically triggered after optimization loop completes.
    """
    
    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        self.reports_dir = self.work_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
    
    def generate(self, 
                 module_name: str,
                 baseline_latency_us: float,
                 results: List[Dict[str, Any]],
                 config: Dict[str, Any],
                 duration_seconds: float) -> OptimizationReport:
        """Generate comprehensive report."""
        
        # Process results
        strategy_results = []
        best_latency = baseline_latency_us
        best_strategy = "baseline"
        best_speedup = 1.0
        
        for r in results:
            sr = StrategyResult(
                name=r.get("strategy", "unknown"),
                latency_us=r.get("latency_us", 0),
                speedup=r.get("speedup", 0),
                correct=r.get("correct", False),
                error=r.get("error"),
                correctness_details=r.get("correctness_details"),
                benchmark_details=r.get("benchmark_details"),
            )
            strategy_results.append(sr)
            
            if sr.correct and sr.latency_us > 0 and sr.latency_us < best_latency:
                best_latency = sr.latency_us
                best_strategy = sr.name
                best_speedup = baseline_latency_us / sr.latency_us
        
        passed = sum(1 for r in strategy_results if r.correct)
        failed = len(strategy_results) - passed
        
        report = OptimizationReport(
            module_name=module_name,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration_seconds,
            baseline_latency_us=baseline_latency_us,
            best_latency_us=best_latency,
            best_speedup=best_speedup,
            best_strategy=best_strategy,
            strategies_tried=len(strategy_results),
            strategies_passed=passed,
            strategies_failed=failed,
            results=strategy_results,
            gpu_device=config.get("gpu_device", "unknown"),
            docker_image=config.get("docker_image", "unknown"),
            warmup_iters=config.get("warmup_iters", 0),
            benchmark_iters=config.get("benchmark_iters", 0),
        )
        
        # Generate all report formats
        self._save_json_report(report)
        self._save_markdown_report(report)
        self._save_text_report(report)
        
        return report
    
    def _save_json_report(self, report: OptimizationReport):
        """Save JSON report."""
        data = {
            "module": report.module_name,
            "timestamp": report.timestamp,
            "duration_seconds": report.duration_seconds,
            "summary": {
                "baseline_latency_us": report.baseline_latency_us,
                "best_latency_us": report.best_latency_us,
                "best_speedup": report.best_speedup,
                "best_strategy": report.best_strategy,
            },
            "statistics": {
                "strategies_tried": report.strategies_tried,
                "strategies_passed": report.strategies_passed,
                "strategies_failed": report.strategies_failed,
                "success_rate": report.strategies_passed / max(report.strategies_tried, 1),
            },
            "configuration": {
                "gpu_device": report.gpu_device,
                "docker_image": report.docker_image,
                "warmup_iters": report.warmup_iters,
                "benchmark_iters": report.benchmark_iters,
            },
            "results": [
                {
                    "strategy": r.name,
                    "latency_us": r.latency_us,
                    "speedup": r.speedup,
                    "correct": r.correct,
                    "error": r.error,
                }
                for r in report.results
            ],
        }
        
        path = self.reports_dir / "optimization_report.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _save_markdown_report(self, report: OptimizationReport):
        """Save Markdown report."""
        md = []
        
        # Header
        md.append(f"# Optimization Report: {report.module_name}")
        md.append("")
        md.append(f"**Generated:** {report.timestamp}")
        md.append(f"**Duration:** {report.duration_seconds:.1f} seconds")
        md.append("")
        
        # Summary
        md.append("## Summary")
        md.append("")
        md.append("| Metric | Value |")
        md.append("|--------|-------|")
        md.append(f"| Baseline Latency | {report.baseline_latency_us:.2f} μs |")
        md.append(f"| Best Latency | {report.best_latency_us:.2f} μs |")
        md.append(f"| **Best Speedup** | **{report.best_speedup:.2f}x** |")
        md.append(f"| Best Strategy | {report.best_strategy} |")
        md.append("")
        
        # Statistics
        md.append("## Statistics")
        md.append("")
        md.append(f"- Strategies Tried: {report.strategies_tried}")
        md.append(f"- Strategies Passed: {report.strategies_passed} ✓")
        md.append(f"- Strategies Failed: {report.strategies_failed} ✗")
        success_rate = report.strategies_passed / max(report.strategies_tried, 1) * 100
        md.append(f"- Success Rate: {success_rate:.1f}%")
        md.append("")
        
        # What Worked
        md.append("## What Worked ✓")
        md.append("")
        worked = [r for r in report.results if r.correct]
        if worked:
            md.append("| Strategy | Latency | Speedup |")
            md.append("|----------|---------|---------|")
            for r in sorted(worked, key=lambda x: x.latency_us):
                marker = " ⭐" if r.name == report.best_strategy else ""
                md.append(f"| {r.name}{marker} | {r.latency_us:.2f} μs | {r.speedup:.2f}x |")
        else:
            md.append("*No strategies passed correctness check*")
        md.append("")
        
        # What Didn't Work
        md.append("## What Didn't Work ✗")
        md.append("")
        failed = [r for r in report.results if not r.correct]
        if failed:
            md.append("| Strategy | Error |")
            md.append("|----------|-------|")
            for r in failed:
                error = r.error or "Correctness check failed"
                md.append(f"| {r.name} | {error[:50]} |")
        else:
            md.append("*All strategies passed!*")
        md.append("")
        
        # Correctness Tests
        md.append("## Correctness Tests")
        md.append("")
        md.append("| Strategy | Status | Details |")
        md.append("|----------|--------|---------|")
        for r in report.results:
            status = "✓ PASSED" if r.correct else "✗ FAILED"
            details = r.error[:30] if r.error else "All outputs match"
            md.append(f"| {r.name} | {status} | {details} |")
        md.append("")
        
        # Benchmark Tests
        md.append("## Benchmark Tests")
        md.append("")
        md.append(f"- Warmup Iterations: {report.warmup_iters}")
        md.append(f"- Benchmark Iterations: {report.benchmark_iters}")
        md.append("")
        md.append("| Strategy | Latency (μs) | vs Baseline |")
        md.append("|----------|--------------|-------------|")
        for r in sorted(report.results, key=lambda x: x.latency_us if x.correct else float('inf')):
            if r.correct and r.latency_us > 0:
                vs_baseline = "faster" if r.speedup > 1 else "slower"
                diff = abs(r.speedup - 1) * 100
                md.append(f"| {r.name} | {r.latency_us:.2f} | {diff:.1f}% {vs_baseline} |")
            else:
                md.append(f"| {r.name} | N/A | failed |")
        md.append("")
        
        # Configuration
        md.append("## Configuration")
        md.append("")
        md.append(f"- GPU Device: {report.gpu_device}")
        md.append(f"- Docker Image: `{report.docker_image}`")
        md.append("")
        
        # Recommendations
        md.append("## Recommendations")
        md.append("")
        if report.best_speedup > 1.5:
            md.append(f"✅ **Strong improvement found!** Use `{report.best_strategy}` for {report.best_speedup:.2f}x speedup.")
        elif report.best_speedup > 1.1:
            md.append(f"⚠️ Moderate improvement. `{report.best_strategy}` gives {report.best_speedup:.2f}x speedup.")
        else:
            md.append("❌ No significant improvement found. Consider:")
            md.append("  - Different optimization strategies")
            md.append("  - Manual kernel analysis")
            md.append("  - Algorithm-level changes")
        md.append("")
        
        path = self.reports_dir / "optimization_report.md"
        with open(path, "w") as f:
            f.write("\n".join(md))
    
    def _save_text_report(self, report: OptimizationReport):
        """Save plain text report for terminal display."""
        lines = []
        
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  OPTIMIZATION REPORT: {report.module_name}")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"  Generated: {report.timestamp}")
        lines.append(f"  Duration:  {report.duration_seconds:.1f} seconds")
        lines.append("")
        
        # Summary Box
        lines.append("  ┌─────────────────────────────────────────────────────────────────┐")
        lines.append("  │  SUMMARY                                                        │")
        lines.append("  ├─────────────────────────────────────────────────────────────────┤")
        lines.append(f"  │  Baseline:     {report.baseline_latency_us:>8.2f} μs                                │")
        lines.append(f"  │  Best:         {report.best_latency_us:>8.2f} μs                                │")
        lines.append(f"  │  Speedup:      {report.best_speedup:>8.2f}x                                 │")
        lines.append(f"  │  Strategy:     {report.best_strategy:<35}        │")
        lines.append("  └─────────────────────────────────────────────────────────────────┘")
        lines.append("")
        
        # Statistics
        lines.append("  STATISTICS")
        lines.append("  " + "-" * 50)
        lines.append(f"    Strategies Tried:  {report.strategies_tried}")
        lines.append(f"    Strategies Passed: {report.strategies_passed} ✓")
        lines.append(f"    Strategies Failed: {report.strategies_failed} ✗")
        success_rate = report.strategies_passed / max(report.strategies_tried, 1) * 100
        lines.append(f"    Success Rate:      {success_rate:.1f}%")
        lines.append("")
        
        # What Worked
        lines.append("  WHAT WORKED ✓")
        lines.append("  " + "-" * 50)
        worked = [r for r in report.results if r.correct]
        if worked:
            lines.append(f"    {'Strategy':<28} {'Latency':>10} {'Speedup':>10}")
            lines.append("    " + "-" * 48)
            for r in sorted(worked, key=lambda x: x.latency_us):
                marker = " ⭐" if r.name == report.best_strategy else ""
                lines.append(f"    {r.name + marker:<28} {r.latency_us:>8.2f} μs {r.speedup:>9.2f}x")
        else:
            lines.append("    (none)")
        lines.append("")
        
        # What Didn't Work
        lines.append("  WHAT DIDN'T WORK ✗")
        lines.append("  " + "-" * 50)
        failed = [r for r in report.results if not r.correct]
        if failed:
            for r in failed:
                error = r.error or "Correctness failed"
                lines.append(f"    {r.name}: {error[:40]}")
        else:
            lines.append("    (all strategies passed!)")
        lines.append("")
        
        # Correctness Tests
        lines.append("  CORRECTNESS TESTS")
        lines.append("  " + "-" * 50)
        for r in report.results:
            status = "✓ PASS" if r.correct else "✗ FAIL"
            lines.append(f"    {r.name:<28} {status}")
        lines.append("")
        
        # Benchmark Tests
        lines.append("  BENCHMARK TESTS")
        lines.append("  " + "-" * 50)
        lines.append(f"    Warmup:     {report.warmup_iters} iterations")
        lines.append(f"    Benchmark:  {report.benchmark_iters} iterations")
        lines.append("")
        lines.append(f"    {'Strategy':<28} {'Latency':>12} {'vs Baseline':>12}")
        lines.append("    " + "-" * 52)
        for r in sorted(report.results, key=lambda x: x.latency_us if x.correct else float('inf')):
            if r.correct and r.latency_us > 0:
                vs = "faster" if r.speedup > 1 else "slower"
                diff = abs(r.speedup - 1) * 100
                lines.append(f"    {r.name:<28} {r.latency_us:>10.2f} μs {diff:>6.1f}% {vs}")
            else:
                lines.append(f"    {r.name:<28} {'N/A':>12} {'failed':>12}")
        lines.append("")
        
        lines.append("=" * 70)
        lines.append("")
        
        text = "\n".join(lines)
        
        path = self.reports_dir / "optimization_report.txt"
        with open(path, "w") as f:
            f.write(text)
        
        # Also print to terminal
        print(text)


def generate_report(work_dir: Path,
                   module_name: str,
                   baseline_latency_us: float,
                   results: List[Dict[str, Any]],
                   config: Dict[str, Any],
                   duration_seconds: float) -> OptimizationReport:
    """
    Convenience function to generate report.
    
    Called automatically at end of optimization loop.
    """
    generator = ReportGenerator(work_dir)
    return generator.generate(
        module_name=module_name,
        baseline_latency_us=baseline_latency_us,
        results=results,
        config=config,
        duration_seconds=duration_seconds,
    )


