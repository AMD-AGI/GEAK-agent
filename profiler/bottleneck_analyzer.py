#!/usr/bin/env python3
"""
Bottleneck Analyzer - Identify kernel performance bottlenecks.

Analyzes hardware counter data to determine if a kernel is:
- Compute-bound (ALU limited)
- Memory-bound (bandwidth limited)
- Latency-bound (kernel launch overhead)
- LDS-bound (shared memory limited)
- Cache-bound (cache thrashing)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class BottleneckType(Enum):
    COMPUTE = "compute"
    MEMORY = "memory"
    LATENCY = "latency"
    LDS = "lds"
    CACHE = "cache"
    BALANCED = "balanced"
    UNKNOWN = "unknown"


@dataclass
class BottleneckReport:
    """Detailed bottleneck analysis report."""
    primary_bottleneck: BottleneckType
    confidence: float  # 0.0 to 1.0
    
    # Utilization metrics (0.0 to 1.0)
    compute_utilization: float = 0.0
    memory_bandwidth_utilization: float = 0.0
    lds_utilization: float = 0.0
    cache_hit_rate: float = 0.0
    occupancy: float = 0.0
    
    # Breakdown of contributing factors
    factors: Dict[str, float] = None
    
    # Detailed analysis
    analysis_text: str = ""
    
    # Optimization priority list
    optimization_priorities: List[Tuple[str, float]] = None


class BottleneckAnalyzer:
    """
    Analyze kernel performance bottlenecks from hardware counters.
    
    Uses the roofline model and hardware counter analysis to identify
    the primary performance limiter.
    """
    
    # AMD MI300X/MI355X specifications
    GPU_SPECS = {
        "gfx942": {  # MI300X
            "peak_tflops_fp16": 1307.0,
            "peak_tflops_fp32": 653.0,
            "peak_bandwidth_gbps": 5300.0,
            "num_cus": 304,
            "simd_width": 64,
            "waves_per_cu": 32,
            "max_waves": 304 * 32,
            "lds_per_cu_kb": 64,
            "l1_cache_per_cu_kb": 32,
            "l2_cache_mb": 256,
            "clock_mhz": 2100,
        },
        "gfx950": {  # MI355X
            "peak_tflops_fp16": 1600.0,
            "peak_tflops_fp32": 800.0,
            "peak_bandwidth_gbps": 8000.0,
            "num_cus": 304,
            "simd_width": 64,
            "waves_per_cu": 32,
            "max_waves": 304 * 32,
            "lds_per_cu_kb": 64,
            "l1_cache_per_cu_kb": 32,
            "l2_cache_mb": 256,
            "clock_mhz": 2500,
        },
    }
    
    def __init__(self, gpu_arch: str = "gfx950"):
        """Initialize with GPU architecture."""
        self.gpu_arch = gpu_arch
        self.specs = self.GPU_SPECS.get(gpu_arch, self.GPU_SPECS["gfx950"])
        
    def analyze(
        self,
        duration_us: float,
        counters: Dict[str, int],
        memory_bytes: int = 0,
        flops: int = 0,
    ) -> BottleneckReport:
        """
        Analyze counters to identify bottleneck.
        
        Args:
            duration_us: Kernel duration in microseconds
            counters: Hardware counter values
            memory_bytes: Total memory transferred (bytes)
            flops: Total floating point operations
            
        Returns:
            BottleneckReport with analysis
        """
        report = BottleneckReport(
            primary_bottleneck=BottleneckType.UNKNOWN,
            confidence=0.0,
            factors={},
            optimization_priorities=[],
        )
        
        if duration_us <= 0:
            report.analysis_text = "Invalid duration"
            return report
            
        # Extract counters
        waves = counters.get("SQ_WAVES", 0)
        valu_insts = counters.get("SQ_INSTS_VALU", 0)
        salu_insts = counters.get("SQ_INSTS_SALU", 0)
        lds_conflicts = counters.get("SQ_LDS_BANK_CONFLICT", 0)
        active_valu = counters.get("SQ_ACTIVE_INST_VALU", 0)
        l2_hits = counters.get("TCC_HIT", 0)
        l2_misses = counters.get("TCC_MISS", 0)
        tcp_reads = counters.get("TCP_TCC_READ_REQ", 0)
        tcp_writes = counters.get("TCP_TCC_WRITE_REQ", 0)
        grbm_active = counters.get("GRBM_GUI_ACTIVE", 0)
        grbm_count = counters.get("GRBM_COUNT", 0)
        
        # Calculate utilization metrics
        duration_cycles = duration_us * self.specs["clock_mhz"]
        
        # Compute utilization
        if grbm_count > 0 and valu_insts > 0:
            # Active cycles / total cycles
            report.compute_utilization = min(1.0, grbm_active / grbm_count)
        elif valu_insts > 0 and duration_cycles > 0:
            # Estimate from VALU instructions
            # Each CU can issue 1 VALU per cycle, 64 threads per wave
            max_valu = self.specs["num_cus"] * duration_cycles
            report.compute_utilization = min(1.0, valu_insts / max_valu)
            
        # Memory bandwidth utilization
        if memory_bytes > 0:
            achieved_bw_gbps = memory_bytes / (duration_us * 1e3)  # bytes/us -> GB/s
            report.memory_bandwidth_utilization = min(
                1.0, achieved_bw_gbps / self.specs["peak_bandwidth_gbps"]
            )
        elif tcp_reads + tcp_writes > 0:
            # Estimate from cache requests (assume 64B per request)
            estimated_bytes = (tcp_reads + tcp_writes) * 64
            achieved_bw_gbps = estimated_bytes / (duration_us * 1e3)
            report.memory_bandwidth_utilization = min(
                1.0, achieved_bw_gbps / self.specs["peak_bandwidth_gbps"]
            )
            
        # Cache hit rate
        total_cache_accesses = l2_hits + l2_misses
        if total_cache_accesses > 0:
            report.cache_hit_rate = l2_hits / total_cache_accesses
            
        # Occupancy
        if waves > 0:
            report.occupancy = min(1.0, waves / self.specs["max_waves"])
            
        # LDS utilization (based on bank conflicts)
        if lds_conflicts > 0 and valu_insts > 0:
            # High conflicts relative to instructions = LDS bottleneck
            conflict_ratio = lds_conflicts / valu_insts
            report.lds_utilization = min(1.0, conflict_ratio * 10)  # Scale factor
            
        # Identify primary bottleneck
        report.primary_bottleneck, report.confidence = self._identify_bottleneck(
            report, duration_us, flops, memory_bytes
        )
        
        # Generate factors breakdown
        report.factors = {
            "compute_utilization": report.compute_utilization,
            "memory_bandwidth_utilization": report.memory_bandwidth_utilization,
            "cache_hit_rate": report.cache_hit_rate,
            "occupancy": report.occupancy,
            "lds_conflict_rate": report.lds_utilization,
        }
        
        # Generate optimization priorities
        report.optimization_priorities = self._prioritize_optimizations(report)
        
        # Generate analysis text
        report.analysis_text = self._generate_analysis_text(report)
        
        return report
    
    def _identify_bottleneck(
        self,
        report: BottleneckReport,
        duration_us: float,
        flops: int,
        memory_bytes: int,
    ) -> Tuple[BottleneckType, float]:
        """Identify the primary bottleneck."""
        
        # Very short kernels are latency-bound
        if duration_us < 5.0:
            return BottleneckType.LATENCY, 0.9
            
        # Check if LDS-bound (high conflict rate)
        if report.lds_utilization > 0.5:
            return BottleneckType.LDS, min(0.9, report.lds_utilization)
            
        # Check cache-bound (low hit rate with high memory access)
        if report.cache_hit_rate < 0.5 and report.memory_bandwidth_utilization > 0.3:
            return BottleneckType.CACHE, 0.7
            
        # Compare compute vs memory utilization
        compute_score = report.compute_utilization
        memory_score = report.memory_bandwidth_utilization
        
        if compute_score > 0.7 and memory_score < 0.5:
            return BottleneckType.COMPUTE, compute_score
        elif memory_score > 0.7 and compute_score < 0.5:
            return BottleneckType.MEMORY, memory_score
        elif compute_score > 0.5 and memory_score > 0.5:
            return BottleneckType.BALANCED, 0.6
        else:
            # Low utilization on both - likely occupancy or other issue
            if report.occupancy < 0.3:
                return BottleneckType.LATENCY, 0.5
            return BottleneckType.UNKNOWN, 0.3
            
    def _prioritize_optimizations(
        self,
        report: BottleneckReport,
    ) -> List[Tuple[str, float]]:
        """Generate prioritized list of optimizations."""
        priorities = []
        
        if report.primary_bottleneck == BottleneckType.MEMORY:
            priorities.append(("Vectorize memory accesses", 0.9))
            priorities.append(("Improve memory coalescing", 0.8))
            priorities.append(("Use shared memory for reused data", 0.7))
            if report.cache_hit_rate < 0.8:
                priorities.append(("Optimize cache access patterns", 0.6))
                
        elif report.primary_bottleneck == BottleneckType.COMPUTE:
            priorities.append(("Use tensor cores (MFMA)", 0.9))
            priorities.append(("Reduce instruction count", 0.7))
            priorities.append(("Use faster math (fast_math)", 0.6))
            
        elif report.primary_bottleneck == BottleneckType.LATENCY:
            priorities.append(("Fuse with adjacent kernels", 0.95))
            priorities.append(("Use persistent kernels", 0.8))
            priorities.append(("Increase work per kernel", 0.7))
            
        elif report.primary_bottleneck == BottleneckType.LDS:
            priorities.append(("Pad shared memory to avoid bank conflicts", 0.9))
            priorities.append(("Reduce shared memory usage", 0.7))
            priorities.append(("Reorganize data layout in LDS", 0.6))
            
        elif report.primary_bottleneck == BottleneckType.CACHE:
            priorities.append(("Improve data locality", 0.9))
            priorities.append(("Use blocking/tiling", 0.8))
            priorities.append(("Prefetch data", 0.6))
            
        # General optimizations for low occupancy
        if report.occupancy < 0.5:
            priorities.append(("Increase occupancy (reduce register usage)", 0.7))
            
        return sorted(priorities, key=lambda x: x[1], reverse=True)
    
    def _generate_analysis_text(self, report: BottleneckReport) -> str:
        """Generate human-readable analysis."""
        lines = []
        
        lines.append(f"Primary Bottleneck: {report.primary_bottleneck.value.upper()}")
        lines.append(f"Confidence: {report.confidence:.0%}")
        lines.append("")
        lines.append("Utilization Metrics:")
        lines.append(f"  Compute: {report.compute_utilization:.1%}")
        lines.append(f"  Memory BW: {report.memory_bandwidth_utilization:.1%}")
        lines.append(f"  Cache Hit: {report.cache_hit_rate:.1%}")
        lines.append(f"  Occupancy: {report.occupancy:.1%}")
        
        if report.lds_utilization > 0:
            lines.append(f"  LDS Conflicts: {report.lds_utilization:.1%}")
            
        lines.append("")
        lines.append("Top Optimization Priorities:")
        for opt, priority in report.optimization_priorities[:3]:
            lines.append(f"  [{priority:.0%}] {opt}")
            
        return "\n".join(lines)
    
    def get_roofline_position(
        self,
        flops: int,
        memory_bytes: int,
        duration_us: float,
    ) -> Tuple[float, float]:
        """
        Calculate position on roofline model.
        
        Returns:
            (arithmetic_intensity, achieved_performance) in (FLOP/byte, TFLOP/s)
        """
        if memory_bytes <= 0 or duration_us <= 0:
            return 0.0, 0.0
            
        # Arithmetic intensity: FLOP per byte
        ai = flops / memory_bytes
        
        # Achieved performance: TFLOP/s
        perf = (flops / duration_us) / 1e6
        
        return ai, perf
    
    def is_memory_bound(
        self,
        flops: int,
        memory_bytes: int,
    ) -> bool:
        """
        Check if kernel is memory-bound based on arithmetic intensity.
        
        Uses ridge point: peak_compute / peak_bandwidth
        """
        if memory_bytes <= 0:
            return False
            
        ai = flops / memory_bytes
        
        # Ridge point for this GPU
        ridge = (self.specs["peak_tflops_fp16"] * 1e12) / (self.specs["peak_bandwidth_gbps"] * 1e9)
        
        return ai < ridge

