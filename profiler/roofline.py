#!/usr/bin/env python3
"""
Roofline Model Analysis for GPU Kernels.

The roofline model helps visualize kernel performance relative to
hardware limits. Kernels fall into three regions:
- Memory-bound: Limited by memory bandwidth
- Compute-bound: Limited by compute throughput
- Balanced: Near the ridge point

Reference: https://crd.lbl.gov/assets/pubs_presos/parlab08-roofline-talk.pdf
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math


@dataclass
class RooflinePoint:
    """A point on the roofline model."""
    name: str
    arithmetic_intensity: float  # FLOP/byte
    performance: float  # TFLOP/s
    
    # Optional metadata
    duration_us: float = 0.0
    flops: int = 0
    memory_bytes: int = 0
    
    # Classification
    is_memory_bound: bool = False
    is_compute_bound: bool = False
    distance_to_roof: float = 0.0  # How far from the roof (0 = on roof)
    efficiency: float = 0.0  # Percentage of theoretical peak


@dataclass
class RooflineSpec:
    """GPU roofline specifications."""
    name: str
    peak_compute_tflops: float  # Peak compute throughput
    peak_bandwidth_gbps: float  # Peak memory bandwidth
    
    # Derived
    ridge_point: float = 0.0  # AI where compute roof meets memory roof
    
    def __post_init__(self):
        # Ridge point: peak_compute / peak_bandwidth
        # Convert to common units: TFLOP/s / (GB/s) = FLOP/byte * 1000
        self.ridge_point = (self.peak_compute_tflops * 1e12) / (self.peak_bandwidth_gbps * 1e9)


class RooflineModel:
    """
    Roofline model for analyzing GPU kernel performance.
    
    Usage:
        model = RooflineModel(gpu_arch="gfx950")
        
        # Add kernel measurements
        model.add_kernel("matmul", flops=1e12, memory_bytes=1e9, duration_us=500)
        
        # Analyze
        for point in model.get_points():
            print(f"{point.name}: {point.efficiency:.1%} efficient")
            
        # Generate visualization data
        plot_data = model.get_plot_data()
    """
    
    # GPU Specifications
    GPU_SPECS = {
        "gfx942": RooflineSpec(
            name="AMD MI300X",
            peak_compute_tflops=1307.0,  # FP16
            peak_bandwidth_gbps=5300.0,
        ),
        "gfx950": RooflineSpec(
            name="AMD MI355X",
            peak_compute_tflops=1600.0,  # FP16 estimated
            peak_bandwidth_gbps=8000.0,
        ),
        "gfx90a": RooflineSpec(
            name="AMD MI250X",
            peak_compute_tflops=383.0,  # FP16
            peak_bandwidth_gbps=3200.0,
        ),
    }
    
    def __init__(
        self,
        gpu_arch: str = "gfx950",
        custom_spec: Optional[RooflineSpec] = None,
    ):
        """
        Initialize roofline model.
        
        Args:
            gpu_arch: GPU architecture string
            custom_spec: Custom RooflineSpec (overrides gpu_arch)
        """
        if custom_spec:
            self.spec = custom_spec
        else:
            self.spec = self.GPU_SPECS.get(gpu_arch, self.GPU_SPECS["gfx950"])
            
        self.points: List[RooflinePoint] = []
        
    def add_kernel(
        self,
        name: str,
        flops: int,
        memory_bytes: int,
        duration_us: float,
    ) -> RooflinePoint:
        """
        Add a kernel measurement to the model.
        
        Args:
            name: Kernel name
            flops: Total floating point operations
            memory_bytes: Total memory transferred (bytes)
            duration_us: Kernel duration (microseconds)
            
        Returns:
            RooflinePoint for this kernel
        """
        if memory_bytes <= 0 or duration_us <= 0:
            raise ValueError("memory_bytes and duration_us must be positive")
            
        # Calculate arithmetic intensity (FLOP/byte)
        ai = flops / memory_bytes
        
        # Calculate achieved performance (TFLOP/s)
        perf = (flops / duration_us) / 1e6
        
        # Calculate theoretical max at this AI
        theoretical_max = self.get_roof_at_ai(ai)
        
        # Calculate efficiency
        efficiency = perf / theoretical_max if theoretical_max > 0 else 0.0
        
        # Classify
        is_memory_bound = ai < self.spec.ridge_point
        is_compute_bound = ai >= self.spec.ridge_point
        
        # Distance to roof (how much room for improvement)
        distance_to_roof = theoretical_max - perf
        
        point = RooflinePoint(
            name=name,
            arithmetic_intensity=ai,
            performance=perf,
            duration_us=duration_us,
            flops=flops,
            memory_bytes=memory_bytes,
            is_memory_bound=is_memory_bound,
            is_compute_bound=is_compute_bound,
            distance_to_roof=distance_to_roof,
            efficiency=efficiency,
        )
        
        self.points.append(point)
        return point
    
    def add_point(
        self,
        name: str,
        arithmetic_intensity: float,
        performance: float,
    ) -> RooflinePoint:
        """
        Add a pre-calculated point to the model.
        
        Args:
            name: Kernel name
            arithmetic_intensity: FLOP/byte
            performance: Achieved TFLOP/s
            
        Returns:
            RooflinePoint
        """
        theoretical_max = self.get_roof_at_ai(arithmetic_intensity)
        efficiency = performance / theoretical_max if theoretical_max > 0 else 0.0
        
        point = RooflinePoint(
            name=name,
            arithmetic_intensity=arithmetic_intensity,
            performance=performance,
            is_memory_bound=arithmetic_intensity < self.spec.ridge_point,
            is_compute_bound=arithmetic_intensity >= self.spec.ridge_point,
            distance_to_roof=theoretical_max - performance,
            efficiency=efficiency,
        )
        
        self.points.append(point)
        return point
    
    def get_roof_at_ai(self, ai: float) -> float:
        """
        Get the roofline ceiling at a given arithmetic intensity.
        
        Args:
            ai: Arithmetic intensity (FLOP/byte)
            
        Returns:
            Maximum achievable performance (TFLOP/s)
        """
        # Memory roof: performance = bandwidth * AI
        memory_roof = (self.spec.peak_bandwidth_gbps * ai) / 1000  # GB/s * FLOP/byte -> TFLOP/s
        
        # Compute roof: flat ceiling
        compute_roof = self.spec.peak_compute_tflops
        
        # Actual roof is minimum of both
        return min(memory_roof, compute_roof)
    
    def get_points(self) -> List[RooflinePoint]:
        """Get all kernel points."""
        return self.points
    
    def get_memory_bound_points(self) -> List[RooflinePoint]:
        """Get only memory-bound kernels."""
        return [p for p in self.points if p.is_memory_bound]
    
    def get_compute_bound_points(self) -> List[RooflinePoint]:
        """Get only compute-bound kernels."""
        return [p for p in self.points if p.is_compute_bound]
    
    def get_plot_data(
        self,
        ai_range: Tuple[float, float] = (0.01, 1000),
        num_points: int = 100,
    ) -> Dict:
        """
        Get data for plotting the roofline.
        
        Returns dictionary with:
        - 'ai_values': List of arithmetic intensity values
        - 'roof_values': List of roof performance values
        - 'ridge_point': (ai, perf) of ridge point
        - 'kernel_points': List of kernel point data
        - 'spec': GPU specification info
        """
        import numpy as np
        
        # Generate AI values (log scale)
        ai_values = np.logspace(
            np.log10(ai_range[0]),
            np.log10(ai_range[1]),
            num_points
        ).tolist()
        
        # Calculate roof at each AI
        roof_values = [self.get_roof_at_ai(ai) for ai in ai_values]
        
        # Kernel points
        kernel_data = []
        for p in self.points:
            kernel_data.append({
                'name': p.name,
                'ai': p.arithmetic_intensity,
                'performance': p.performance,
                'efficiency': p.efficiency,
                'is_memory_bound': p.is_memory_bound,
            })
        
        return {
            'ai_values': ai_values,
            'roof_values': roof_values,
            'ridge_point': (self.spec.ridge_point, self.spec.peak_compute_tflops),
            'kernel_points': kernel_data,
            'spec': {
                'name': self.spec.name,
                'peak_compute_tflops': self.spec.peak_compute_tflops,
                'peak_bandwidth_gbps': self.spec.peak_bandwidth_gbps,
                'ridge_point': self.spec.ridge_point,
            }
        }
    
    def generate_ascii_plot(self, width: int = 80, height: int = 20) -> str:
        """
        Generate ASCII roofline plot.
        
        Returns:
            ASCII art string of the roofline plot
        """
        import numpy as np
        
        # Get range of AI values
        if self.points:
            min_ai = min(p.arithmetic_intensity for p in self.points) * 0.5
            max_ai = max(p.arithmetic_intensity for p in self.points) * 2
        else:
            min_ai = 0.01
            max_ai = 100
            
        # Ensure ridge point is visible
        min_ai = min(min_ai, self.spec.ridge_point * 0.1)
        max_ai = max(max_ai, self.spec.ridge_point * 10)
        
        # Create plot grid
        ai_values = np.logspace(np.log10(min_ai), np.log10(max_ai), width)
        roof_values = [self.get_roof_at_ai(ai) for ai in ai_values]
        
        max_perf = max(roof_values) * 1.1
        if self.points:
            max_perf = max(max_perf, max(p.performance for p in self.points) * 1.1)
            
        # Build plot
        plot = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Draw roof
        for i, (ai, perf) in enumerate(zip(ai_values, roof_values)):
            y = int((1 - perf / max_perf) * (height - 1))
            y = max(0, min(height - 1, y))
            plot[y][i] = '-'
            
        # Mark ridge point
        ridge_x = int(np.log10(self.spec.ridge_point / min_ai) / np.log10(max_ai / min_ai) * (width - 1))
        ridge_x = max(0, min(width - 1, ridge_x))
        ridge_y = int((1 - self.spec.peak_compute_tflops / max_perf) * (height - 1))
        ridge_y = max(0, min(height - 1, ridge_y))
        plot[ridge_y][ridge_x] = 'R'
        
        # Plot kernel points
        markers = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        legend = []
        for i, p in enumerate(self.points[:26]):
            x = int(np.log10(p.arithmetic_intensity / min_ai) / np.log10(max_ai / min_ai) * (width - 1))
            y = int((1 - p.performance / max_perf) * (height - 1))
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            marker = markers[i]
            plot[y][x] = marker
            legend.append(f"{marker}: {p.name} ({p.efficiency:.0%})")
            
        # Convert to string
        lines = []
        lines.append(f"Roofline Model - {self.spec.name}")
        lines.append("=" * width)
        lines.append(f"Peak: {self.spec.peak_compute_tflops:.0f} TFLOP/s | BW: {self.spec.peak_bandwidth_gbps:.0f} GB/s")
        lines.append("-" * width)
        
        # Y-axis labels
        for i, row in enumerate(plot):
            perf = max_perf * (1 - i / (height - 1))
            label = f"{perf:6.0f}|" if i % 4 == 0 else "      |"
            lines.append(label + ''.join(row))
            
        # X-axis
        lines.append("      +" + "-" * width)
        lines.append(f"       {min_ai:.2f}" + " " * (width - 20) + f"{max_ai:.1f}")
        lines.append("       " + " " * (width // 2 - 10) + "Arithmetic Intensity (FLOP/byte)")
        
        # Legend
        lines.append("")
        lines.append("Legend: R=Ridge Point")
        for leg in legend:
            lines.append(f"  {leg}")
            
        return "\n".join(lines)
    
    def get_optimization_guidance(self, point: RooflinePoint) -> List[str]:
        """
        Get optimization guidance for a kernel point.
        
        Args:
            point: RooflinePoint to analyze
            
        Returns:
            List of optimization suggestions
        """
        guidance = []
        
        if point.is_memory_bound:
            guidance.append("MEMORY-BOUND OPTIMIZATIONS:")
            guidance.append("  - Increase arithmetic intensity by fusing operations")
            guidance.append("  - Use data compression/quantization")
            guidance.append("  - Improve cache utilization (tiling, blocking)")
            guidance.append("  - Use vectorized memory operations (float4, etc.)")
            guidance.append("  - Consider persistent kernels to reduce memory traffic")
            
            if point.efficiency < 0.5:
                guidance.append("")
                guidance.append("  LOW EFFICIENCY DETECTED:")
                guidance.append("  - Check memory coalescing")
                guidance.append("  - Look for bank conflicts in shared memory")
                guidance.append("  - Verify alignment of memory accesses")
                
        else:  # Compute bound
            guidance.append("COMPUTE-BOUND OPTIMIZATIONS:")
            guidance.append("  - Use tensor cores (MFMA instructions)")
            guidance.append("  - Reduce instruction count")
            guidance.append("  - Use fast math approximations")
            guidance.append("  - Increase ILP (instruction-level parallelism)")
            
            if point.efficiency < 0.5:
                guidance.append("")
                guidance.append("  LOW EFFICIENCY DETECTED:")
                guidance.append("  - Check for control flow divergence")
                guidance.append("  - Increase occupancy")
                guidance.append("  - Balance register usage")
                
        # General guidance
        guidance.append("")
        guidance.append(f"Current efficiency: {point.efficiency:.1%}")
        guidance.append(f"Potential improvement: {point.distance_to_roof:.1f} TFLOP/s")
        
        return guidance
    
    def clear(self):
        """Clear all kernel points."""
        self.points = []


# Utility function for quick roofline analysis
def quick_roofline_analysis(
    kernels: List[Dict],
    gpu_arch: str = "gfx950",
) -> str:
    """
    Quick roofline analysis for a list of kernels.
    
    Args:
        kernels: List of dicts with keys: name, flops, memory_bytes, duration_us
        gpu_arch: GPU architecture
        
    Returns:
        Analysis string
    """
    model = RooflineModel(gpu_arch=gpu_arch)
    
    for k in kernels:
        model.add_kernel(
            name=k['name'],
            flops=k['flops'],
            memory_bytes=k['memory_bytes'],
            duration_us=k['duration_us'],
        )
        
    # Generate report
    lines = []
    lines.append(model.generate_ascii_plot())
    lines.append("")
    lines.append("=" * 80)
    lines.append("DETAILED ANALYSIS")
    lines.append("=" * 80)
    
    for point in model.get_points():
        lines.append("")
        lines.append(f"Kernel: {point.name}")
        lines.append("-" * 40)
        lines.append(f"  AI: {point.arithmetic_intensity:.2f} FLOP/byte")
        lines.append(f"  Performance: {point.performance:.2f} TFLOP/s")
        lines.append(f"  Efficiency: {point.efficiency:.1%}")
        lines.append(f"  Bound: {'Memory' if point.is_memory_bound else 'Compute'}")
        lines.append("")
        for g in model.get_optimization_guidance(point):
            lines.append(g)
            
    return "\n".join(lines)

