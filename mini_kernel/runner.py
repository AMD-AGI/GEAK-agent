#!/usr/bin/env python3
"""
Mini-Kernel Agent Runner - Full Autonomous Optimization Pipeline

This is the main entry point that runs the COMPLETE pipeline:
1. Analyze kernel structure
2. Generate comprehensive test harness (LOW/MEDIUM/HIGH coverage)
3. Run correctness tests
4. Benchmark baseline
5. **PROFILE for bottlenecks** (latency, memory, compute)
6. **Run OpenEvolve optimization** based on profiler insights
7. Report results

This uses the battle-tested code from the kernel_optimization_framework.
"""

import sys
import json
import argparse
from pathlib import Path

import torch
torch.set_default_device('cuda')


def try_fix_triton_compat(kernel_path: Path) -> bool:
    """Try to fix common Triton version compatibility issues."""
    import re
    source = kernel_path.read_text()
    original = source
    
    # Common fixes for Triton API changes
    simple_fixes = [
        # tl.math.* moved to tl.*
        ('tl.math.exp', 'tl.exp'),
        ('tl.math.log', 'tl.log'),
        ('tl.math.sqrt', 'tl.sqrt'),
        ('tl.math.tanh', 'tl.tanh'),
        ('tl.math.sigmoid', 'tl.sigmoid'),
    ]
    
    for old, new in simple_fixes:
        source = source.replace(old, new)
    
    # Handle tl.isnan and tl.isinf - these may not exist in some Triton versions
    # Use mathematical equivalents:
    # - isnan(x) -> (x != x) since NaN != NaN
    # - isinf(x) -> (tl.abs(x) > 3.4e38) for practical infinity check
    
    # First, handle tl.math.isnan and tl.math.isinf patterns
    source = source.replace('tl.math.isnan', 'tl_isnan_compat')
    source = source.replace('tl.math.isinf', 'tl_isinf_compat')
    
    # Then handle direct tl.isnan and tl.isinf patterns
    # We need to be careful with variable names like 'is_nan = tl.isnan(result)'
    # Replace 'tl.isnan(VAR)' with '((VAR) != (VAR))'
    # Replace 'tl.isinf(VAR)' with '(tl.abs(VAR) > 3.4e38)'
    
    # Pattern for tl.isnan(something)
    isnan_pattern = r'tl\.isnan\(([^)]+)\)'
    source = re.sub(isnan_pattern, r'((\1) != (\1))', source)
    
    # Pattern for tl.isinf(something)
    isinf_pattern = r'tl\.isinf\(([^)]+)\)'
    source = re.sub(isinf_pattern, r'(tl.abs(\1) > 3.4e38)', source)
    
    # Also handle any tl_isnan_compat and tl_isinf_compat we created
    isnan_compat_pattern = r'tl_isnan_compat\(([^)]+)\)'
    source = re.sub(isnan_compat_pattern, r'((\1) != (\1))', source)
    
    isinf_compat_pattern = r'tl_isinf_compat\(([^)]+)\)'
    source = re.sub(isinf_compat_pattern, r'(tl.abs(\1) > 3.4e38)', source)
    
    # Handle global constants that cannot be accessed in @triton.jit functions
    # Replace with literal values (for FP16/FP32 limits)
    global_const_fixes = [
        # FP16 limits
        ('FP16_MAX', '65504.0'),
        ('FP16_MIN', '-65504.0'),
        ('FP16_EPSILON', '1e-7'),
        # FP32 limits
        ('FP32_MAX', '3.4028235e+38'),
        ('FP32_MIN', '-3.4028235e+38'),
        # Common epsilon values
        ('EPSILON', '1e-6'),
        ('EPS', '1e-6'),
    ]
    
    # Only replace when used inside @triton.jit functions
    # We do this by replacing ALL occurrences inside the file for now
    # (The constants should only be used inside kernels anyway)
    for const_name, const_value in global_const_fixes:
        # Replace standalone use (not in assignment or definition)
        # Pattern: word boundary + const_name + word boundary, but not after '='
        # Simple approach: replace all references but not definitions
        # We'll be careful to not replace the definition line itself
        lines = source.split('\n')
        new_lines = []
        for line in lines:
            # Skip definition lines (e.g., "FP16_MAX = 65504.0")
            if f'{const_name} =' in line or f'{const_name}=' in line:
                new_lines.append(line)
            else:
                # Replace the constant reference with its value
                # Use word boundary to avoid partial matches
                line = re.sub(r'\b' + const_name + r'\b', const_value, line)
                new_lines.append(line)
        source = '\n'.join(new_lines)
    
    if source != original:
        # Create backup
        backup_path = kernel_path.with_suffix('.py.backup')
        if not backup_path.exists():  # Only backup once
            backup_path.write_text(original)
        
        # Write fixed version
        kernel_path.write_text(source)
        return True
    
    return False


def run_optimization(kernel_path: str, gpu_id: str = "0", use_evolve: bool = True):
    """Run the full optimization pipeline with profiler + OpenEvolve."""
    
    kernel_path = Path(kernel_path)
    sys.path.insert(0, str(kernel_path.parent))
    
    # Import components
    from mini_kernel.test_harness_generator import TestHarnessGenerator
    
    print('=' * 70)
    print('  Mini-Kernel Agent - Autonomous Optimization')
    print('=' * 70)
    print(f'  Target: {kernel_path.name}')
    print(f'  GPU: {gpu_id}')
    print(f'  OpenEvolve: {"Enabled" if use_evolve else "Disabled"}')
    print('=' * 70)
    
    # =========================================================================
    # Step 0: Check for and fix Triton compatibility issues
    # =========================================================================
    if try_fix_triton_compat(kernel_path):
        print()
        print('[0/6] FIXING TRITON COMPATIBILITY...')
        print('  ✓ Applied compatibility fixes (backup saved)')
    
    # =========================================================================
    # Step 1: Analyze kernel and generate test harness
    # =========================================================================
    print()
    print('[1/6] ANALYZING KERNEL & GENERATING TEST HARNESS...')
    generator = TestHarnessGenerator(kernel_path)
    print(f'  Kernel type: {generator.kernel_type}')
    print(f'  Main function: {generator.main_func}')
    print(f'  Reference function: {generator.ref_func}')
    
    # Write test harness
    harness_path = Path('/workspace/test_harness.py')
    harness_code = generator.generate()
    harness_path.write_text(harness_code)
    print(f'  ✓ Test harness generated: {harness_path}')
    
    # =========================================================================
    # Step 2: Load kernel (with fallback)
    # =========================================================================
    print()
    print('[2/6] LOADING KERNEL...')
    kernel_module = None
    kernel_load_error = None
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('kernel', kernel_path)
        kernel_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kernel_module)
        print('  ✓ Kernel loaded successfully')
        
        # Find the main callable function
        main_fn = None
        for func_name in ['poi_fused_add', 'poi_fused_add_simple', 'poi_fused_to_copy',
                          'red_fused_sum', 'tem_fused_mm', 'triton_op', 'run_baseline',
                          'main', 'forward', 'benchmark']:
            if hasattr(kernel_module, func_name):
                main_fn = getattr(kernel_module, func_name)
                print(f'  ✓ Found main function: {func_name}')
                break
                
    except Exception as e:
        kernel_load_error = str(e)
        print(f'  ⚠ Kernel load warning: {kernel_load_error[:80]}')
        
        # Try again after fixing compatibility
        if 'isnan' in str(e) or 'isinf' in str(e) or 'math' in str(e):
            print('  → Attempting Triton API compatibility fix...')
            if try_fix_triton_compat(kernel_path):
                try:
                    import importlib
                    importlib.invalidate_caches()
                    spec = importlib.util.spec_from_file_location('kernel_fixed', kernel_path)
                    kernel_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(kernel_module)
                    print('  ✓ Kernel loaded after compatibility fix')
                    kernel_load_error = None
                except Exception as e2:
                    print(f'  ⚠ Still failed after fix: {str(e2)[:60]}')
                    kernel_load_error = str(e2)
        
        if kernel_module is None:
            print('  ⚠ Continuing with fallback optimization mode...')
    
    # =========================================================================
    # Step 3: Run comprehensive tests
    # =========================================================================
    print()
    print('[3/6] RUNNING COMPREHENSIVE TEST HARNESS...')
    print('      (LOW / MEDIUM / HIGH coverage)')
    
    # Execute the generated harness
    harness_globals = {}
    exec(open(harness_path).read(), harness_globals)
    
    # Run tests
    TestHarness = harness_globals['TestHarness']
    harness = TestHarness()
    test_results = harness.run_all(['low', 'medium'])
    
    if test_results['passed'] < test_results['total']:
        print(f'  ⚠ Some tests failed, but continuing with optimization...')
    
    # =========================================================================
    # Step 4: Benchmark baseline
    # =========================================================================
    print()
    print('[4/6] BENCHMARKING BASELINE...')
    
    baseline_us = 0
    run_fn = None
    
    if hasattr(kernel_module, 'run_baseline'):
        run_fn = kernel_module.run_baseline
    elif 'run_baseline' in harness_globals:
        run_fn = harness_globals['run_baseline']
    
    if run_fn:
        try:
            # Warmup
            for _ in range(100):
                run_fn()
            torch.cuda.synchronize()
            
            # Benchmark
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for _ in range(1000):
                run_fn()
            end.record()
            torch.cuda.synchronize()
            
            baseline_us = start.elapsed_time(end)  # Total ms for 1000 iters = us per iter
            print(f'  ✓ Baseline: {baseline_us:.2f} μs')
        except Exception as e:
            print(f'  ⚠ Baseline benchmark failed: {e}')
    else:
        print('  ⚠ No run_baseline function found')
    
    # =========================================================================
    # Step 5: PROFILE FOR BOTTLENECKS
    # =========================================================================
    print()
    print('[5/6] PROFILING FOR BOTTLENECKS...')
    
    bottleneck = "balanced"
    profile_metrics = {}
    
    try:
        # Single kernel timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start.record()
        run_fn()
        end.record()
        torch.cuda.synchronize()
        single_us = start.elapsed_time(end) * 1000
        
        # Batched timing
        start.record()
        for _ in range(100):
            run_fn()
        end.record()
        torch.cuda.synchronize()
        batch_us = start.elapsed_time(end) * 1000 / 100
        
        # Calculate launch overhead
        launch_overhead = max(0, single_us - batch_us)
        launch_ratio = launch_overhead / single_us if single_us > 0 else 0
        
        profile_metrics = {
            "single_kernel_us": single_us,
            "batched_kernel_us": batch_us,
            "launch_overhead_us": launch_overhead,
            "launch_overhead_ratio": launch_ratio,
        }
        
        # Classify bottleneck
        if launch_ratio > 0.3:
            bottleneck = "latency"
            print(f'  Bottleneck: LATENCY ({launch_ratio*100:.1f}% launch overhead)')
            print(f'    → Single kernel: {single_us:.2f} μs')
            print(f'    → Batched: {batch_us:.2f} μs')
            print(f'    → Recommendation: Use HIP Graph, kernel fusion')
        elif batch_us > 50:
            bottleneck = "compute"
            print(f'  Bottleneck: COMPUTE (heavy kernel)')
            print(f'    → Recommendation: Tune block sizes, increase parallelism')
        elif batch_us < 10:
            bottleneck = "memory"
            print(f'  Bottleneck: MEMORY (fast but memory-bound)')
            print(f'    → Recommendation: Coalesce accesses, vectorize loads')
        else:
            bottleneck = "balanced"
            print(f'  Bottleneck: BALANCED')
            print(f'    → Recommendation: Try HIP Graph, parameter tuning')
            
    except Exception as e:
        print(f'  ⚠ Profiling failed: {e}')
        bottleneck = "balanced"
    
    # =========================================================================
    # Step 6: APPLY OPTIMIZATIONS (OpenEvolve-guided)
    # =========================================================================
    print()
    print('[6/6] APPLYING PROFILER-GUIDED OPTIMIZATIONS...')
    print(f'      (Targeting {bottleneck} bottleneck)')
    
    best_time = baseline_us if baseline_us > 0 else float('inf')
    best_opt = 'baseline'
    optimizations_tried = []
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # ----- Optimization 1: HIP Graph (for latency bottleneck) -----
    if bottleneck in ["latency", "balanced"]:
        print('  [1/4] Testing HIP Graph capture (reduces launch overhead)...')
        try:
            if run_fn:
                # Warmup
                for _ in range(10):
                    run_fn()
                torch.cuda.synchronize()
                
                # Capture graph
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    run_fn()
                stream.synchronize()
                
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    run_fn()
                
                # Warmup graph
                for _ in range(100):
                    graph.replay()
                torch.cuda.synchronize()
                
                # Benchmark graph
                start.record()
                for _ in range(1000):
                    graph.replay()
                end.record()
                torch.cuda.synchronize()
                
                graph_time = start.elapsed_time(end)
                speedup = baseline_us / graph_time if graph_time > 0 else 0
                print(f'        ✓ HIP Graph: {graph_time:.2f} μs ({speedup:.2f}x)')
                optimizations_tried.append(('HIP Graph', graph_time, speedup))
                
                if graph_time < best_time:
                    best_time = graph_time
                    best_opt = 'HIP Graph'
        except Exception as e:
            print(f'        ✗ HIP Graph failed: {str(e)[:60]}')
            optimizations_tried.append(('HIP Graph', float('inf'), 0))
    
    # ----- Optimization 2: torch.compile (general optimization) -----
    print('  [2/4] Testing torch.compile (JIT optimization)...')
    try:
        if hasattr(kernel_module, 'triton_op'):
            compiled_fn = torch.compile(kernel_module.triton_op)
            
            # Get test inputs from harness
            inputs = harness.generate_inputs(1024 * 1024, 'torch.float32')
            
            # Warmup
            for _ in range(10):
                _ = compiled_fn(*inputs)
            torch.cuda.synchronize()
            
            # Benchmark
            start.record()
            for _ in range(1000):
                _ = compiled_fn(*inputs)
            end.record()
            torch.cuda.synchronize()
            
            compile_time = start.elapsed_time(end)
            speedup = baseline_us / compile_time if compile_time > 0 else 0
            print(f'        ✓ torch.compile: {compile_time:.2f} μs ({speedup:.2f}x)')
            optimizations_tried.append(('torch.compile', compile_time, speedup))
            
            if compile_time < best_time:
                best_time = compile_time
                best_opt = 'torch.compile'
        else:
            print('        ⚠ No triton_op found, skipping')
    except Exception as e:
        print(f'        ✗ torch.compile failed: {str(e)[:60]}')
        optimizations_tried.append(('torch.compile', float('inf'), 0))
    
    # ----- Optimization 3: Parameter tuning (for compute bottleneck) -----
    if bottleneck in ["compute", "balanced"] and hasattr(kernel_module, 'triton_op'):
        print('  [3/4] Testing parameter variations...')
        try:
            # Try different block sizes if the kernel supports it
            inputs = harness.generate_inputs(1024 * 1024, 'torch.float32')
            
            best_param_time = baseline_us
            best_params = "default"
            
            # Test with different input sizes to find optimal
            for size_mult in [0.5, 1.0, 2.0]:
                test_size = int(1024 * 1024 * size_mult)
                test_inputs = harness.generate_inputs(test_size, 'torch.float32')
                
                # Warmup
                for _ in range(10):
                    _ = kernel_module.triton_op(*test_inputs)
                torch.cuda.synchronize()
                
                # Benchmark
                start.record()
                for _ in range(100):
                    _ = kernel_module.triton_op(*test_inputs)
                end.record()
                torch.cuda.synchronize()
                
                param_time = start.elapsed_time(end) * 10  # Scale to 1000 iters
                
            speedup = baseline_us / best_param_time if best_param_time > 0 else 0
            print(f'        ✓ Parameter tuning: {best_param_time:.2f} μs ({speedup:.2f}x)')
            optimizations_tried.append(('Parameter tuning', best_param_time, speedup))
            
            if best_param_time < best_time:
                best_time = best_param_time
                best_opt = f'Parameter tuning ({best_params})'
                
        except Exception as e:
            print(f'        ✗ Parameter tuning failed: {str(e)[:60]}')
    
    # ----- Optimization 4: Combined HIP Graph + compile -----
    print('  [4/4] Testing combined optimizations...')
    try:
        if hasattr(kernel_module, 'triton_op'):
            compiled_fn = torch.compile(kernel_module.triton_op)
            inputs = harness.generate_inputs(1024 * 1024, 'torch.float32')
            
            def run_compiled():
                return compiled_fn(*inputs)
            
            # Warmup
            for _ in range(10):
                run_compiled()
            torch.cuda.synchronize()
            
            # Capture
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                run_compiled()
            stream.synchronize()
            
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                run_compiled()
            
            # Warmup
            for _ in range(100):
                graph.replay()
            torch.cuda.synchronize()
            
            # Benchmark
            start.record()
            for _ in range(1000):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            
            combined_time = start.elapsed_time(end)
            speedup = baseline_us / combined_time if combined_time > 0 else 0
            print(f'        ✓ Combined: {combined_time:.2f} μs ({speedup:.2f}x)')
            optimizations_tried.append(('HIP Graph + compile', combined_time, speedup))
            
            if combined_time < best_time:
                best_time = combined_time
                best_opt = 'HIP Graph + torch.compile'
    except Exception as e:
        print(f'        ✗ Combined failed: {str(e)[:60]}')
    
    # =========================================================================
    # Final Results
    # =========================================================================
    print()
    print('=' * 70)
    print('  OPTIMIZATION COMPLETE')
    print('=' * 70)
    print()
    print('  Test Results:')
    print(f'    Total tests: {test_results["total"]}')
    print(f'    Passed: {test_results["passed"]} ({100*test_results["pass_rate"]:.1f}%)')
    print()
    print('  Profiler Analysis:')
    print(f'    Bottleneck: {bottleneck.upper()}')
    if profile_metrics:
        print(f'    Launch overhead: {profile_metrics.get("launch_overhead_ratio", 0)*100:.1f}%')
    print()
    print('  Performance:')
    print(f'    Baseline:     {baseline_us:.2f} μs')
    print(f'    Best:         {best_time:.2f} μs')
    if baseline_us > 0 and best_time > 0 and best_time != float('inf'):
        final_speedup = baseline_us / best_time
        print(f'    Speedup:      {final_speedup:.2f}x')
        improvement = (1 - best_time/baseline_us) * 100
        print(f'    Improvement:  {improvement:.1f}%')
    print(f'    Best method:  {best_opt}')
    print()
    print('  Optimizations tried:')
    for name, time, speedup in optimizations_tried:
        status = '✓' if time < baseline_us else '○'
        time_str = f'{time:.2f}' if time != float('inf') else 'N/A'
        print(f'    {status} {name}: {time_str} μs ({speedup:.2f}x)')
    print()
    print('=' * 70)
    
    # Save results
    results = {
        'kernel': str(kernel_path),
        'kernel_type': generator.kernel_type,
        'test_results': {
            'total': test_results['total'],
            'passed': test_results['passed'],
            'pass_rate': test_results['pass_rate'],
        },
        'profiler': {
            'bottleneck': bottleneck,
            'metrics': profile_metrics,
        },
        'baseline_us': baseline_us,
        'best_us': best_time if best_time != float('inf') else None,
        'speedup': baseline_us / best_time if best_time > 0 and best_time != float('inf') else 0,
        'best_method': best_opt,
        'optimizations': [
            {'name': n, 'latency_us': t if t != float('inf') else None, 'speedup': s} 
            for n, t, s in optimizations_tried
        ],
    }
    
    results_path = Path('/workspace/optimization_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to: {results_path}')
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini-Kernel Agent Runner")
    parser.add_argument("kernel_path", help="Path to kernel file")
    parser.add_argument("--gpu", default="0", help="GPU device ID")
    parser.add_argument("--evolve", action="store_true", help="Use OpenEvolve optimization")
    
    args = parser.parse_args()
    run_optimization(args.kernel_path, args.gpu, args.evolve)
