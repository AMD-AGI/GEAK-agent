# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

"""
Display utilities for GEAK-Agent Tutorial
Handles all formatting and results display logic
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

def setup_environment():
    """
    Auto-detect and configure paths for the tutorial.
    Returns: (TUTORIAL_DIR, SRC_DIR, CORPUS_PATH, TutorialDataloader)
    
    This also imports and returns TutorialDataloader for convenience.
    """
    import sys
    
    # Detect tutorial directory
    cwd = os.getcwd()
    if cwd.endswith('tutorial'):
        tutorial_dir = cwd
    elif os.path.exists(os.path.join(cwd, 'tutorial')):
        tutorial_dir = os.path.join(cwd, 'tutorial')
    else:
        tutorial_dir = cwd
    
    geak_agent_dir = os.path.dirname(tutorial_dir)
    src_dir = os.path.join(geak_agent_dir, 'src')
    
    # Add to path if not already there
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    if tutorial_dir not in sys.path:
        sys.path.insert(0, tutorial_dir)
    
    corpus_path = os.path.join(src_dir, 'dataloaders', 'TB_eval', 'train_crawl.json')
    
    # Now we can import TutorialDataloader since paths are set
    from tutorial_utils.tutorial_dataloader import TutorialDataloader
    
    return tutorial_dir, src_dir, corpus_path, TutorialDataloader


def print_header(title, width=60):
    """Print a formatted header box."""
    print('╔' + '═' * width + '╗')
    print('║' + f' {title}'.ljust(width) + '║')
    print('╚' + '═' * width + '╝')


def print_section(title, width=60):
    """Print a section divider."""
    print('\n' + '═' * width)
    print(f' {title}')
    print('═' * width)


def print_config(config_dict, title="Configuration"):
    """Print configuration in a nice format."""
    print('\n' + '═' * 60)
    print(f' {title}')
    print('═' * 60)
    for k, v in config_dict.items():
        print(f'  • {k}: {v}')
    print('═' * 60)


def display_kernel_info(problem_states):
    """Display kernel information in formatted boxes."""
    for ps in problem_states:
        print('\n┌' + '─' * 70 + '┐')
        print(f'│ 📄 {ps.filename.ljust(65)}│')
        print('├' + '─' * 70 + '┤')
        
        # Show instruction preview
        preview = ps.instruction[:400].replace('\n', ' ')
        for i in range(0, len(preview), 68):
            line = preview[i:i+68]
            print(f'│ {line.ljust(68)} │')
        if len(ps.instruction) > 400:
            print(f'│ {"...[truncated]".ljust(68)} │')
        
        print('├' + '─' * 70 + '┤')
        print(f'│ ✓ Has reference code: {str(ps.label is not None).ljust(46)}│')
        print(f'│ ✓ Has test code: {str(ps.test_code is not None).ljust(51)}│')
        print('└' + '─' * 70 + '┘')


def load_results(output_dir):
    """
    Load the most recent results from the output directory.
    Returns: (results_dict, iteration_number) or (None, None)
    """
    pattern = os.path.join(output_dir, 'tutorial_results_mem_*.json')
    mem_files = sorted(glob.glob(pattern))
    
    if not mem_files:
        return None, None
    
    latest = mem_files[-1]
    iteration = os.path.basename(latest).split('_')[-1].replace('.json', '')
    
    with open(latest, 'r') as f:
        results = json.load(f)
    
    return results, iteration


def display_results_summary(results, iteration=None):
    """Display results in a formatted table."""
    if not results:
        print('⚠️ No results found. Run the agent first.')
        return
    
    print('\n╔' + '═' * 70 + '╗')
    print('║' + ' 📊 GEAK-Agent Results Summary'.ljust(70) + '║')
    if iteration:
        print('║' + f' Iteration: {iteration}'.ljust(70) + '║')
    print('╠' + '═' * 70 + '╣')
    print('║' + ' Kernel              │ Call │ Exec │ Perf │ Speedup'.ljust(70) + '║')
    print('╠' + '─' * 70 + '╣')
    
    for kernel_name, data in results.items():
        call = '✓' if data.get('pass_call') else '✗'
        exe = '✓' if data.get('pass_exe') else '✗'
        perf = '✓' if data.get('pass_perf') else '✗'
        
        speedup = '-'
        perf_candidates = data.get('perf_candidates', [])
        if perf_candidates:
            speedups = [c[1] for c in perf_candidates if len(c) > 1 and c[1]]
            if speedups:
                speedup = f'{max(speedups):.4f}x'
        
        row = f' {kernel_name[:18].ljust(18)} │  {call}   │  {exe}   │  {perf}   │ {speedup.ljust(8)}'
        print('║' + row.ljust(70) + '║')
    
    print('╚' + '═' * 70 + '╝')
    print('\n Legend: ✓ = Passed, ✗ = Failed')
    print(' Speedup: reference_time / generated_time (>1.0 = faster)')


def display_generated_code(results, kernel_name=None):
    """Display generated code for one or all kernels."""
    if not results:
        print('⚠️ No results available.')
        return
    
    kernels = [kernel_name] if kernel_name else list(results.keys())
    
    for name in kernels:
        if name not in results:
            print(f'⚠️ Kernel "{name}" not found.')
            continue
        
        data = results[name]
        
        # Get best available code
        code = data.get('exe_candidate') or data.get('call_candidate')
        if not code and data.get('perf_candidates'):
            code = data['perf_candidates'][0][0]
        
        if not code:
            print(f'⚠️ No code available for "{name}".')
            continue
        
        # Status
        status = '✓ Correct' if data.get('pass_exe') else '✗ Needs fixing'
        
        print('\n┌' + '─' * 70 + '┐')
        print(f'│ 🔧 {name} - {status}'.ljust(71) + '│')
        
        # Speedup if available
        if data.get('pass_perf'):
            speedups = [c[1] for c in data.get('perf_candidates', []) if len(c) > 1]
            if speedups:
                print(f'│    Speedup: {max(speedups):.4f}x'.ljust(71) + '│')
        
        print('└' + '─' * 70 + '┘')
        print(code)
        print('\n' + '─' * 72)


def display_strategy(results, kernel_name=None):
    """Display optimization strategies."""
    if not results:
        return
    
    kernels = [kernel_name] if kernel_name else list(results.keys())
    
    for name in kernels:
        if name not in results:
            continue
        
        data = results[name]
        strategy = data.get('temp_strategy') or data.get('perf_strategy')
        
        print(f'\n📋 {name} - Optimization Strategy:\n')
        if strategy:
            print(f'> {strategy[:500]}')
            if len(strategy) > 500:
                print('\n> ...[truncated]')
        else:
            print('> No strategy recorded.')

def plot_speedup_curve_all_iterations(output_dir, kernel_name=None):
    """
    Plot speedup progress by loading ALL iteration result files from output_dir.
    This ensures we capture valid speedups from every step of the optimization.
    """
    sns.set_theme(style="whitegrid")
    
    # 1. Find all result files
    # Pattern is tutorial_results_mem_X.json
    pattern = os.path.join(output_dir, 'tutorial_results_mem_*.json')
    result_files = sorted(glob.glob(pattern), key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    
    if not result_files:
        print('⚠️ No result files found in output directory.')
        return

    # 2. Collect data per kernel across iterations
    kernel_history = {} # {kernel_name: {iter_num: max_speedup}}
    
    for file_path in result_files:
        try:
            # Extract iteration number from filename (e.g., ..._mem_0.json -> 0)
            iter_num = int(os.path.basename(file_path).split('_')[-1].split('.')[0])
            real_iter = iter_num + 1  # Display as 1-based index
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for name, content in data.items():
                if kernel_name and name != kernel_name:
                    continue
                    
                if name not in kernel_history:
                    kernel_history[name] = {'iterations': [], 'speedups': []}
                
                # Check for valid speedups in this iteration's candidates
                # 'perf_candidates' usually holds [code, speedup, ...]
                candidates = content.get('perf_candidates', [])
                if candidates:
                    valid_speedups = [c[1] for c in candidates if len(c) > 1 and isinstance(c[1], (int, float)) and c[1] > 0]
                    if valid_speedups:
                        best_speedup = max(valid_speedups)
                        kernel_history[name]['iterations'].append(real_iter)
                        kernel_history[name]['speedups'].append(best_speedup)
                        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    # 3. Plot for each kernel found
    for name, history in kernel_history.items():
        iterations = history['iterations']
        speedups = history['speedups']
        
        if not speedups:
            print(f"⚠️ No valid speedup data found for {name}")
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Plot line
        plt.plot(iterations, speedups, marker='o', linestyle='-', linewidth=2, markersize=8, color='#00a4ef')
        
        # Styling
        plt.title(f'Correction & Optimization Progress: {name}', fontsize=14, pad=15)
        plt.xlabel('Agent Iteration', fontsize=12)
        plt.ylabel('Speedup (vs Reference)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Force integer x-axis ticks
        if len(iterations) > 0:
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlim(min(iterations)-0.5, max(iterations)+0.5)
        
        # Annotate values
        for x, y in zip(iterations, speedups):
            plt.annotate(
                f'{y:.2f}x', 
                (x, y), 
                textcoords="offset points", 
                xytext=(0, 10), 
                ha='center', 
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#00a4ef", alpha=0.8)
            )
            
        plt.tight_layout()
        plt.show()