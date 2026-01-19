# 🚀 Mini-Kernel Agent

**Autonomous GPU Kernel Optimization using Profiler-Guided Evolutionary Search**

Mini-Kernel Agent automatically optimizes GPU kernels (Triton, CUDA, HIP) by:
1. **Profiling** - Identifying bottlenecks (latency, memory, compute)
2. **Evolving** - Using OpenEvolve to explore optimization strategies
3. **Generating** - Creating actual kernel code variants (not just wrappers!)
4. **Benchmarking** - Measuring correctness and performance
5. **Reporting** - Producing detailed optimization reports

## 🎯 Key Features

- **Autonomous Operation** - Minimal user intervention required
- **Profiler-Guided** - Optimizations based on actual hardware bottlenecks
- **Evolutionary Search** - OpenEvolve explores, exploits, merges strategies
- **Kernel Code Generation** - Generates actual Triton/HIP kernel variants
- **Parameter Tuning** - Automatically tunes block sizes, warps, splits, etc.
- **Correctness Verification** - Ensures optimized kernels produce correct results

## 📦 Quick Start

### Prerequisites

- Docker with GPU support (ROCm or CUDA)
- Python 3.10+

### Installation

```bash
# Clone the repository
git clone <repo-url> mini-kernel-agent
cd mini-kernel-agent

# Run setup (uses Docker - no local dependencies needed!)
./setup.sh
```

### Basic Usage

```bash
# Optimize a kernel (runs in Docker automatically)
./mini-kernel optimize /path/to/your/kernel.py

# With specific GPU
./mini-kernel optimize /path/to/kernel.py --gpu 0

# Extended evolution (more generations)
./mini-kernel optimize /path/to/kernel.py --generations 10 --population 16
```

### Example: Optimize the Add Kernel

```bash
# Run the included example
./mini-kernel optimize examples/add_kernel/kernel.py --gpu 0

# Or run interactively in Docker
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  -v $(pwd):/workspace \
  -w /workspace \
  lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x \
  python -m mini_kernel.cli examples/add_kernel/kernel.py
```

## 🔧 How It Works

### 1. Profiler Analysis
```
[PROFILER] Analyzing kernel bottlenecks...
[PROFILER] Bottleneck: latency (93% launch overhead)
  → Recommended: HIP Graph, Kernel Fusion, Persistent Kernels
```

### 2. OpenEvolve Brain
```
[OpenEvolve] Population: 12 genomes
[OpenEvolve] Generation 1/5
  ⭐ NEW BEST: params_n64_h4_s16_w4
     Latency: 15.2 μs → 12.8 μs
     Speedup: 1.19x
```

### 3. Code Generation
The agent generates actual kernel variants:
- **Parameter variants** - Different block sizes, warps, splits
- **Fused variants** - Combine multiple kernels
- **Algorithmic variants** - Different tiling, memory patterns

### 4. Results
```
============================================================
  OPTIMIZATION COMPLETE
============================================================
  Baseline:   17.11 μs
  Best:       14.32 μs
  Speedup:    1.19x
  Strategy:   fused_kernel + params_n128_h8_s8_w4
============================================================
```

## 📁 Project Structure

```
mini-kernel-agent/
├── mini_kernel/           # Core package
│   ├── __init__.py
│   ├── cli.py             # Command-line interface
│   ├── profiler.py        # Bottleneck analysis
│   ├── openevolve_brain.py # Evolutionary optimization
│   ├── kernel_codegen.py  # Kernel code generation
│   └── harness.py         # Test harness generation
├── examples/              # Example kernels
│   ├── add_kernel/        # Simple vector add
│   ├── matmul_kernel/     # Matrix multiplication
│   └── attention_kernel/  # Flash attention
├── configs/               # Configuration presets
│   ├── default.yaml
│   ├── aggressive.yaml    # More exploration
│   └── quick.yaml         # Fast optimization
├── docs/                  # Documentation
├── setup.sh               # Setup script
├── mini-kernel            # CLI wrapper
└── README.md
```

## ⚙️ Configuration

### Command Line Options

```bash
./mini-kernel optimize <kernel_path> [options]

Options:
  --gpu GPU_ID          GPU device to use (default: 0)
  --generations N       Number of evolution generations (default: 5)
  --population N        Population size (default: 12)
  --timeout SECONDS     Maximum optimization time (default: 1800)
  --output DIR          Output directory for results
  --config FILE         Configuration preset file
```

### Configuration File (YAML)

```yaml
# configs/default.yaml
evolution:
  population_size: 12
  generations: 5
  mutation_rate: 0.3
  crossover_rate: 0.7
  elite_count: 2

kernel:
  generate_variants: true
  variant_types:
    - params
    - fused
    - persistent_v2
  
  parameter_ranges:
    block_n: [32, 64, 128, 256]
    block_h: [1, 2, 4, 8, 16]
    num_splits: [4, 8, 16, 32]
    num_warps: [2, 4, 8]

profiler:
  warmup_iters: 100
  benchmark_iters: 1000

docker:
  image: lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x
```

## 🧬 OpenEvolve Brain

The evolutionary optimization uses:

1. **Population Initialization**
   - Single optimizations (HIP Graph, Fusion, etc.)
   - Known good combinations
   - Random exploration

2. **Fitness Evaluation**
   - Correctness check (must pass!)
   - Latency measurement
   - Speedup calculation

3. **Selection** (Tournament)
   - Favor solutions that work and are fast

4. **Crossover** (Strategy Merging)
   - Combine successful optimization strategies
   - Merge kernel parameters from parents

5. **Mutation** (Exploration)
   - Flip optimizations on/off
   - Mutate kernel parameters (block sizes, etc.)

## 📊 Supported Kernel Types

| Type | Auto-Detection | Code Generation |
|------|---------------|-----------------|
| Triton | ✅ | ✅ Full variants |
| CUDA/HIP | ✅ | ⚠️ Parameter only |
| PyTorch | ✅ | ✅ Wrapper opts |
| CK (Composable Kernel) | ✅ | ⚠️ Planned |

## 🔬 Profiler Bottleneck Types

| Bottleneck | Symptoms | Recommended Optimizations |
|------------|----------|--------------------------|
| **Latency** | High launch overhead | HIP Graph, Fusion, Persistent |
| **Memory** | Low bandwidth util | Coalescing, Vectorization, LDS |
| **Compute** | Low ALU util | Better tiling, Warp efficiency |
| **Balanced** | Mixed | General tuning |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your kernel type or optimization strategy
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- OpenEvolve for evolutionary optimization concepts
- AMD ROCm team for profiler tools
- Triton for the kernel DSL

