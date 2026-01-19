# 🚀 Mini-Kernel Agent

**Autonomous GPU Kernel Optimization using LLM + Profiler-Guided Evolutionary Search**

Mini-Kernel Agent automatically optimizes GPU kernels (Triton, HIP, CUDA) by:
1. **Detecting** - Auto-discovers kernels in files/folders
2. **Profiling** - Uses `rocprof-compute` for hardware bottleneck analysis
3. **Analyzing** - LLM (Claude) interprets profiler output
4. **Evolving** - OpenEvolve generates, mutates, and crossovers optimization strategies
5. **Evaluating** - Measures speedup for each candidate
6. **Reporting** - Returns best optimization with full history

## 🎯 Key Features

- **Fully Autonomous** - Just point it at a kernel file or folder
- **LLM-Powered** - Claude analyzes bottlenecks and generates optimizations
- **Evolutionary Search** - OpenEvolve explores, mutates, crossovers strategies
- **Real-time Logging** - See every step: profiling, LLM calls, evaluations
- **Generic** - Works with single files, modules, or folders with multiple kernels

## 📦 Quick Start

### Prerequisites

- Docker with GPU support (ROCm)
- Python 3.10+
- **API Key** for AMD LLM Gateway (Claude)

### Installation

```bash
# Clone the repository
git clone git@github.com:AMD-AGI/GEAK-agent.git mini-kernel-agent
cd mini-kernel-agent
git checkout opti

# Install Python dependencies
pip install openai tenacity
```

### ⚠️ Set Your API Key (REQUIRED)

```bash
# Set your AMD LLM Gateway API key
export MINI_KERNEL_API_KEY="your-api-key-here"

# Verify it's set
echo $MINI_KERNEL_API_KEY
```

The agent uses Claude (via AMD LLM Gateway) for intelligent optimization.
**You must set this environment variable before running.**

### Run the Agent

```bash
# Single kernel file
python run_agent.py examples/add_kernel/kernel.py

# Module folder
python run_agent.py /path/to/your/module/

# With options
python run_agent.py kernel.py --generations 5 --population 5 --target 2.0
```

## 🔧 How It Works - 5-Step Pipeline

```
[1/5] DETECTING KERNELS
      Scanning path, finding Triton/HIP/CUDA kernels

[2/5] PROFILING WITH ROCPROF-COMPUTE
      Hardware-level bottleneck analysis (latency, memory, compute)

[3/5] MEASURING BASELINE PERFORMANCE
      Benchmarking original kernel latency

[4/5] LLM ANALYSIS + OPENEVOLVE OPTIMIZATION
      ├── LLM analyzes profiler output → identifies bottleneck
      ├── Generates initial population of optimization strategies
      ├── Evaluates each candidate (runs in Docker)
      ├── Evolution loop:
      │   ├── MUTATION: LLM modifies parameters/algorithm
      │   ├── CROSSOVER: LLM combines two good strategies
      │   └── NEW: LLM generates fresh approaches
      └── Returns best solution

[5/5] FINAL REPORT
      Best speedup, strategy name, saved code
```

## ⚙️ Command Line Options

```bash
python run_agent.py <path> [options]

Arguments:
  path                    Path to kernel file or module directory

Options:
  --generations, -g N     Number of evolution generations (default: 3)
  --population, -p N      Population size per generation (default: 3)
  --target, -t SPEEDUP    Target speedup, e.g., 1.5 for 50% faster (default: 1.5)
  --quiet, -q             Reduce output verbosity
```

## 🧬 OpenEvolve Brain

The evolutionary optimization uses LLM as an intelligent mutation/crossover operator:

| Operation | Description |
|-----------|-------------|
| **Generate** | LLM creates new optimization based on bottleneck |
| **Mutate** | LLM modifies parameters or algorithm of existing solution |
| **Crossover** | LLM combines best aspects of two parent solutions |
| **Select** | Tournament selection favors higher fitness |
| **Elite** | Best solutions always survive to next generation |

## 🔬 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MINI_KERNEL_API_KEY` | ✅ Yes | API key for AMD LLM Gateway |
| `MINI_KERNEL_API_URL` | No | Custom API endpoint |
| `MINI_KERNEL_DOCKER` | No | Docker image (default: lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x) |

## 📁 Project Structure

```
mini-kernel-agent/
├── run_agent.py           # 🚀 MAIN ENTRY POINT
├── mini_kernel/
│   ├── llm_brain.py       # LLM integration (Claude API)
│   ├── profiler.py        # rocprof-compute integration
│   └── ...
├── examples/
│   └── add_kernel/        # Example Triton kernel
└── README.md
```

## 📄 License

MIT License
