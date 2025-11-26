# OpenEvolve Standalone Tutorial

This tutorial demonstrates how to use **GEAK-OpenEvolve** as a standalone GPU kernel optimization framework, without AIG-Eval integration.

## What is OpenEvolve?

OpenEvolve is an LLM-guided evolutionary framework for optimizing GPU kernels. It uses Large Language Models to generate kernel variants, evaluates their performance, and iteratively evolves toward better implementations.

## Prerequisites

### 1. Conda Environment
```bash
conda activate the_rock_0924
```

### 2. Environment Variables
```bash
export OPENAI_API_KEY="your-api-key"
export ROCM_GOLDEN_DATA_PATH="/path/to/golden_results"
```

The notebook will automatically set `ROCM_GOLDEN_DATA_PATH` to use the cloned GEAK-eval repository.

### 3. Install OpenEvolve
```bash
cd geak-openevolve
pip install -e .
```

## Quick Start

```bash
cd geak-openevolve/tutorial/
jupyter lab RUN_OPENEVOLVE_STANDALONE.ipynb
```

## Notebook Structure

| Step | Description |
|------|-------------|
| 1 | Environment setup |
| 2 | Set environment variables |
| 3 | Verify installation |
| 4 | Select example kernel |
| 5 | Setup evaluator (rocm_evaluator.py) |
| 6 | Configure evolution parameters |
| 7 | Setup output directory |
| 8 | Run evolution pipeline |
| 9 | View results |
| 10 | Explore output files |

## Configuration

You can customize evolution by modifying variables in **Step 6**:

```python
MAX_ITERATIONS = 10     # More iterations = more optimization time
POPULATION_SIZE = 30    # More variants explored per iteration
NUM_ISLANDS = 4         # Parallel evolution for diversity
```

## Understanding the Pipeline

### 1. Initial Program
Start with a baseline GPU kernel (Triton/HIP/CUDA).

### 2. Evolution Loop
For each iteration:
- **Generate**: LLM creates kernel variants
- **Evaluate**: Test compilation, correctness, performance
- **Select**: Keep best performing variants
- **Evolve**: Generate next generation

### 3. Results
Best program is saved with:
- Optimized kernel code
- Performance metrics (speedup, latency)
- Evolution metadata

## Output Directory Structure

```
runs/tutorial_run_TIMESTAMP/
├── best/
│   ├── best_program.py           # Optimized kernel
│   └── best_program_info.json    # Metrics and metadata
├── checkpoints/
│   └── checkpoint_N/             # Evolution snapshots
├── logs/
│   └── openevolve_TIMESTAMP.log  # Detailed logs
└── database/
    └── programs.db               # All evaluated programs
```

## Example Results

```json
{
  "id": "abc123...",
  "generation": 5,
  "iteration": 10,
  "metrics": {
    "success": 1.0,
    "correctness_score": 1.0,
    "speedup": 1.35,
    "combined_score": 1.35
  }
}
```

**Interpretation:**
- `success = 1.0`: Compiled and ran successfully
- `correctness_score = 1.0`: Produces correct results
- `speedup = 1.35`: **35% faster than baseline!**
- `combined_score = 1.35`: Overall quality score

## Command-Line Usage

You can also run OpenEvolve from the command line:

```bash
openevolve-run \
  GEAK-eval-OE/geak_eval/data/ROCm/data/ROCm_v1/test_add_kernel.py \
  examples/tb/rocm_evaluator.py \
  --config configs/default_config.yaml \
  --output runs/my_run
```

**Note**: The tutorial uses `test_add_kernel.py` from GEAK-eval-OE (the cloned GEAK-eval repository), which is a validated ROCm-optimized Triton kernel with built-in performance tests.

## Evaluator Details

The `rocm_evaluator.py` evaluator:
- **Compiles** kernel using Triton compiler
- **Tests correctness** against reference implementation
- **Benchmarks performance** across multiple configurations
- **Calculates speedup** vs golden baseline

## Customization

### Use Your Own Kernel

1. Create a kernel file: `tutorial/my_kernel.py`
2. Update **Step 4** to use your kernel:
   ```python
   INITIAL_KERNEL = TUTORIAL_DIR / "my_kernel.py"
   ```

### Custom Configuration

1. Copy a config: `cp examples/tb/configs/claude.yaml tutorial/my_config.yaml`
2. Edit parameters in `my_config.yaml`
3. Update **Step 6** to use your config

### Different LLM Models

Edit the config file:
```yaml
llm:
  models:
    - name: claude-sonnet-4
      api_key: your-key
      api_base: your-endpoint
```

## Troubleshooting

### Issue: openevolve-run command not found
**Solution:** Ensure OpenEvolve is installed:
```bash
cd geak-openevolve && pip install -e .
```

### Issue: Import errors
**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: ROCM_GOLDEN_DATA_PATH not found
**Solution:** The notebook sets this automatically from GEAK-eval-OE. Ensure it's present:
```bash
ls geak-openevolve/GEAK-eval-OE/geak_eval/data/ROCm/data/performance/golden_results
```

### Issue: No GPU available
**Solution:** Ensure ROCm is installed and GPU is accessible:
```bash
rocm-smi
```

## Advanced Topics

### Island-Based Evolution

OpenEvolve uses multiple "islands" that evolve independently:
- **Diversity**: Different islands explore different optimization strategies
- **Migration**: Good programs occasionally move between islands
- **Parallelism**: Islands can evolve concurrently

### MAP-Elites Architecture

OpenEvolve maintains a grid of programs across feature dimensions:
- **Behavioral Dimensions**: e.g., memory usage, compute intensity
- **Elites**: Best program for each feature combination
- **Exploration**: Encourages diverse optimizations

## Performance Tips

1. **Start Small**: Use 2-3 iterations to verify setup
2. **Scale Up**: Increase to 10+ iterations for real optimization
3. **Monitor Progress**: Check logs in real-time
4. **Multiple Runs**: Different random seeds may find different optimizations

## Resources

- **Main Documentation**: `../README.md`
- **Examples**: `../examples/tb/`
- **ROCm Evaluator**: `../examples/tb/rocm_evaluator.py`
- **Configuration**: `../configs/`

## Support

For issues or questions:
- Check OpenEvolve documentation
- Review example configurations
- Examine evaluator logs

---

**Status:** ✅ Standalone tutorial ready
**Location:** `geak-openevolve/tutorial/RUN_OPENEVOLVE_STANDALONE.ipynb`
**Date:** November 26, 2025

