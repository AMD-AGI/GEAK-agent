# Cursor Agent: {KERNEL_NAME} Kernel Optimization

## Your Mission
Optimize the **{KERNEL_NAME}** kernel for AMD MI325X GPUs. Achieve maximum speedup while maintaining correctness.

## Environment Setup

```bash
docker exec -it minikernel_sdubagun bash

export HIP_VISIBLE_DEVICES={GPU_DEVICE}
export PYTHONPATH="/home/sdubagun/work/repos/GEAK-agent:/home/sdubagun/work/repos/aiter"
export AMD_LLM_API_KEY="471c248fdb454e8b96173c8d25b03593"
```

## Kernel Location
- **Workspace**: `/home/sdubagun/work/repos/GEAK-agent/{WORKSPACE_DIR}`
- **Kernel File**: `/home/sdubagun/work/repos/GEAK-agent/{WORKSPACE_DIR}/kernel.py`
- **Original aiter**: `/home/sdubagun/work/repos/aiter/aiter/ops/triton/{ORIGINAL_PATH}`
- **GPU**: {GPU_DEVICE}

## MCP Tools

### Metrix (Detailed Hardware Profiler)
```python
from mini_kernel.mcp_tools.metrix import MetrixTool
metrix = MetrixTool(gpu_device="{GPU_DEVICE}")
result = metrix.profile(kernel_code, num_replays=3)
# Returns: duration_us, bottleneck, coalescing_efficiency, arithmetic_intensity, suggestions
```

### Profiler (Simple Timing)
```python
from mini_kernel.mcp_tools.profiler import ProfilerTool
profiler = ProfilerTool(gpu_device="{GPU_DEVICE}")
result = profiler.profile(kernel_code)
```

### Benchmark
```python
from mini_kernel.mcp_tools.bench import BenchTool
bench = BenchTool(gpu_device="{GPU_DEVICE}")
result = bench.benchmark(kernel_code, warmup_iters=1000, bench_iters=3000)
```

### Verify
```python
from mini_kernel.mcp_tools.verify import VerifyTool
verify = VerifyTool(gpu_device="{GPU_DEVICE}")
result = verify.verify(baseline_code, optimized_code)
```

## LLM API (Claude Opus 4.5)

```python
from mini_kernel.llm_optimizer import AmdLLMClient, LLMConfig
config = LLMConfig(model_name="claude-opus-4-5", api_key="471c248fdb454e8b96173c8d25b03593")
client = AmdLLMClient(config)
```

## Quick Test

```bash
cd /home/sdubagun/work/repos/GEAK-agent/{WORKSPACE_DIR} && python kernel.py
```

## OpenEvolve Automated Tool

```bash
cd /home/sdubagun/work/repos/GEAK-agent/{WORKSPACE_DIR} && ./run.sh
```

This runs OpenEvolve which uses profiler bottlenecks to guide optimization strategies.
