# Profiling

The QAIC model runner integrates with `torch_qaic.profile` to capture device-level latency traces during inference.

## Overview

When profiling is enabled, each model forward pass is wrapped with `torch_qaic.profile.ProfileForwardWithSampling`, which captures LRT (Low-Level Runtime) traces showing on-device execution timing.

## Enabling Profiling

Set the `VLLM_TORCH_PROFILER_DIR` environment variable before starting the server:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_TORCH_PROFILER_DIR` | None (disabled) | Output directory for profiling data. Set to any path to enable. |

```bash
export VLLM_TORCH_PROFILER_DIR=/tmp/qaic_profiles

vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-num-seqs 4 \
  --max-model-len 2048 \
  --quantization mxfp6 \
  --kv-cache-dtype mxint8
```

## Output Structure

Profiling produces per-iteration trace directories:

```text
$VLLM_TORCH_PROFILER_DIR/
  lrt_trace_iter0_pid0/
  lrt_trace_iter1_pid0/
  ...
```

Each directory contains device-level execution traces from the Cloud AI runtime.

## Implementation

The instrumentation lives in `vllm/v1/worker/qaic_model_runner.py`:

```python
with optional_qaic_profiling(
    profiling_dir=envs.VLLM_TORCH_PROFILER_DIR,
    profiling_wrapper=qaic_profile.ProfileForwardWithSampling,
    model=self.model,
    n_samples=10,
    profiling_iter=self.profiling_iter,
):
    # Forward pass executes here
```

- **Zero overhead when disabled** (`VLLM_TORCH_PROFILER_DIR` unset)
- Iteration counter increments after each profiled step
- Traces written immediately under `profiling_dir`

## Analysis

Use the Cloud AI SDK tools to analyze LRT traces:

```bash
# View trace summary
ls $VLLM_TORCH_PROFILER_DIR/lrt_trace_iter0_pid0/
```

## Interpreting Results

| Observation | Likely Bottleneck |
|-------------|-------------------|
| High device execution time per iteration | Memory-bandwidth bound on hardware |
| Device time similar between SpD and no-spec | Hardware saturated — SpD cannot improve further |
| Device time low but throughput low | CPU-side scheduling or sampling overhead |

!!! info "Standard vLLM profiling"
    The standard vLLM `VLLM_TORCH_PROFILER_DIR` mechanism is reused by the QAIC backend.
    The `torch_qaic.profile` wrapper adds QAIC-specific device tracing on top of the
    standard PyTorch profiler infrastructure.
