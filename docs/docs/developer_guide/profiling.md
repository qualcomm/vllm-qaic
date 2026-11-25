# Profiling

The QAIC backend supports two independent profilers:

| Profiler | What it captures | How to enable |
|----------|------------------|---------------|
| [Device-level (LRT)](#device-level-profiling-lrt) | On-device execution timing from the Cloud AI runtime | `VLLM_TORCH_QAIC_PROFILER_DIR` environment variable |
| [Host-level (torch profiler)](#host-level-profiling-torch-profiler) | Host-side (CPU) operator timing | `--profiler-config` server argument |

!!! note "PyT mode only"
    Both profilers are available only in PyT mode. In AoT mode,
    `QaicModelRunnerAoT` does not wrap the forward pass with the LRT hook, and
    `QaicWorkerAoT.profile()` only logs `profile is not supported with QAIC AoT mode`,
    so `/start_profile` returns success without producing a trace.

## Device-level profiling (LRT)

The QAIC model runner integrates with `torch_qaic.profile` to capture device-level latency traces during inference. When enabled, each model forward pass is wrapped with `torch_qaic.profile.ProfileForwardWithSampling`, which captures LRT (Low-Level Runtime) traces showing on-device execution timing.

### Enabling device-level profiling

Set the `VLLM_TORCH_QAIC_PROFILER_DIR` environment variable before starting the server:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_TORCH_QAIC_PROFILER_DIR` | None (disabled) | Output directory for LRT device traces. Set to any path to enable. |

```bash
export VLLM_TORCH_QAIC_PROFILER_DIR=/tmp/qaic_profiles

vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-num-seqs 4 \
  --max-model-len 2048
```

Device-level profiling is not request-scoped: there is no start/stop endpoint, and every forward pass from server start writes a trace. Keep the profiled workload small.

### Output structure

Profiling produces one trace directory per forward pass, named by iteration and global rank:

```text
$VLLM_TORCH_QAIC_PROFILER_DIR/
  lrt_trace_iter0_pid0/
  lrt_trace_iter1_pid0/
  ...
```

Each directory contains device-level execution traces from the Cloud AI runtime.

### Implementation

The instrumentation lives in `QaicModelRunnerPyt.execute_model` in `vllm_qaic/worker/model_runner.py`:

```python
with optional_qaic_profiling(
    profiling_dir=envs.VLLM_TORCH_QAIC_PROFILER_DIR,
    profiling_wrapper=qaic_profile.ProfileForwardWithSampling,
    model=self.model,
    n_samples=1,
    skip_first_n=0,
    device_name=str(self.device),
    profiling_iter=self.profiling_iter,
) as self.model:
    # Forward pass executes here
```

- **Zero overhead when disabled** (`VLLM_TORCH_QAIC_PROFILER_DIR` unset)
- Iteration counter increments after each step
- Traces written immediately under `$VLLM_TORCH_QAIC_PROFILER_DIR/lrt_trace_iter<N>_pid<rank>/`

### Analysis

Use the Cloud AI SDK tools to analyze LRT traces:

```bash
# View trace summary
ls $VLLM_TORCH_QAIC_PROFILER_DIR/lrt_trace_iter0_pid0/
```

### Interpreting results

| Observation | Likely Bottleneck |
|-------------|-------------------|
| High device execution time per iteration | Memory-bandwidth bound on hardware |
| Device time similar between SpD and no-spec | Hardware saturated — SpD cannot improve further |
| Device time low but throughput low | CPU-side scheduling or sampling overhead |

## Host-level profiling (torch profiler)

Host-side profiling uses upstream vLLM's `--profiler-config` server argument. The legacy `VLLM_TORCH_PROFILER_DIR` environment variable was removed upstream and has no effect.

### Enabling host-level profiling

Pass `--profiler-config` when launching the server, either as JSON or as dotted flags:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-num-seqs 4 \
  --max-model-len 2048 \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/tmp/vllm_profile"}'

# Equivalent dotted form
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-num-seqs 4 \
  --max-model-len 2048 \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_dir=/tmp/vllm_profile
```

`torch_profiler_dir` is required when `profiler` is `torch`. Only `profiler=torch` is supported on QAIC.

Profiling is request-scoped: the server records between `POST /start_profile` and `POST /stop_profile`. `vllm bench serve --profile` calls both endpoints around the benchmark:

```bash
vllm bench serve \
  --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --num-prompts 16 \
  --profile
```

If the server was started without `--profiler-config`, `/start_profile` fails with `Profiling is not enabled. Please set --profiler-config to enable profiling.`

### Output

`torch_profiler_dir` receives a Chrome/TensorBoard trace per worker (`*.pt.trace.json.gz`) and a `profiler_out_<rank>.txt` operator summary table. View traces in [Perfetto](https://ui.perfetto.dev) or TensorBoard.

!!! info "Host-side time only"
    `QaicWorkerPyt.profile()` creates the torch profiler with `activities=["CPU"]`,
    because QAIC exposes no torch-level device activity. Device execution appears
    as opaque wait time; use [device-level profiling](#device-level-profiling-lrt)
    to see on-device timing.

For the full set of `--profiler-config` options (iteration windows, stack/shape/memory recording, gzip), see the upstream vLLM [profiling guide](https://docs.vllm.ai/en/latest/contributing/profiling.html).
