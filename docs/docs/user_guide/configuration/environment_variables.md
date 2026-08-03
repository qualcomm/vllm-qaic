# Environment Variables

QAIC-specific environment variables that control plugin behavior.

## QAIC Plugin Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VLLM_QAIC_COMPILER_ARGS` | str | None | Extra arguments passed to the QAIC compiler during QPC compilation |
| `VLLM_QAIC_DFS_EN` | bool | True | Enable/disable DFS (Dynamic Frequency Scaling) on the device |
| `VLLM_QAIC_MAX_CPU_THREADS` | int | None | Maximum CPU threads for host-side processing |
| `VLLM_QAIC_MOS` | int | None | MOS (Memory Operating State) setting for the device |
| `VLLM_QAIC_NUM_CORES` | int | None | Number of NSP cores to allocate per device (overrides `additional_config`) |
| `VLLM_QAIC_QPC_PATH` | str | None | Path to a pre-compiled QPC directory (skips compilation) |
| `VLLM_DISABLE_LD_PRELOAD_OPT` | bool | `0` | Set to `1` to skip the automatic `KMP_*` OpenMP tuning applied when `libiomp5.so` is on `LD_PRELOAD` (AOT mode, CPU-bound SpD proposers) |
| `VLLM_TORCH_QAIC_BASE_PATH` | str | `/opt/qti-aic/integrations/torch_qaic` | Base path for torch_qaic wheels (PYT mode) |

## Standard Variables (QAIC-Relevant)

| Variable | Description |
|----------|-------------|
| `QAIC_VISIBLE_DEVICES` | Comma-separated QID list (e.g., `0,1,2,3`). Controls which devices are accessible. |
| `QEFF_HOME` | QEfficient cache directory (default: `~/.cache/qefficient/`) |
| `QAIC_DEVICE_LOG_LEVEL` | Device-level log verbosity |
| `QAIC_DEBUG` | Enable QAIC debug logging (`0` or `1`) |
| `VLLM_TORCH_PROFILER_DIR` | Output directory for profiling data. Enables `torch_qaic.profile.ProfileForwardWithSampling` device-level latency tracing in the QAIC model runner. |
| `LD_PRELOAD` | If set to include `libiomp5.so`, triggers automatic OpenMP tuning for CPU-bound SpD proposers in AOT mode — see [Speculative Decoding](../features/speculative_decoding.md) |

## Usage Examples

```bash
# Use pre-compiled QPC (skip compilation)
export VLLM_QAIC_QPC_PATH=/path/to/compiled/qpc_dir

# Enable device-level profiling
export VLLM_TORCH_PROFILER_DIR=/tmp/my_profiles

# Limit to specific devices
export QAIC_VISIBLE_DEVICES=0,1,2,3

# Disable DFS for consistent benchmarking
export VLLM_QAIC_DFS_EN=0
```
