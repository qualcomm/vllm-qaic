# Testing

The vLLM QAIC plugin has comprehensive CI test infrastructure covering accuracy, performance, and feature validation.

## Test Structure

Tests are organized in `tests/test_qaic/`:

```text
tests/test_qaic/
├── test_accuracy.py            # Model accuracy validation
├── test_benchmark.py           # Performance benchmarking
├── test_consistency.py         # Output determinism
├── test_embed.py               # Embedding models
├── test_multimodal.py          # Vision-language models
├── test_online_serving.py      # Server API tests
├── test_samplers.py            # Sampling parameter tests
├── spec_decode/                # Speculative decoding tests
├── lora/                       # LoRA adapter tests
├── disaggregated_serving/      # Disaggregated serving tests
├── kv_transfer/                # KV transfer tests
├── gptoss/                     # Reasoning backend tests
└── kernels/                    # Kernel unit tests
```

## CI Scripts

21 CI scripts in `ci_scripts/` automate test execution:

| Script | Coverage |
|--------|----------|
| `ci_fasttest_qaic.sh` | Quick sanity check |
| `ci_fasttest_qaic_eager_mode.sh` | Eager mode sanity |
| `ci_fasttest_qaic_plugin.sh` | Plugin-only tests |
| `run_accuracy_test_qaic.sh` | Model accuracy |
| `run_benchmark_test_qaic.sh` | Performance benchmarks |
| `run_consistency_test_qaic.sh` | Output reproducibility |
| `run_disaggregated_test_qaic.sh` | Disaggregated serving |
| `run_embed_test_qaic.sh` | Embedding models |
| `run_multimodal_test_qaic.sh` | Vision-language models |
| `run_online_test_qaic.sh` | Online serving |
| `run_samplers_test_qaic.sh` | Sampler correctness |
| `run_spd_test_qaic.sh` | Speculative decoding |
| `run_lmcache_disaggregated_test_qaic.sh` | LMCache integration |

## Running Tests Locally

### Prerequisites

```bash
export QAIC_VISIBLE_DEVICES=0  # or appropriate QIDs
```

### Single Test

```bash
python -m pytest tests/test_qaic/test_accuracy.py -v
```

### Full CI Suite

```bash
# Fast test (quick sanity)
bash ci_scripts/ci_fasttest_qaic.sh

# Accuracy test
bash ci_scripts/run_accuracy_test_qaic.sh
```

### With Custom Device Group

Most CI scripts accept device parameters:

```bash
QAIC_VISIBLE_DEVICES=4,5,6,7 bash ci_scripts/run_spd_test_qaic.sh
```

## Test Categories

### Accuracy Tests

Validate model output against reference implementations. Tests compare generated text or logits with expected results within tolerance.

### Benchmark Tests

Measure throughput (tokens/s), TTFT, and TPOT under various configurations. Used to detect performance regressions.

### Consistency Tests

Verify deterministic output across runs with the same seed and parameters.

### Feature Tests

End-to-end validation of specific features (SpD, LoRA, disaggregated, multimodal) ensuring correct behavior.

## Adding New Tests

1. Create test file in `tests/test_qaic/`
2. Follow existing patterns (use `pytest` fixtures, device group parametrization)
3. Add a CI script in `ci_scripts/` if the test needs special setup
4. Ensure tests clean up device resources (`gc.collect()`, explicit `del`)
