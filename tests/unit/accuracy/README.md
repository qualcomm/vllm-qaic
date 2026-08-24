# accuracy — Unit Tests

Tests for QAIC accuracy evaluation configuration.

## What is tested

The accuracy CI script (`run_accuracy_test_qaic.sh`) uses **lm-evaluation-harness** to measure model accuracy on standard benchmarks (ARC, HellaSwag, MMLU, etc.).

These unit tests cover the **QAIC-specific configuration logic** that affects accuracy results — specifically the quantization config and `override_qaic_config` normalisation applied before the QPC is compiled. They do **not** test E2E accuracy (that requires hardware and model downloads).

## Test files

| File | Tests | Description |
|------|-------|-------------|
| `test_accuracy_config.py` | 13 | `QAIC_QUANTIZATION_LIST` membership (mxfp6 present, no duplicates); `_clean_config()` for mxfp6/mxint8/num_cores/mos normalisation; `QaicQuantConfig.get_name()`, `get_supported_act_dtypes()`, `get_config_filenames()`; `QaicPlatform.get_supported_quantization()` |

## Running

```bash
# All tests (no hardware required)
pytest accuracy/ -v
```

## Existing upstream vLLM tests (informational)

The following upstream vLLM tests are relevant to accuracy evaluation but are **not QAIC-specific** and are not copied here:

| Upstream file | What it tests |
|---------------|---------------|
| `tests/lm_eval_harness/test_lm_eval_correctness.py` | lm-eval-harness integration, model accuracy on ARC/HellaSwag |
| `tests/accuracy/` | Generic accuracy evaluation helpers |

These can be run directly from the upstream vLLM test suite without QAIC hardware.
