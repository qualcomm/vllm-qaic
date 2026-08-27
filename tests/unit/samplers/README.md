# samplers — Unit Tests

**Sampling parameters — rejection sampler, config parsing, ODS constraints (pure Python)**

> **Note:** `test_samplers_integration.py` is empty — tests were consolidated into `test_samplers.py`.

## Tests

| File | Tests |
|------|-------|
| `test_rejection_sampler.py` | 12 |
| `test_samplers.py` | 8 |
| `test_samplers_integration.py` | 0 (empty) |

**Total: 20 tests, pure Python (no hardware)**

## `test_rejection_sampler.py` — Pure Python (12 tests)

Tests the rejection sampler math and patch import.

- `TestPatchImport` (1 test): rejection sampler module is importable
- `TestGreedyPathMath` (4 tests): greedy path token selection
- `TestLogitsIndices` (4 tests): logits index handling
- `TestSkipCloneOptimization` (3 tests): skip-clone optimization

## `test_samplers.py` — Pure Python (8 tests)

### `TestAicIncludeSamplerParsing` (6 tests)
Tests `_clean_config({"aic_include_sampler": ...})`:
- `"true"` → `True`, `"false"` → `False`
- `"1"` → `True`, `"0"` → `False`
- `True` → `True`, `False` → `False`

### `TestODSSpDMutualExclusion` (2 tests)
- `test_ods_with_spd_raises` — ODS + SpD must raise AssertionError
- `test_ods_without_spd_allowed` — ODS without SpD must not raise

> On-device sampler smoke tests (presence/frequency/repetition penalty, top-p, top-k)
> are covered by the pre-existing
> `vllm-qaic/tests/device_unit_and_e2e/test_qaic_samplers.py` (`TestQaicSamplers`).
> This file's former `TestSamplersOnDevice` class (which depended on a nonexistent
> `llm` fixture) has been removed as redundant.

## `test_samplers_integration.py` — Empty

All tests previously here have moved to their canonical locations:
- `test_qaic_samplers.py::TestQaicSamplers` (on-device smoke tests)
- `device_unit_and_e2e/test_unit_qaic_sampling_params.py::TestSamplingConstraintsOnDevice` (constraint tests)
- `device_unit_and_e2e/test_qaic_output_consistency.py` (greedy determinism)

## Running

```bash
# Pure Python tests (no hardware needed for anything in this folder)
pytest samplers/ -v
```
