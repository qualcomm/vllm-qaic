# benchmark — Unit Tests

**Benchmark configuration for QAIC throughput/latency runs**

Tests the `_clean_config()` normalisation for benchmark-specific keys: `num_cores`, `mos`, `qpc_path`, device group, and ignore-list keys (`batch_size`, `full_batch_size`).

> **Note:** mxfp6/mxint8 normalisation and ctx_len ignore are tested in `accuracy/test_accuracy_config.py` — not duplicated here.

## Tests

| File | Tests | Unit | Integration |
|------|-------|------|-------------|
| `test_benchmark_utils.py` | 12 | 12 | 0 |

**Total: 12 tests (12 unit, 0 integration)**

### `TestCleanConfigBenchmark` (4 tests)
- `test_num_cores_string_to_int` — benchmark uses 14 cores (not 16 like accuracy)
- `test_mos_string_to_int` — benchmark uses mos=4
- `test_aic_num_cores_alias` — `aic_num_cores` maps to `num_cores`
- `test_typical_benchmark_config` — num_cores=14, mos=4, mxfp6=true, mxint8=true

### `TestQpcPathPreservation` (3 tests)
- `test_qpc_path_preserved_verbatim` — path must not be lowercased
- `test_qpc_path_with_uppercase_preserved`
- `test_mdp_load_partition_config_preserved`

### `TestDeviceGroupBenchmark` (3 tests)
- `test_device_group_list` — `[0,1,2,3]` → device_group=[0,1,2,3], num_devices=4
- `test_device_group_string_bracket` — `"[0,1,2,3]"` → same
- `test_single_device` — `device_id=0` → device_group=[0], num_devices=1

### `TestIgnoreListBenchmark` (2 tests)
- `test_batch_size_ignored` — batch_size must not reach the compiler
- `test_full_batch_size_ignored` — full_batch_size must not reach the compiler

## Running

```bash
pytest benchmark/test_benchmark_utils.py -v
