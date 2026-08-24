# on_device_sampling — Unit Tests

**On-device sampling (ODS) — config parsing, ODS+SpD constraint, on-device [xfail]**

> **Current status:** ODS (`aic_include_sampler=True`) moves token sampling from host CPU to QAIC device. The on-device tests are marked `xfail(strict=False)` — they pass as XPASS when ODS is supported on the hardware.

## Tests

| File | Tests | Unit | Integration | Hardware |
|------|-------|------|-------------|----------|
| `test_on_device_sampling.py` | 11 | 9 | 2 | ✅ 2 |

**Total: 11 tests (9 unit, 2 integration)**

## `test_on_device_sampling.py`

### `TestAicIncludeSamplerParsing` (6 tests, pure Python)
Tests `_clean_config({"aic_include_sampler": ...})`:
- `"true"` → `True`
- `"false"` → `False`
- `"1"` → `True`
- `"0"` → `False`
- `True` → `True`
- `False` → `False`

### `TestODSConstraints` (3 tests, pure Python)
- `test_ods_with_spd_raises` — ODS + SpD must raise AssertionError
- `test_ods_without_spd_allowed` — ODS without SpD must not raise
- `test_ods_disabled_with_spd_allowed` — ODS=False with SpD must not raise

### `TestOnDeviceSamplingOnDevice` 🔧 ⚠️ (2 tests, xfail `strict=False`)
- `test_ods_greedy_output_non_empty`
- `test_ods_greedy_deterministic`

These tests are marked `xfail(strict=False)` — they pass as XPASS when ODS is supported on the hardware. In the test run logs, they appear as `XPASS` (unexpected pass) which is allowed.

## Running

```bash
# Pure Python tests
pytest on_device_sampling/test_on_device_sampling.py -v -k "not OnDevice"

# All tests (requires QAIC hardware)
pytest on_device_sampling/ -v \
    --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --seq-len 128 --ctx-len 256 --decode-bsz 4 \
    --dtype mxfp6 --kv-dtype mxint8 --device-group 1 --device-id 0
