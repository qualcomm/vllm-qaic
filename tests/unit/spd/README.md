# spd — Unit Tests

**Speculative decoding (SpD/PLD/DLM) — metadata, budget, config (pure Python)**

> **API note:** Uses `speculative_config={"method": "ngram", "num_speculative_tokens": 3, ...}` dict format. The deprecated `speculative_model="[ngram]"` parameter is NOT used.

## Tests

| File | Tests |
|------|-------|
| `test_calc_spec_decode_metadata.py` | 4 |
| `test_max_num_batched_tokens.py` | 17 |
| `test_spec_decode_unit.py` | 13 |
| `test_variable_decode_specializations.py` | 32 |

**Total: 66 tests, pure Python (no hardware)**

## Pure Python Tests

### `test_calc_spec_decode_metadata.py` (4 tests)
Tests `calc_spec_decode_metadata()` for various batch configurations.

### `test_max_num_batched_tokens.py` (17 tests)
Tests `max_num_batched_tokens` calculation for SpD configurations.

### `test_variable_decode_specializations.py` (32 tests)
Tests variable decode specialization logic for different batch sizes and token counts.

### `test_spec_decode_unit.py` (13 tests)

**`TestDraftModelConfigExtraction`** (5 tests): draft_override takes priority, fallback, empty cases

**`TestSpDConfigValidation`** (2 tests): LoRA+SpD raises, ODS config parsing

## On-device Tests

> On-device SpD tests (`test_ngram_spd_output_non_empty`, `test_ngram_spd_matches_baseline`
> — SpD output vs. non-SpD baseline) live in
> `vllm-qaic/tests/device_unit_and_e2e/test_unit_qaic_spd.py`. This folder's own
> `test_spd_integration.py`, which depended on a nonexistent `llm` fixture, has been
> removed as redundant.

## Running

```bash
# Pure Python tests
pytest spd/ -v

# On-device tests (requires QAIC hardware)
pytest device_unit_and_e2e/test_unit_qaic_spd.py -v \
    --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --seq-len 128 --ctx-len 256 --decode-bsz 4 \
    --dtype mxfp6 --kv-dtype mxint8 --device-group 1 --device-id 0
```
