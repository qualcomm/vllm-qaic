# generic — Unit Tests

**Cross-feature pure-Python tests — platform config, chunked prefill, error handling, env/patch/quant utils**

> **Note:** Greedy determinism and batch=single tests are in `consistency/` — not duplicated here.
> Smoke tests for individual sampling parameters (presence_penalty, top_p, etc.) are in `samplers/test_samplers.py`.

## Tests

| File | Tests |
|------|-------|
| `test_error_handling.py` | 8 |
| `test_chunked_prefill.py` | 21 |
| `test_platform.py` | 36 |
| `test_envs.py` | 27 |
| `test_patch_config.py` | 14 |
| `test_patch_utils.py` | 7 |
| `test_qaic_utils.py` | 79 |
| `test_quant_config.py` | 31 |

**Total: 223 tests, all pure Python (no hardware)**

## `test_error_handling.py`

### `TestConfigValidationErrors` (8 tests, pure Python)
- `test_lora_with_spd_raises` — LoRA + SpD must raise
- `test_lora_with_multimodal_raises`
- `test_spd_with_multimodal_raises`
- `test_ods_with_spd_raises` — ODS + SpD must raise
- `test_invalid_config_string_raises`
- `test_non_dict_config_raises`
- `test_prefix_caching_with_kv_consumer_raises` — verifies prefix caching disabled (not AssertionError)
- `test_lora_with_disagg_raises`

> On-device runtime error handling tests (`TestRuntimeErrorHandling`) live in
> `vllm-qaic/tests/device_unit_and_e2e/test_unit_qaic_error_handling.py`, using
> the `device/` fixture system.

## `test_chunked_prefill.py`

Pure-Python coverage of chunked-prefill config derivation (AOT/eager branches, seq_len
bucketing, `max_num_batched_tokens` formulas).

> On-device chunked-prefill correctness tests (long prompts, mixed batches, chunked vs.
> non-chunked output parity) live in
> `vllm-qaic/tests/device_unit_and_e2e/test_unit_qaic_chunked_prefill.py`.

## `test_platform.py`

Pure-Python coverage of `QaicPlatform.check_and_update_config()` — AOT and eager branches
(device config, cache config, scheduler config, SpD/ODS/kv-transfer/async-scheduling
constraints).

## Running

```bash
# Pure Python tests only (no hardware needed for anything in this folder)
pytest generic/ -v
```
