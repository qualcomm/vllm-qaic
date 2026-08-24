# lora — Unit Tests

**LoRA adapter support — cache search, ID consistency (pure Python)**

## Tests

| File | Tests |
|------|-------|
| `test_lora_unit.py` | 17 |

**Total: 17 tests, pure Python (no hardware)**

> On-device LoRA integration tests (loading a LoRA adapter, generating output,
> verifying base vs. LoRA output differ, determinism) are covered by the
> pre-existing `vllm-qaic/tests/device_unit_and_e2e/lora/test_qaic_lora.py` and
> `test_qaic_generate_multiple_loras.py` — this folder's own `test_lora_integration.py`
> (which depended on a nonexistent `llm` fixture) has been removed as redundant.

## `test_lora_unit.py` — Pure Python (17 tests)

### `TestSearchAdaptersInCache` (5 tests)
Tests `search_adapters_in_cache(base_model_name)`:
- `test_empty_cache_returns_empty`
- `test_matching_adapter_is_found`
- `test_two_matching_adapters_found`
- `test_adapter_for_different_base_not_returned`
- `test_hf_home_respected`

> **Note:** Uses `peft_type` in the mock adapter config. The PEFT API requires `peft_type` to be present in `adapter_config.json`.

### `TestVerifyAdapternameToIdConsistency` (7 tests)
Tests `verify_adaptername_to_id_consistency(mapping, modules)`:
- `test_matching_mapping_returns_true`
- `test_extra_key_in_mapping_ignored`
- `test_missing_key_returns_false`
- `test_empty_mapping_empty_modules`
- `test_empty_mapping_non_empty_modules`
- `test_order_does_not_matter`
- `test_single_adapter`

> **Note:** The mapping uses **1-indexed** values: `{"adapter_0": 1, "adapter_1": 2}` (not 0-indexed). This matches the `verify_adaptername_to_id_consistency` implementation which uses `i+1`.

### `TestAdapternameToIdJson` (5 tests)
Tests `adaptername_to_id_json(path, mapping, modules)`:
- `test_write_and_read_round_trip`
- `test_nonexistent_path_raises`
- `test_inconsistent_mapping_raises`
- `test_consistent_mapping_does_not_raise`
- `test_not_written_twice`

## Running

```bash
# Pure Python tests (no hardware needed)
pytest lora/test_lora_unit.py -v

# On-device LoRA tests live under device_unit_and_e2e/lora/
pytest device_unit_and_e2e/lora/ -v
```
