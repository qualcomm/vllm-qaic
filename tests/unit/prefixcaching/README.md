# prefixcaching — Unit Tests

**Prefix caching — disable logic in AOT mode, disagg constraints, on-device [xfail]**

> **Current status:** Prefix caching is NOT supported in vllm-qaic AOT mode. The platform forcibly disables it. On-device tests are marked `xfail(strict=False)` — they pass as XPASS when prefix caching is supported.

## Tests

| File | Tests | Unit | Integration | Hardware |
|------|-------|------|-------------|----------|
| `test_prefix_caching.py` | 13 | 10 | 3 | ✅ 3 |

**Total: 13 tests (10 unit, 3 integration)**

## `test_prefix_caching.py`

### `TestPrefixCachingDisabledInAOT` (4 tests, pure Python)
- `test_prefix_caching_disabled_when_enabled` — `enable_prefix_caching=True` → `False` in AOT
- `test_mamba_block_size_reset` — reset to `max_model_len`
- `test_mamba_cache_mode_reset` — reset to `"none"`
- `test_prefix_caching_already_disabled_no_change`

### `TestBlockSizeConfig` (2 tests, pure Python)
- `test_aot_block_size_equals_max_model_len` — block_size = max_model_len in AOT
- `test_block_size_formula` — verified for 512/1024/2048/4096

### `TestDisaggregatedPrefixCaching` (4 tests, pure Python)
- `test_prefix_caching_with_kv_consumer_raises` — verifies `enable_prefix_caching=False` after call
- `test_prefix_caching_with_kv_both_raises` — same
- `test_prefix_caching_with_kv_producer_allowed` — kv_producer allows prefix caching
- `test_no_prefix_caching_with_kv_consumer_allowed`

> **Note:** In AOT mode, the platform disables prefix caching before checking disagg constraints. Tests verify the final state (`enable_prefix_caching=False`) rather than expecting an `AssertionError`.

> **Note on `test_prefix_caching_with_kv_producer_allowed`:** Requires `stages="1"` in `override_qaic_config` to avoid `TypeError: int() argument must be a string, not 'NoneType'` in `platform_base.py:449`.

### `TestPrefixCachingOnDevice` 🔧 ⚠️ (3 tests, xfail `strict=False`)
- `test_prefix_cache_hit_on_repeated_prompt`
- `test_prefix_cache_output_matches_no_cache`
- `test_shared_prefix_batch_consistency`

These tests are marked `xfail(strict=False)` — they pass as XPASS when prefix caching is supported on the hardware.

## Running

```bash
# Pure Python tests only
pytest prefixcaching/test_prefix_caching.py -v -k "not OnDevice"

# All tests (requires QAIC hardware)
pytest prefixcaching/ -v \
    --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --seq-len 128 --ctx-len 256 --decode-bsz 4 \
    --dtype mxfp6 --kv-dtype mxint8 --device-group 1 --device-id 0
