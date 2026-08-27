# consistency — Unit Tests

**Output consistency — `_clean_config()` normalisation for keys that affect determinism (pure Python)**

> **Canonical location for greedy determinism / batch-consistency tests on device:**
> `vllm-qaic/tests/device_unit_and_e2e/test_qaic_output_consistency.py`
> (`TestQaicOutputConsistency`) — not duplicated here.

## Tests

| File | Tests |
|------|-------|
| `test_output_consistency.py` | 7 |

**Total: 7 tests, pure Python (no hardware)**

## `test_output_consistency.py`

### `TestCleanConfigConsistency` (7 tests)
Tests `_clean_config()` for consistency-relevant keys:
- mxfp6/mxint8/dfs normalisation
- num_cores/mos int conversion
- None config, empty config

> On-device greedy determinism / batch-consistency tests (same prompt → same
> token_ids across runs, batch output matches single-prompt output, async
> scheduling matches sync) live in the pre-existing
> `vllm-qaic/tests/device_unit_and_e2e/test_qaic_output_consistency.py`
> (`TestQaicOutputConsistency`) — this folder's own `test_consistency_integration.py`
> (which depended on a nonexistent `llm` fixture) has been removed as redundant.

## Running

```bash
# Pure Python tests
pytest consistency/test_output_consistency.py -v

# On-device tests (requires QAIC hardware)
pytest device_unit_and_e2e/test_qaic_output_consistency.py -v \
    --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --seq-len 128 --ctx-len 256 --decode-bsz 4 \
    --dtype mxfp6 --kv-dtype mxint8 --device-group 1 --device-id 0
```
