# disaggregated — Unit Tests

**Disaggregated prefill/decode serving — constraints, scheduling policies, streaming**

All tests are pure Python (no hardware required).

## Tests

| File | Tests | Unit | Integration |
|------|-------|------|-------------|
| `test_disaggregated_integration.py` | 10 | 10 | 0 |
| `test_policy.py` | 6 | 6 | 0 |
| `test_prefill_streaming_utils.py` | 17 | 17 | 0 |

**Total: 33 tests (33 unit, 0 integration)**

Despite the filename, every test here is pure Python
(`check_and_update_config()` config validation) or requires only the real
`qaic_disagg` package (no hardware) — none are hardware/model-name-gated
integration tests.

## `test_disaggregated_integration.py`

### `TestDisaggregatedConstraints` (8 tests)
- `test_lora_with_disagg_raises` — LoRA + disagg must raise AssertionError
- `test_unsupported_spd_type_raises` — unsupported SpD method must raise
- `test_ngram_spd_allowed` — ngram SpD is allowed with disagg
- `test_draft_model_spd_allowed` — draft_model SpD is allowed with disagg
- `test_prefix_caching_kv_consumer_disabled` — verifies `enable_prefix_caching=False` after call
- `test_prefix_caching_kv_both_disabled` — same
- `test_prefix_caching_kv_producer_allowed` — kv_producer allows prefix caching (requires `stages="1"`)
- `test_no_prefix_caching_kv_consumer_allowed`

> **Note on prefix caching tests:** In AOT mode, the platform disables prefix caching before checking disagg constraints — `check_and_update_config()` never raises for these cases (empirically verified), so the tests call it directly and assert `enable_prefix_caching=False` after the call, rather than tolerating either an exception or the disabled state.

> **Note on `test_prefix_caching_kv_producer_allowed`:** Requires `stages="1"` in `override_qaic_config` to avoid `TypeError: int() argument must be a string, not 'NoneType'` in `platform_base.py:449`.

### `TestDisaggregatedSchedulingPolicies` (2 tests)
- `test_round_robin_cycles` — round robin cycles through servers
- `test_least_outstanding_initial_distribution` — first server is valid

> **Note:** These tests require the real `qaic_disagg` package to be
> installed (top-level `qaic_disagg`, not `vllm_qaic.qaic_disagg` — that
> submodule does not exist in this repo). They skip gracefully via
> `pytest.importorskip` if the package is not available; when it is
> available, both tests exercise the actual
> `qaic_disagg.proxy.server.SchedulingPolicy` API (no-arg constructor,
> `.schedule(cycler, instances)`), not a `get_server()` method.

## `test_policy.py` (6 tests)

Pure Python tests for scheduling policy logic.

## `test_prefill_streaming_utils.py` (17 tests)

Pure Python tests for prefill streaming utilities.

## Running

```bash
pytest disaggregated/ -v
