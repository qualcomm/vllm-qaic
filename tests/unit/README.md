## Parallel execution

`run_all_tests.sh` runs each `test_*.py` file sequentially by default. Pass
`--parallel` to instead dispatch every file as an independent job across a
worker pool (default 8, override with `--workers N`):

```bash
bash run_all_tests.sh --parallel
bash run_all_tests.sh --parallel --workers 4
bash run_all_tests.sh --parallel --feature generic   # combines with any other flag
```

This mirrors the job-collection + concurrent-dispatch pattern
`ci_scripts/collect_jobs.py` / `ci_scripts/scheduler.py` use to parallelize
`device_unit_and_e2e/` — `unit/collect_jobs.py` groups jobs (one per test
file, resolving the same embed/lora/device flag routing `run_all_tests.sh`
already does) and `unit/scheduler.py` runs them concurrently, printing each
job's full captured output as one atomic block as it finishes so parallel
output still reads cleanly.

Unlike the `device_unit_and_e2e/` version, there is **no device-pool,
export/compile cold-start gating, or device acquire/release logic** here —
`unit/` tests are pure-Python and never contend for a real QAIC device pool.
(`custom_ops/test_grouped_topk_qaic.py` used to be the one exception — all 4
of its tests required real hardware via `torch.device("qaic:0")` — but it was
a byte-for-byte duplicate of `device_unit_and_e2e/test_unit_qaic_grouped_topk.py`,
which already covers that ground, so it has been moved to `tests/_deprecated/`.)
So `unit/scheduler.py` is deliberately simpler: a plain
`ThreadPoolExecutor` bounded by `--workers`, nothing more.

---

**`TestODSConstraints`** (3 tests): ODS+SpD raises, ODS without SpD allowed, ODS disabled with SpD allowed

> On-device correctness tests (`TestOnDeviceSamplingOnDevice`, marked
> `xfail(strict=False)` — ODS is not yet supported in vllm-qaic) now live in
> `vllm-qaic/tests/device_unit_and_e2e/test_unit_qaic_on_device_sampling.py`,
> which always runs (no CLI opt-in skip guard) since the `device/` suite is
> AOT-only by convention; the `xfail` marker is preserved there.

---

### `prefixcaching/`

**Prefix caching — disable logic in AOT mode, disagg constraints (pure Python)**

- **10 tests** (10 run, 0 skip)

| File | Tests | Runs | Skips |
|------|-------|------|-------|
| `test_prefix_caching.py` | 10 | 10 | 0 |

**`TestPrefixCachingDisabledInAOT`** (4 tests): disabled when enabled, mamba_block_size reset, mamba_cache_mode reset, already disabled no change

**`TestBlockSizeConfig`** (2 tests): block_size = max_model_len in AOT, formula for various sizes

**`TestDisaggregatedPrefixCaching`** (4 tests):
- `test_prefix_caching_with_kv_consumer_raises` — verifies `enable_prefix_caching=False` after call
- `test_prefix_caching_with_kv_both_raises` — same
- `test_prefix_caching_with_kv_producer_allowed` — requires `stages="1"` in `override_qaic_config` and the `patch_qaic_executor_import_bug` fixture (see `disaggregated/` note above — same underlying source bug)
- `test_no_prefix_caching_with_kv_consumer_allowed`

> **Note:** In AOT mode, the platform disables prefix caching before checking disagg constraints. Tests verify the final state (`enable_prefix_caching=False`) rather than expecting an `AssertionError`.

> On-device correctness tests (`TestPrefixCachingOnDevice`, marked
> `xfail(strict=False)` — prefix caching is not yet supported in vllm-qaic
> AOT mode) now live in
> `vllm-qaic/tests/device_unit_and_e2e/test_unit_qaic_prefix_caching.py`,
> which always runs (no CLI opt-in skip guard); the `xfail` marker is
> preserved there.

---

### `samplers/`

**Sampling parameters — rejection sampler, all penalty types (pure Python)**

- **20 tests** (20 run, 0 skip)
- *`test_samplers_integration.py` is empty — tests consolidated into `test_samplers.py`.*

| File | Tests | Runs | Skips |
|------|-------|------|-------|
| `test_rejection_sampler.py` | 12 | 12 | 0 |
| `test_samplers.py` | 8 | 8 | 0 |
| `test_samplers_integration.py` | 0 | 0 | 0 |

**`TestAicIncludeSamplerParsing`** (6 tests): "true"/"false"/"1"/"0"/True/False → bool

**`TestODSSpDMutualExclusion`** (2 tests): ODS+SpD raises, ODS without SpD allowed

**`TestPatchImport`** / **`TestGreedyPathMath`** / **`TestLogitsIndices`** / **`TestSkipCloneOptimization`** (12 tests): pure Python rejection sampler logic

> On-device smoke tests (`TestSamplersOnDevice`, formerly in
> `test_samplers.py`) are covered by the pre-existing
> `vllm-qaic/tests/device_unit_and_e2e/test_qaic_samplers.py`
> (`TestQaicSamplers`) — same coverage (presence/frequency/repetition
> penalty, top-p, top-k), different class name; the leftover class has been
> removed from `test_samplers.py`.

---

### `spd/`

**Speculative decoding (SpD/PLD/DLM) — metadata, budget, config (pure Python)**

- **66 tests** (66 run, 0 skip)

| File | Tests | Runs | Skips |
|------|-------|------|-------|
| `test_calc_spec_decode_metadata.py` | 4 | 4 | 0 |
| `test_max_num_batched_tokens.py` | 17 | 17 | 0 |
| `test_spec_decode_unit.py` | 13 | 13 | 0 |
| `test_variable_decode_specializations.py` | 32 | 32 | 0 |

> **API note:** Uses `speculative_config={"method": "ngram", "num_speculative_tokens": 3, ...}` dict format (not deprecated `speculative_model="[ngram]"`).

**`TestDraftModelConfigExtraction`** (5 tests): draft_override takes priority, fallback, empty cases

**`TestSpDConfigValidation`** (2 tests): LoRA+SpD raises, ODS config parsing

> On-device tests (`test_ngram_spd_output_non_empty` and
> `test_ngram_spd_matches_baseline`, formerly in `test_spec_decode_unit.py`;
> plus the redundant `test_spd_integration.py`, which depended on a
> nonexistent `llm` fixture and has been removed) now live in
> `vllm-qaic/tests/device_unit_and_e2e/test_unit_qaic_spd.py`.

---
