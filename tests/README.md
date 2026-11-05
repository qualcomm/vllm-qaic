# vllm-qaic test suite

Guide for developers writing tests here. Start with `tests/e2e/conftest.py` — it has the QAIC
device pool, markers, and the `qaic_model`/`make_runner` fixtures. `tests/conftest.py` (one level
up) just provides the generic `VllmRunner` wrapper with no QAIC specifics.

## Writing a new test

Most tests need the `qaic_model` fixture plus a `qaic_test_config` marker. For
tests that run in both AOT and PyTorch eager mode, provide mode-specific
configuration so eager mode does not inherit AOT quantization settings:

```python
@pytest.mark.qaic_test_config(
    {
        "aot": dict(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            seq_len=128,
            ctx_len=256,
            decode_bsz=4,
            dtype="mxfp6",
            kv_dtype="mxint8",
        ),
        "eager": dict(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            seq_len=128,
            ctx_len=256,
            decode_bsz=4,
        ),
    }
)
class TestSomething:
    def test_x(self, qaic_model, sample_prompts):
        outputs = qaic_model.generate(sample_prompts, sampling_params)
        ...
```

For online-serving tests (spawn a real OpenAI-compatible server subprocess), use `server_runner`
instead. For multi-device runs, add `num_device_groups`/`device_group_size` to the marker and use
`device_groups`/`device_group`.

`qaic_test_config(**kwargs)` is the per-test/class config-override marker — resolved as marker
kwarg → matching CLI option → default. A single positional dictionary with `aot` and `eager`
keys selects mode-specific values. It's open-ended; common keys are `model_name`, `seq_len`,
`ctx_len`, `decode_bsz`, `dtype`, `quantization`, `kv_dtype`,
`num_device_groups`/`device_group_size`, LoRA options
(`enable_lora`, `max_loras`, `lora_modules`), and the disaggregated-serving worker/group-size keys
(`num_prefill_workers`, `prefill_device_group_size`, `num_decode_workers`,
`decode_device_group_size`, `prefill_max_num_seqs`).

### `qaic_model` is class-scoped — reuse it

```python
@pytest.fixture(scope="class")
def qaic_model(...):
```

Building a `VllmRunner`/`vllm.LLM` triggers a real AIC100 compile to QPC, which is expensive.
Because this fixture is class-scoped, it's built **once per test class** and every test method in
that class reuses the same live compiled model instead of recompiling per test. Group tests that
can share one config into the same class under one `qaic_test_config` marker rather than writing
bare module-level functions each with their own marker — the latter forces a recompile per test.

### Markers that gate whether a test runs at all

Registered in `pyproject.toml` and enforced in `tests/e2e/conftest.py`'s
`pytest_collection_modifyitems`, which adds a `skip` marker at collection time with a specific
reason:

- **`qaic_aot_mode(reason=None)`** — skipped unless
  `current_platform.is_aot_inference()` is true. An optional reason customizes
  the skip message, for example `@pytest.mark.qaic_aot_mode("requires dual-QPC execution")`.
- **`qaic_disagg_installed`** — skipped unless the `qaic_disagg` package is importable.
- Device-pool sizing (not a marker, same hook) — skipped if `num_device_groups * device_group_size`
  exceeds the run's device pool (`--device-pool-size` or `len(--device-id)`).

Add `qaic_aot_mode`/`qaic_disagg_installed` to any test that only makes sense under that mode/
package instead of hand-rolling a platform check or `importorskip`.

## CI runner scripts (`ci_scripts/`)

This suite uses its own scheduler instead of `pytest -n <N> --dist=loadscope`, so that compilation
(expensive, must not race) and device assignment (must not double-book) are handled correctly
across parallel jobs:

- **`collect_jobs.py`** — runs `pytest --collect-only`, groups tests into jobs (one per class/module
  scope), drops anything already skipped, and records each job's device requirement and config key.
- **`scheduler.py`** — dispatches each job as its own `pytest` subprocess with a disjoint slice of
  real device IDs, and serializes the first job per model/config so concurrent jobs don't race on
  the same QEfficient export/compile cache directory.
- **`ci_fasttest_qaic_<mode>.sh`** — runs `collect_jobs.py` then `scheduler.py` against `tests/e2e`.
- **`ci_pipeline_run_plugin.sh`** — installs the plugin, then sources the matching
  `ci_fasttest_qaic_<mode>.sh`.

If you add a test needing a nonstandard device count, make sure `qaic_test_config` reflects it —
`collect_jobs.py` reads the same marker kwargs to size the job for the scheduler.
