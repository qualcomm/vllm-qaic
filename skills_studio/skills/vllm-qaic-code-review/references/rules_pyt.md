# Role: pyt — 13 rules

Mined from this repo's human review history. Background in `repo_conventions.md`.

**Framing.** Eager mode has little direct review history; the team develops
against AoT and treats PyT as something that must not regress. Rules 1–3 are the
core job: catching eager-mode breakage from changes made for AoT.

| # | Rule | Severity | Trigger |
|---|---|---|---|
| 1 | `pyt.cross_mode_regression` | BLOCKING | Diff changes shared code (`platform_base.py`, `worker.py`, `model_runner.py`, `patch/**`) on a path eager mode also takes, with no evidence the eager impact was considered. Set `affects_other_mode: true` |
| 2 | `pyt.per_device_seed` | BLOCKING | Global seed set where eager needs per-device (`torch.manual_seed` per device) |
| 3 | `pyt.eager_only_accessor` | BLOCKING | `get_num_cores()` / `get_num_hvx_threads()` called on a path reachable in AoT — they raise `NotImplementedError` there (`QaicPlatform` in `platform_base.py`) |
| 4 | `pyt.test_location` | non-blocking | Eager custom-op test or kernel benchmark not under the eager test tree (`tests/**/eager/...`) |
| 5 | `pyt.torch_qaic_install` | non-blocking | Code assumes a `torch_qaic` wheel path (e.g. `/opt/qti-aic/integrations/`) instead of deferring to the installer script |
| 6 | `pyt.register_fake_meta` | BLOCKING | New `qaic::*` custom op registered without a matching `@register_fake("qaic::<op>")` meta kernel — `torch.compile` traces with FakeTensors and fails without it. Fake kernels belong in torch-qaic's `_meta_registrations.py`, NOT in vllm_qaic |
| 7 | `pyt.forbidden_import` | BLOCKING | Bare `import torch_qaic` / `from torch_qaic import ...` in changed Python |
| 8 | `pyt.no_scalar_hot_path` | non-blocking | Bulk per-element scalar work in a hot path that belongs on HVX/HMX |
| 9 | `pyt.contiguous_operands` | non-blocking | Non-contiguous operand handed to a device kernel without canonicalizing or falling back to CPU |
| 10 | `pyt.avoid_format_conversion` | non-blocking | Flat ↔ Crouton layout conversion inserted in a hot path |
| 11 | `pyt.bare_except` | non-blocking | `except:` with no exception type |
| 12 | `pyt.type_hints` | non-blocking | New public function missing type hints |
| 13 | `pyt.snake_case` | non-blocking | Non-snake_case function or file name |
