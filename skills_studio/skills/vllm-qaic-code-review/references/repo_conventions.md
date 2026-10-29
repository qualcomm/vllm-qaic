# vllm-qaic repo conventions (reviewer's quick reference)

The contract the rule files judge against. Every fact here is anchored to a
symbol in the tree rather than a line number, so it survives edits — grep for
the symbol to find the current location. If a fact here disagrees with the
tree, the tree wins: say so in the review instead of citing this file.

## What the repo is

`vllm_qaic` is a **vLLM plugin, not a fork**. It registers a QAIC platform,
worker, model runner, attention backend, and ops with upstream vLLM through
vLLM's plugin interfaces. Upstream vLLM is installed as a dependency; the
plugin never carries a modified copy of it.

## The two modes

| Mode | Execution | Package it needs | Selected by |
|---|---|---|---|
| `aot` | ahead-of-time compile via QEfficient + AI compiler | `QEfficient` | `./scripts/install.sh aot` |
| `pyt` | eager execution via `torch_qaic` | `torch_qaic` | `./scripts/install.sh pyt` |

- Mode is **install-time**: `QaicPlatform.is_aot = not _torch_qaic_installed`
  (`vllm_qaic/platform_base.py`). Callers use `current_platform.is_aot_inference()`
  or `cls.is_aot` inside the platform. It is never a runtime or per-request flag.
- The two modes are mutually exclusive per environment, so code reachable in
  both modes must not import `QEfficient` or `torch_qaic` at module scope —
  import lazily inside the mode branch. The established guard for a
  module-scope import in shared code is `try: import torch_qaic ... except
  ImportError: qaic = PlaceholderModule("qaic")` (`platform_base.py`,
  `worker/model_runner.py`). `vllm_qaic/_custom_ops.py` is PyT-only and imports
  `torch_qaic` unguarded.
- Mode-specific classes: `QaicWorkerAoT` / `QaicWorkerPyt` (`vllm_qaic/worker/worker.py`,
  both subclass `QaicWorker`), `QaicModelRunnerAoT` / `QaicModelRunnerPyt`
  (`vllm_qaic/worker/model_runner.py`, both subclass upstream `GPUModelRunner`).
  `QaicPlatform.worker_cls_name` picks the worker from `is_aot`.
- **Eager-only accessors.** `QaicPlatform.get_num_cores()` and
  `get_num_hvx_threads()` raise `NotImplementedError` in AoT.
- **Deliberate capability asymmetry.** In `QaicPlatform.check_and_update_config`,
  the `if not cls.is_aot:` branch rejects disaggregated serving
  (`kv_transfer_config`), raises on speculative decoding, and turns async
  scheduling off. This asymmetry is intended; a *new* feature that ignores it
  is the bug.

## Where things belong

| Kind of change | Home |
|---|---|
| Config validation / mutation of `vllm_config` and its sub-configs | `QaicPlatform.check_and_update_config` (`vllm_qaic/platform_base.py`) |
| Platform capability / hook override | `vllm_qaic/platform_base.py`, `vllm_qaic/platform.py` |
| Op override | a `CustomOp` subclass under `vllm_qaic/ops/`, registered with `CustomOp.register_oot(_decorated_op_cls=..., name=...)` in `vllm_qaic/ops/__init__.py` |
| `qaic::*` custom op definitions (`torch.library.custom_op`) | `vllm_qaic/_custom_ops.py` |
| Upstream behavior that can't be reached any other way | a monkey-patch module `vllm_qaic/patch/patch_<target>.py`, imported and described in `vllm_qaic/patch/__init__.py` |
| Plugin environment variables | `vllm_qaic/envs.py` (`TYPE_CHECKING` annotation + entry in `qaic_environment_variables`) |
| Logging | `vllm_qaic/logger.py`, `%`-style args (ruff `G` is enabled) |

### Escalation order for touching upstream behavior

Extend `vllm_qaic/**` → override a Platform hook → `CustomOp.register_oot` →
monkey-patch in `vllm_qaic/patch/` → **never** edit upstream `vllm/`.

### Patch module conventions

- `vllm_qaic/patch/__init__.py` opens with a "What's Patched and how it works"
  block: per patch file, the patched upstream symbol, a **Why**, and a **How**.
  A new patch module is expected to add its entry there.
- A patch rebinds an existing upstream symbol (`vllm.x.y.fn = _patched_fn`);
  it does not bolt new names onto upstream modules.
- Vendored upstream code in a patch keeps upstream's style; the QAIC-specific
  edits inside it are delimited by comments. Vendored files carry the upstream
  `SPDX-FileCopyrightText` / `Apache-2.0` lines below the Qualcomm header.

## Headers

Every source file starts with:

```text
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
```

Note: **BSD-3-Clause-Clear**. Shell scripts put it after the shebang.
`tools/pre_commit/check_qualcomm_header.py` enforces it for Python only.

## What pre-commit already enforces (don't re-review)

`.pre-commit-config.yaml` runs ruff check/format, typos, clang-format,
markdownlint, actionlint, mypy, shellcheck, DCO sign-off, and the local checks
in `tools/pre_commit/`: Qualcomm header (Python), lazy imports in
`vllm_qaic/__init__.py`, no spaces in filenames, forbidden imports
(`pickle`/`cloudpickle`, `base64` → `pybase64`, `re` → `regex`), config
dataclass defaults + docstrings, and boolean ops in `with` statements. A
review does not repeat these as findings; they fail CI on their own.

## Version pins

These constants live in several files that must agree:

| Constant | Sources |
|---|---|
| `VLLM_QAIC_VERSION` | `get_qaic_sdk_version()` in `setup.py`; `scripts/utility.sh`; every `ARG VLLM_QAIC_VERSION` stage in `docker/Dockerfile.aot` and `docker/Dockerfile.pyt`; rows in `docs/installation.md` |
| `TORCH_VERSION_AOT`, `TORCHVISION_VERSION_AOT` | `scripts/utility.sh`; every `ARG` stage in `docker/Dockerfile.aot`; `docs/installation.md` |
| `TORCH_VERSION_PYT`, `TORCHVISION_VERSION_PYT`, `TORCHAUDIO_VERSION_PYT` | `scripts/utility.sh`; every `ARG` stage in `docker/Dockerfile.pyt`; `docs/installation.md` |

`docs/installation.md` has **several independent tables** (one per
Dockerfile's ARGs, plus a "Version Reference" table for `scripts/utility.sh`),
and they drift independently. `scripts/run_review_greps.sh` prints every
occurrence and flags disagreement.

## Tests

- `tests/unit` — no hardware, no compile. Pure logic (shapes, block/KV index
  math, config validation, arg plumbing) goes here.
- `tests/e2e` — needs QAIC devices. Fixtures in `tests/e2e/conftest.py`:
  `qaic_model` (offline `LLM`, **class-scoped**) and `server_runner`
  (OpenAI-compatible server subprocess). Configure via
  `@pytest.mark.qaic_test_config(...)` (`model_name`, `seq_len`, `ctx_len`,
  `decode_bsz`, `dtype`, `kv_dtype`, `num_device_groups`/`device_group_size`,
  LoRA and disagg keys).
- Markers registered in `pyproject.toml`: `qaic_test_config`, `qaic_aot_mode`,
  `qaic_disagg_installed`.
- `ci_scripts/collect_jobs.py` reads the same marker kwargs to size each CI
  job's device allocation.
- CI entry points: `ci_scripts/ci_fasttest_qaic_aot.sh`,
  `ci_scripts/ci_fasttest_qaic_pyt.sh`, `ci_scripts/ci_pipeline_run_plugin.sh`.
  `pytest -n` is not used — it races on compile caches and double-books devices.

## Public repository

The repo is public, so anything committed — including Markdown and the assets
under `skills_studio/` — is public permanently. A machine-specific path or
internal reference can't be fixed by a follow-up commit, because the value stays
in git history.
