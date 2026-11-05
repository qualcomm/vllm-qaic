# Role: shared — 39 rules

Applies to every diff. Rules 1–34 were mined from this repo's human review
history. Rules 35–39 restate the contributor ground rules in
`skills_studio/AGENTS.md` that pre-commit does **not** enforce; they are
documented policy, not mined history. Background for every rule is in
`repo_conventions.md`.

| # | Rule | Severity | Trigger |
|---|---|---|---|
| 1 | `patch.no_direct_upstream_edit` | BLOCKING | Diff contains a `+++ b/vllm/...` hunk (upstream file modified). Upstream files must NEVER be edited. Escalation order: extend `vllm_qaic/**` → override a Platform hook → `CustomOp.register_oot` → monkey-patch in `vllm_qaic/patch/` → never edit upstream |
| 2 | `patch.last_resort_justified` | non-blocking | A **new** `patch_*.py` module with no comment/commit text saying why extending the plugin, overriding a Platform hook, or `register_oot` was insufficient |
| 3 | `patch.replace_not_append` | BLOCKING | Patch module *adds* a new name to an upstream module instead of rebinding the existing symbol (`vllm.x.y.fn = _patched_fn`) |
| 4 | `patch.mode_gated` | BLOCKING | A patch valid in only one mode applied unconditionally — no `is_aot` / `is_aot_inference()` / `enforce_eager` guard |
| 5 | `patch.documented_why` | non-blocking | New import added to `vllm_qaic/patch/__init__.py` without an explanatory comment, unlike its neighbours |
| 6 | `patch.mark_qaic_blocks` | non-blocking | Patch file vendoring upstream code doesn't delimit the QAIC-specific edits with comments |
| 7 | `patch.file_named_for_target` | non-blocking | `patch_*.py` filename doesn't name what it patches |
| 8 | `patch.prefer_platform_override` | non-blocking | Patching where a `Platform`/base-class override in `platform.py` would work |
| 9 | `ops.forward_oot_only` | BLOCKING | A `CustomOp` subclass defines `forward_static` (or another hook) when `forward_oot` alone suffices |
| 10 | `ops.register_oot` | BLOCKING | Op override not registered via `CustomOp.register_oot(...)`, or a stray `@CustomOp.register_oot` decorator left on a class after registration moved to `ops/__init__.py` |
| 11 | `upstream.minimize_diff` | BLOCKING | Upstream-derived code rewritten with no algorithmic change (reformat, rename, restructure) |
| 12 | `upstream.no_deletion_on_bump` | BLOCKING | Version-bump diff deletes upstream code rather than merging plugin changes on top of it |
| 13 | `upstream.comment_for_upstreaming` | non-blocking | Non-obvious QAIC-specific edit with no explanatory comment — either in a file destined for upstreaming, or in logic a later reader cannot reconstruct |
| 14 | `upstream.reuse_existing_arg` | non-blocking | New CLI arg duplicates an existing upstream argument |
| 15 | `config.validate_keys` | BLOCKING | `override_qaic_config` / compile-config dict indexed as `cfg["k"]` with no membership check or `.get` default |
| 16 | `config.str_and_bool_forms` | BLOCKING | Config value treated as bool when it can arrive as a string from CLI/env (`'false'`, `'0'`, `'true'`) |
| 17 | `config.explicit_arg_errors` | non-blocking | Missing/empty required arg group raises a generic error, or none |
| 18 | `config.localize_to_check_and_update_config` | BLOCKING | A `vllm_config` field (or an alias — `self.parallel_config`, `self.model_config`, etc.) is mutated somewhere other than `Platform.check_and_update_config` (`vllm_qaic/platform_base.py`) — e.g. inside a worker/runner `__init__`, or in a patch module |
| 19 | `error.no_silent_except` | BLOCKING | Bare or broad `except` that neither logs nor re-raises |
| 20 | `error.mode_capability_guard` | BLOCKING | New feature reachable in a mode that cannot support it, with no `is_aot` guard. Anchor: the `if not cls.is_aot:` block in `QaicPlatform.check_and_update_config` rejects disagg serving, spec decode, async scheduling in eager |
| 21 | `error.no_none_for_empty` | non-blocking | Returns `[]` where callers test `None`, or vice versa |
| 22 | `concurrency.unbounded_queue_get` | BLOCKING | `queue.get()` with no `timeout=` on a step-to-step handoff |
| 23 | `state.no_shared_mutation` | BLOCKING | In-place mutation of an object held by reference from shared state (use `dataclasses.replace`) |
| 24 | `state.parallel_paths_updated` | BLOCKING | Fast path updates a field that its sibling scatter/reorder path does not |
| 25 | `concurrency.startup_timeout` | non-blocking | Work added to an import/startup path already near a supervisor timeout (uvicorn's 5 s heartbeat) |
| 26 | `hygiene.no_nonblocking_cpu_copy` | non-blocking | `non_blocking=True` on a CPU→CPU `copy_()` |
| 27 | `hygiene.confirm_init` | non-blocking | Attribute used whose initialization isn't visible; risks silently defaulting a feature off |
| 28 | `hygiene.dead_code` | non-blocking | Commented-out code, unused helper, leftover debug print |
| 29 | `change.justify_new_knob` | non-blocking | New env var, CLI arg, config key, constructor param, or per-iteration initialization with no comment or commit text saying why it is needed. Asks for rationale, not a change — never BLOCKING |
| 30 | `change.derive_dont_duplicate` | non-blocking | Value taken as a new arg/env/param when already reachable from an object in scope (`vllm_config`, `cache_config`, `self.*`), or two parameters introduced for one quantity |
| 31 | `build.qaic_sdk_version_sync` | BLOCKING | Diff changes `get_qaic_sdk_version()` in `setup.py`, `VLLM_QAIC_VERSION` in `scripts/utility.sh`, any `ARG VLLM_QAIC_VERSION` in `docker/Dockerfile.aot` / `docker/Dockerfile.pyt`, or a `VLLM_QAIC_VERSION` row in a `docs/installation.md` reference table — read the other sources directly from `[REPO CHECKOUT PATH]`; fail if they don't all agree with the value the diff introduces, and fail if a docs table row shows a value copy-pasted from a different constant (e.g. a `VLLM_QAIC_VERSION` row showing `VLLM_VERSION`'s value) |
| 32 | `build.torch_version_sync` | BLOCKING | Diff changes a `TORCH_VERSION_AOT` / `TORCHVISION_VERSION_AOT` pin in `scripts/utility.sh` or `docker/Dockerfile.aot`, a `TORCH_VERSION_PYT` / `TORCHVISION_VERSION_PYT` / `TORCHAUDIO_VERSION_PYT` pin in `scripts/utility.sh` or `docker/Dockerfile.pyt`, or the matching row in a `docs/installation.md` reference table — read the sibling files directly from `[REPO CHECKOUT PATH]`; fail if the AOT trio, PYT trio, or a docs table row disagree (check every `ARG` occurrence across build stages, not just the first) |
| 33 | `leak.no_machine_specific_path` | BLOCKING | Added line hard-codes an absolute path that exists only on one machine or one person's account — a user home, a personal or team workspace tree, a network/scratch mount, a nightly-build or artifact drop directory, or a specific developer checkout — instead of deriving it from a CLI arg, env var, config value, or repo-relative path. Applies to docs, tests, CI scripts, Dockerfiles, and assistant assets exactly as to library code. See the carve-out for paths that are legitimately absolute |
| 34 | `leak.no_internal_reference` | BLOCKING | Added line names something a reader outside the company cannot resolve, in what is a **public** repository: an internal code-review or Git server URL, an internal package index or artifact host, an internal-only project or mirror name, a lab/host/device identifier, an individual's username or email address, or any credential, token, or key. Quote the minimal substring needed to anchor, and describe the class of leak in `message` — never reproduce a secret's value in the finding |
| 35 | `mode.lazy_package_import` | BLOCKING | Added module-scope `import QEfficient…` / `from QEfficient…` / `import torch_qaic…` / `from torch_qaic…` in a file reachable in both modes, not wrapped in `try: … except ImportError:` (the `PlaceholderModule` idiom in `platform_base.py`) and not inside an `is_aot` / `is_aot_inference()` branch. The other mode's package is not installed, so the import crashes plugin load there. Set `affects_other_mode: true` |
| 36 | `config.env_via_envs_py` | non-blocking | Added `os.getenv` / `os.environ.get` / `os.environ[...]` read of a **plugin** setting under `vllm_qaic/**` outside `vllm_qaic/envs.py`, or a new `vllm_qaic/envs.py` variable missing its `TYPE_CHECKING` annotation, default, or docstring. Not for writes to third-party/runtime variables (`KMP_*`, `LD_PRELOAD`, the device-control var) |
| 37 | `hygiene.log_not_print` | non-blocking | Added `print(` under `vllm_qaic/**`. Use `vllm_qaic/logger.py` with `%`-style args |
| 38 | `hygiene.license_header` | BLOCKING | New non-Python source file (shell script, C/C++ source or header, Dockerfile) whose added lines lack the `Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.` + `SPDX-License-Identifier: BSD-3-Clause-Clear` header. Python is already enforced by pre-commit — `na` for `.py` |
| 39 | `hygiene.no_committed_artifacts` | BLOCKING | Diff adds a generated or runtime artifact: QPC output, `*.onnx`, model weights (`*.safetensors`, `*.bin`, `*.pt`, `*.gguf`), anything under `scheduler_output/`, `*.log`, `*.pyc` / `__pycache__/`, editor backups. Anchor on the `diff --git` or `+++ b/` line naming the file |

**Note on rules 33–34.** This repository is public. These two rules are the last
gate before a local convenience becomes a permanent part of public history —
and unlike most rules here, a miss cannot be fixed by a follow-up commit,
because the value stays in the git history. Apply them to **every** changed
file, including Markdown, CI scripts, and the assistant assets under
`skills_studio/`, which are as publicly visible as the plugin source. They are
the one place where the review is looking at what the text *says*, not at what
the code *does*, so a path or URL inside a comment, docstring, or prose
paragraph counts exactly as much as one in a string literal. Both have
carve-outs in the agent contract — read them before firing, since legitimate
absolute paths (the documented SDK install tree, container-internal paths,
temporary directories) are common in this repo and must not be flagged.

**Note on rules 31–32.** These are the only shared rules needing more than the
diff: detecting drift against an untouched sibling file requires reading
`scripts/utility.sh`, both Dockerfiles, and `docs/installation.md`'s reference
tables directly from `[REPO CHECKOUT PATH]`. `docs/installation.md` carries
multiple independent tables (one per Dockerfile's ARGs, plus a
`scripts/utility.sh` constant table) that have drifted silently before,
including a row whose value was copy-pasted from a different constant
entirely (`VLLM_QAIC_VERSION` shown as `VLLM_VERSION`'s value) — treat a docs
table row that matches neither the diff's new value nor its old value as
equally suspicious as one that just looks stale. The anchoring hard gate still
applies — `quoted_line` must be the diff's own `+` line introducing the new
value; the mismatch is explained in `message`, never quoted from a file the
diff left untouched. If none of these locations was touched by this diff,
both rules are `na`.

**Note on rule 35.** `vllm_qaic/_custom_ops.py` is PyT-only and imports
`torch_qaic` unguarded by design — `na` there. When `pyt.forbidden_import`
fires on the same line, the merger keeps one finding.
