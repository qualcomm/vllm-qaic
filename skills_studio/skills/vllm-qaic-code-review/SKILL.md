---
name: vllm-qaic-code-review
description: Repo-specific code review for vllm-qaic. Auto-detects PyT/AoT mode from the diff, spawns the matching rule overlays, anchors every finding to a real diff line. Use when asked to review vllm-qaic changes — the working tree, a PR number, a path, or a fork branch URL. Accepts `--runs=2` to review twice and vote.
---

# vllm-qaic-code-review — vllm-qaic code review

Rules were mined from roughly a thousand human reviewer comments on this
project's own review history — they encode what reviewers here actually ask for,
not generic Python or vLLM advice. The worker contract lives in the
`vllm-qaic-code-review` agent (`skills_studio/agents/vllm-qaic-code-review.md`;
activate it with `make claude` or `make codex` from the repo root).

Args: `$ARGUMENTS`

---

## Step 1 — Collect the diff

Pick the first that applies:

| Arg form | Command |
|---|---|
| empty | `git --no-pager diff --no-color HEAD` |
| `<N>` (digits) | `gh pr diff <N> --repo qualcomm/vllm-qaic` |
| a GitHub branch link — `https://github.com/<owner>/<repo>/tree/<branch>`, or a repo URL (`git@github.com:<owner>/<repo>.git` / `https://github.com/<owner>/<repo>`) plus a branch name given in prose | Clone-and-checkout workflow — see **Step 1a** |
| a commit or revision range (`<sha>`, `<base>..<head>`) | `git --no-pager diff --no-color <range>` |
| a path | `git --no-pager diff --no-color HEAD -- <path>` |

Also note `--runs=2` if present (see Step 3).

If the diff is empty, say so and stop.

**Transport.** If the diff is ≤ ~50 KB, inline it in each agent prompt. If
larger, pass each agent the exact command to re-run instead. Never write a
diff file *inside* a repo checkout — CI checkouts are read-only and a stray
tracked-looking file would dirty `git status`. A scratch file for transport
(e.g. handing an inline-sized diff to several subagent prompts at once) belongs
in a temporary directory, never inside any clone.

## Step 1a — Cloning a fork + branch

Triggered whenever Step 1's arg is a GitHub branch link, or a repo URL paired
with a branch name. Fixed procedure, run it the same way every time:

1. **Parse** `<owner>`, `<repo>`, `<branch>` from the input. A `/tree/<branch>`
   URL segment can itself contain `/` (branch names like `dev/v0.30.0`) — take
   everything after `/tree/` as the branch, not just the first segment.
2. **Location.** Clone into a scratch directory **outside** any existing
   checkout: `${TMPDIR:-/tmp}/vllm-qaic-review/<owner>-<repo>`. **Never** clone
   into the user's own working clone of `vllm-qaic`, and never nest the clone
   inside it — a review must not touch the checkout the user is working in.
   State the path you chose before cloning.
3. **Reuse, don't re-clone.** If that directory already exists as a git repo,
   `cd` into it and `git fetch origin` instead of cloning fresh.
4. **Clone.** `git clone --no-single-branch git@github.com:<owner>/<repo>.git <path>`.
   If SSH isn't set up, fall back to `https://github.com/<owner>/<repo>.git`.
5. **Checkout.** `git fetch origin <branch>` then `git checkout <branch>`
   (use `git switch -c <branch> origin/<branch>` if no local branch exists yet).
6. **Determine the diff.** Default to everything unique to the branch: find
   the fork's default branch (`git symbolic-ref refs/remotes/origin/HEAD`,
   normally `main`) and diff against their merge-base with three-dot syntax:
   `git --no-pager diff --no-color origin/<default>...<branch>`
   This is "what this branch would show as a PR against `origin/<default>`".
7. **Sanity-check the range before reviewing.** Run
   `git log --oneline origin/<default>...<branch>` first. If the commit list
   visibly bundles more than one independent concern — e.g. two separate
   version bumps, or a rebase plus an unrelated feature — don't silently pick
   one; print the commit list and ask which range to review, per Step 2's
   existing ambiguity policy. Otherwise proceed with the full range.
8. Continue at **Step 2** with the resulting diff.

## Step 2 — Route: detect mode from changed paths

Classify every changed file:

| Path pattern | Bucket |
|---|---|
| `vllm_qaic/worker/worker.py` → `QaicWorkerAoT` hunks, `vllm_qaic/worker/model_runner.py` → `QaicModelRunnerAoT` hunks | **aot** |
| `vllm_qaic/model_loader/qaic.py`, `qaic_session_np.py`, `spec_decode/**`, `distributed/kv_transfer/**`, `patch/patch_graph_pickler.py`, `patch/patch_rejection_sampler.py` | **aot** |
| `QaicWorkerPyt` / `QaicModelRunnerPyt` hunks, `vllm_qaic/ops/**`, `vllm_qaic/_custom_ops.py` | **pyt** |
| `vllm_qaic/platform_base.py`, `platform.py`, `envs.py`, `patch/**` (other), `utils/**`, `attention/**`, `executor/**`, `quantization/**`, base `QaicWorker`/`GPUModelRunner` hunks | **shared** |
| `vllm/**` (upstream) | **shared** — fires `patch.no_direct_upstream_edit` |
| `tests/**`, `examples/**`, `docs/**`, `ci_scripts/**`, `setup.py`, `requirements/**` | **shared** (plus **pyt** if under an `eager/` test path) |
| `scripts/utility.sh`, `docker/Dockerfile.aot`, `docker/Dockerfile.pyt`, `docs/installation.md` (version tables) | **shared** — build/version-pin files; fires `build.qaic_sdk_version_sync` / `build.torch_version_sync` |
| `skills_studio/**`, `*.md`, `Makefile`, dotfiles | **shared** — no mode signal, but still publicly visible: fires `leak.no_machine_specific_path` / `leak.no_internal_reference` |

Rules 33–34 (`leak.*`) are **not** routed — they apply to every changed file in
every bucket, including files whose only change is prose.

Then decide which roles to spawn:

- Only **aot** paths → spawn `shared` + `aot`.
- Only **pyt** paths → spawn `shared` + `pyt`.
- Both, **or any `shared` path that contains an `is_aot` / `enforce_eager`
  branch in the diff** → spawn `shared` + `aot` + `pyt`, and say so:
  *"Diff straddles shared code; reviewing both modes."*
- Only **documentation, assistant assets, or other no-mode-signal files** →
  spawn `shared` alone. Don't ask which mode to review; there is no mode
  question, and the `leak.*` rules still have to run.

**When to stop and ask.** If routing is genuinely ambiguous — a new *code* file
whose mode you cannot infer, or a `shared` file where you cannot tell which mode
the changed branch serves — **do not guess**. Use `AskUserQuestion` with the
specific files listed, offering: review as AoT / review as PyT / review both.
Ambiguity here is exactly what the user asked to be consulted on.

Print the routing decision before Step 3.

## Step 3 — Fan out (parallel)

Spawn every selected role in **one** assistant message so they run concurrently.
`subagent_type: "vllm-qaic-code-review"`. Each prompt =
`[ROLE] <name>` + the role's `RULE CHECKLIST` from Step 3a + `[REPO CHECKOUT PATH]`
(the absolute path of the checkout the diff came from — the primary working
clone for a local/PR diff, or the Step 1a clone path for a branch review) +
`[DIFF TO REVIEW]`. `[REPO CHECKOUT PATH]` exists so `build.qaic_sdk_version_sync`
and `build.torch_version_sync` can read sibling files the diff didn't touch.

Default is **one run per role**. If `--runs=2` was passed, spawn each role twice
on the identical diff; the merger unions results and attaches `votes: k/2`.

### Step 3a — Role rule checklists

Paste the matching block verbatim into that role's prompt.

---

#### ROLE: shared — 34 rules

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
| 18 | `config.localize_to_check_and_update_config` | BLOCKING | A `vllm_config` field (or an alias — `self.parallel_config`, `self.model_config`, etc.) is mutated somewhere other than `Platform.check_and_update_config` (`platform_base.py:173`) — e.g. inside a worker/runner `__init__`, or in a patch module |
| 19 | `error.no_silent_except` | BLOCKING | Bare or broad `except` that neither logs nor re-raises |
| 20 | `error.mode_capability_guard` | BLOCKING | New feature reachable in a mode that cannot support it, with no `is_aot` guard. Anchor: `platform_base.py:283-297` rejects disagg serving, spec decode, async scheduling in eager |
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

---

#### ROLE: aot — 8 rules

| # | Rule | Severity | Trigger |
|---|---|---|---|
| 1 | `aot.ccl_defaults` | BLOCKING | `comp_ctx_lengths_prefill` / `_decode` left empty instead of defaulting to `[max_model_len]` when CCL is disabled (`ccl_enabled` false/absent) |
| 2 | `aot.mdp_stages_consistency` | BLOCKING | `mdp_num_partitions` and `cfg['stages']` set independently with no assert that they agree |
| 3 | `aot.qpc_dict_keys` | BLOCKING | QPC path extracted from the compile-result dict without validating the expected keys exist |
| 4 | `aot.specialization_covers_shapes` | BLOCKING | New runtime shape/seq-len path with no matching compile specialization (`prefill_seq_len`, `vision_size`) |
| 5 | `aot.session_input_guard` | BLOCKING | Feeding a session input without checking `in self.session.input_names` |
| 6 | `disagg.device_group_validation` | BLOCKING | Prefill/decode/encode device groups unvalidated — empty group, or duplicate QIDs across groups |
| 7 | `disagg.proc_lifecycle` | non-blocking | New subprocess/resource-tracker with no account of the P/D process fan-out (6P4D → 10 vllm processes, each spawning trackers) |
| 8 | `spd.aot_only` | BLOCKING | Spec-decode path reachable without the AoT guard (`platform_base.py:288-291`) |

---

#### ROLE: pyt — 13 rules

**Framing.** Eager mode has little direct review history; the team develops
against AoT and treats PyT as something that must not regress. Rules 1–3 are the
core job: catching eager-mode breakage from changes made for AoT.

| # | Rule | Severity | Trigger |
|---|---|---|---|
| 1 | `pyt.cross_mode_regression` | BLOCKING | Diff changes shared code (`platform_base.py`, `worker.py`, `model_runner.py`, `patch/**`) on a path eager mode also takes, with no evidence the eager impact was considered. Set `affects_other_mode: true` |
| 2 | `pyt.per_device_seed` | BLOCKING | Global seed set where eager needs per-device (`torch.manual_seed` per device) |
| 3 | `pyt.eager_only_accessor` | BLOCKING | `get_num_cores()` / `get_num_hvx_threads()` called on a path reachable in AoT — they raise there (`platform_base.py:124-136`) |
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

---

## Step 4 — Merge

One `Agent()` call, `subagent_type: "vllm-qaic-code-review"`, `[ROLE] merger`, with
every role's YAML plus `[FULL DIFF]`. The merger must:

1. **Re-verify every anchor** — re-grep the diff for each `quoted_line`. Drop
   any `fail` whose quote is not a literal substring. This is a hard gate.
2. **Surface `leak.*` findings first, then cross-mode findings.** A `leak.*`
   finding leads the report: it is the only class here that a follow-up commit
   cannot fix, because the value survives in git history. Next, any finding with
   `affects_other_mode: true` — the bug class the single-mode reviewer misses and
   the whole reason both overlays run.
3. **Dedupe** across roles by `(file, line, rule)`.
4. **Under `--runs=2` only**: union runs A and B by `(file, line, rule)`, attach
   `votes: k/2`. Votes annotate confidence; they do not gate the verdict.
5. **Tally coverage** per role from `pass`/`fail`/`na` counts.
6. **Aggregate the verdict:**

| Findings | Verdict |
|---|---|
| Any `BLOCKING` | `CHANGES REQUIRED` |
| Only `non-blocking` | `APPROVED WITH COMMENTS` |
| None | `APPROVED` |

## Step 5 — Output

```text
Routing: <files> → roles <shared, aot, pyt>
Coverage: shared 34/34 (2 failed, 9 n/a) · aot 8/8 (1 failed, 5 n/a)

## Leaks (machine-specific paths, internal references)
<leak.* findings, or "none">

## Cross-mode hazards
<findings with affects_other_mode, or "none">

## BLOCKING
- `<file>:<line>` **<rule>** — <message>
  > <quoted_line>
  Fix: <suggested_fix>

## Non-blocking
<same shape>

## Verdict
<CHANGES REQUIRED | APPROVED WITH COMMENTS | APPROVED>
```

Print the coverage line always — a silently skipped rule must be visible.

## Step 6 — Fix offer

Ask whether to apply fixes. On yes, apply one precise edit per finding, printing
one confirmation line each. Never fix and report in the same breath — the user
sees the review before anything changes.
