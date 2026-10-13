## `vllm-qaic` — Agent Instructions

Ground rules and commands for humans and coding agents contributing to this repo.
Process/PR mechanics live in [CONTRIBUTING.md](CONTRIBUTING.md); install detail lives in
[docs/installation.md](docs/installation.md); test-suite internals live in
[tests/README.md](tests/README.md). This file is the short operational summary.

## What this repo is

`vllm_qaic` is a **vLLM plugin**, not a fork. It registers a QAIC platform, worker, model
runner, attention backend, and ops with upstream vLLM. Two mutually exclusive inference modes,
one per environment:

| Mode | How | Selected by |
|---|---|---|
| `pyt` | eager execution via `torch_qaic` | `./scripts/install.sh pyt` |
| `aot` | ahead-of-time compile via QEfficient + AI compiler | `./scripts/install.sh aot` |

Target: Python 3.12, vLLM pinned in `pyproject.toml` (`[tool.uv.sources]`), Cloud AI SDK >= 1.22.

## Contribution ground rules

1. **Branch off `main`, PR into `main`.** Keep each PR to one logical change; split unrelated
   work. Sign off every commit (`git commit -s`) — DCO is enforced.
2. **Plugin discipline — don't fork upstream behavior.** Extend vLLM through its plugin
   interfaces. If upstream must be changed, put it in `vllm_qaic/patch/` as a narrow, documented
   monkeypatch with the upstream version it targets, rather than duplicating an upstream file.
3. **Respect the mode boundary.** Code reachable in both modes must not import AOT-only
   (`QEfficient`) or PYT-only (`torch_qaic`) packages at module scope. Gate on
   `current_platform.is_aot_inference()` and import lazily inside the branch. Never assume the
   other mode's package is installed.
4. **Keep imports lazy at package roots.** `vllm_qaic/__init__.py` is imported during vLLM
   plugin discovery — heavy/optional imports there break unrelated setups. Enforced by
   `tools/pre_commit/check_init_lazy_imports.py`.
5. **Configuration goes through `vllm_qaic/envs.py`**, not scattered `os.getenv` calls. Every
   new env var needs a type annotation, a default, and a docstring.
6. **No silent fallbacks.** If a feature is unsupported on QAIC, raise with an actionable
   message naming the unsupported option. A quiet degradation to a wrong-but-running path is a
   worse bug than a loud failure.
7. **Log, don't print**, via `vllm_qaic/logger.py`. Use `%`-style args, not f-strings (ruff `G`).
8. **Forbidden imports:** `pickle`/`cloudpickle` (security), `base64` (use `pybase64`), `re`
   (use `regex as re`). See `tools/pre_commit/check_forbidden_imports.py`.
9. **Every source file carries the Qualcomm copyright + `SPDX-License-Identifier` header.**
10. **Don't hand-edit generated/vendored artifacts** or commit QPCs, ONNX dumps, model weights,
    or `scheduler_output/`.
11. **Tests and docs ship with the change**, not in a follow-up. A behavior change with no test
    is treated as unverified.

## Setup

```bash
# one-time: hooks (formatting, lint, typos, license headers, DCO sign-off)
pip install -r requirements/lint.txt
pre-commit install

# install the plugin into an activated py3.12 conda env / venv — pick ONE mode
./scripts/install.sh aot
./scripts/install.sh pyt

# test deps (single source of truth)
pip install -r requirements/test.txt
```

Useful install knobs (full list in `scripts/install.sh` header):
`TRANSFORMERS_VERSION_AOT` / `TRANSFORMERS_VERSION_PYT` (pin transformers),
`VLLM_BUILD_RUST=1` (build vLLM's Rust frontend from source),
`VLLM_QAIC_INSTALL_SOURCE=wheel` + `VLLM_QAIC_SDK_PATH=<dir>` (install a prebuilt SDK wheel).

Sanity check after install:

```bash
python -c "from vllm.platforms import current_platform; \
print(current_platform.device_name, current_platform.is_aot_inference())"
/opt/qti-aic/tools/qaic-util -q | grep -E 'QID|Status'   # devices Ready?
```

## Agents and skills (`skills_studio/`)

Repo-specific AI assistant assets are version-controlled under `skills_studio/`:

```text
skills_studio/
├── AGENTS.md   # this file; symlinked to the repo root as AGENTS.md and CLAUDE.md
├── agents/     # subagent definitions   → <name>.md  (Markdown + YAML frontmatter)
├── commands/   # slash commands         → <name>.md
├── skills/     # skills                 → <name>/SKILL.md (+ scripts/, references/)
└── tools/      # helpers used by `make codex`
```

This file lives here rather than at the root because it is one of the assistant assets, not a
build input. The root `AGENTS.md` and `CLAUDE.md` are both relative symlinks to it, so Codex and
Claude Code read the same guide and it can never drift. Edit `skills_studio/AGENTS.md`.

No assistant reads `skills_studio/` directly — each looks in its own dotdir, and those are
gitignored. So `skills_studio/` is the reviewed source of truth, and `.claude/`, `.agents/`
and `.codex/` are per-developer activation layers built from it.

### Importing them

From the repo root:

```bash
make claude      # Claude Code  → .claude/{skills,agents,commands}
make codex       # Codex        → .agents/skills, .codex/agents/*.toml
make clean-ai    # remove everything the two targets above created
```

Both targets are idempotent — rerun after a `git pull` that adds an asset. `make clean-ai`
only removes what it created; unrelated content in `.claude/` or `.codex/` (your settings,
session state) is left alone.

**Assets are symlinked, not copied.** A `git pull` then updates them in place, and a fix you
make mid-session lands in a git-tracked file instead of a private fork that gets lost. The one
exception is Codex subagents: Codex wants standalone TOML with a `developer_instructions` key
rather than Markdown + frontmatter, so `make codex` *generates*
`.codex/agents/<name>.toml` from `skills_studio/agents/<name>.md` via
`skills_studio/tools/md_agent_to_codex_toml.py`. Edit the Markdown and rerun `make codex`;
never edit the generated TOML.

Where each assistant looks, for reference:

| Asset | Claude Code | Codex |
|---|---|---|
| Skills | `.claude/skills/` | `.agents/skills/` |
| Subagents | `.claude/agents/` (`*.md`) | `.codex/agents/` (`*.toml`) |
| Slash commands | `.claude/commands/` | no project-level equivalent — use the skill |

Project scope is deliberate: these assets encode *this* repo's workflows and shouldn't fire in
unrelated sessions. If you do want one everywhere, symlink that single asset into `~/.claude/`
or `~/.agents/skills/` yourself — and only if it's genuinely repo-independent.

If a target reports `SKIP ... is not a symlink`, you already have a real directory there;
link the individual assets into it instead:

```bash
ln -sfn "$PWD/skills_studio/skills/vllm-qaic-rebase" .claude/skills/vllm-qaic-rebase
```

Verify by starting a session in the repo root and checking the asset is offered. One that
doesn't appear almost always has malformed frontmatter, or a `name:` that doesn't match its
file/directory name.

### Contributing a new agent/skill/command

- Add it under `skills_studio/<type>/`, never straight into `.claude/` or `.codex/` — those are
  gitignored, so anything created there reaches nobody else.
- Required frontmatter: `name` (kebab-case, matching the file/directory name) and a
  `description` that says **when to use it**, not just what it is. That line is all the model
  sees when deciding whether to invoke the asset, so lead with the trigger conditions.
- Keep it portable across assistants: Markdown + frontmatter, no host-specific tool names in
  the instructions where a plain description works. That's what lets one source file serve both
  `make claude` and `make codex`.
- Scope it to this repo's real workflows. Generic Python or vLLM advice belongs in the model,
  not in a skill.
- State provenance for anything derived from external data (review history, CI logs, a doc), as
  `agents/vllm-qaic-code-review.md` does — a reader needs to know whether a rule was mined from
  evidence or invented.
- Keep instructions in the asset, not in a prompt the caller has to remember; keep commands thin
  — orchestration in the command, judgment in the agent or skill it spawns.
- Lint before committing: `pre-commit run --files skills_studio/...`.

## Lint and static checks

```bash
pre-commit run --all-files                  # everything the pre-commit CI gate runs
pre-commit run --all-files --hook-stage manual   # + mypy 3.11/3.12, CI-only hooks
pre-commit run ruff-check --files <paths>   # fast loop on your diff
```

Never bypass with `--no-verify`. To skip one hook with a reason, `SKIP=<hook-id> git commit -s`.

## Running tests

Unit tests need no hardware and should be the fast inner loop:

```bash
pytest tests/unit -q
```

E2E tests need real QAIC devices. Always pass the device pool explicitly:

```bash
export QAIC_VISIBLE_DEVICES=0
pytest tests/e2e/test_qaic_generate.py --device-id 0 -q
pytest tests/e2e -k "not lora" --device-id 0,1 -q
```

Full CI sweep (job collection + device-aware scheduling — do **not** use `pytest -n`, it races
on compile caches and double-books devices):

```bash
bash -x ci_scripts/ci_fasttest_qaic_aot.sh --device-ids 0,1,2,3 --timeout 1800
```

Whole pipeline as CI runs it (fresh venv, install, then the fasttest sweep):

```bash
bash -x ci_scripts/ci_pipeline_run_plugin.sh \
  --work-dir <dir> --hf-home <dir> --qeff-home <dir>
```

Device IDs resolve as `--device-ids` → `$DEVICE_IDS` → auto-discovered idle devices. Cache
`HF_HOME`/`QEFF_HOME` on a roomy filesystem; AOT compiles are expensive.

> Only `ci_fasttest_qaic_aot.sh` exists today; the PYT equivalent is still a TODO in
> `ci_pipeline_run_plugin.sh`. Run PYT e2e tests with `pytest` directly.

## Adding tests

- **Put it at the cheapest level that can catch the bug.** Pure logic (shapes, block/KV index
  math, config validation, arg plumbing) belongs in `tests/unit` with no device and no compile.
  Reach for `tests/e2e` only when the behavior genuinely requires a live model on hardware.
- **Assert behavior, not implementation.** Check generated output, raised errors, and observable
  state — not that a private helper was called.
- **Use the fixtures.** `qaic_model` (offline `LLM`) and `server_runner` (real
  OpenAI-compatible server subprocess) from `tests/e2e/conftest.py`. Configure via the
  `@pytest.mark.qaic_test_config(...)` marker (`model_name`, `seq_len`, `ctx_len`, `decode_bsz`,
  `dtype`, `kv_dtype`, `num_device_groups`/`device_group_size`, LoRA and disagg keys).
- **`qaic_model` is class-scoped — group tests that share a config into one class** so the
  expensive compile happens once. Bare module-level test functions each with their own marker
  force a recompile per test.
- **Gate mode/package-specific tests with markers**, not ad-hoc runtime checks:
  `qaic_aot_mode`, `qaic_disagg_installed`. Register any new marker in `pyproject.toml`.
- **Declare device needs in the marker.** `collect_jobs.py` reads the same kwargs to size the
  scheduler job; an unset `num_device_groups`/`device_group_size` means a silently wrong
  allocation in CI.
- **Deterministic by default.** Fixed prompts, `temperature=0` (or a seed) for output
  assertions; no network beyond the HF cache, no sleeps as synchronization, no cross-test order
  dependencies. Tolerance-based accuracy checks need a justified bound in a comment.
- **A bug fix starts with a test that fails before the fix.** Say so in the PR description.
- Long-running or resource-heavy additions: state the runtime and device count in the PR — the
  fasttest sweep is meant to stay fast.
