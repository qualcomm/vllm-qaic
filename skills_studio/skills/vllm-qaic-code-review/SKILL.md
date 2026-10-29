---
name: vllm-qaic-code-review
description: Review changes to the vllm-qaic plugin (qualcomm/vllm-qaic) as this repo's own reviewers would. Use whenever the user asks to review, audit, or sanity-check a vllm-qaic change — phrases like "review PR 212", "review my working tree", "look at this fork branch", "is this ready to merge", "check this diff for leaks", or "does this break eager mode". Accepts the working tree, a PR number, a commit or range, a path, or a fork branch URL, plus `--runs=2` to review twice and vote. Routes the diff to AoT / PyT / shared rule overlays, runs a mechanical grep pre-pass, anchors every finding to a literal diff line, and surfaces public-repo leaks and cross-mode hazards first. NOT for upstream vLLM PRs, QEfficient, or generic Python review — the rules encode contracts specific to this plugin.
---

# vllm-qaic code review

You review changes against `qualcomm/vllm-qaic` for the hazards this repo's
reviewers actually raise: upstream edits instead of patches, a change made for
one mode that breaks the other, config handling that misreads CLI/env strings,
version pins that drift between files, and — because the repo is public —
machine-specific paths and internal references that would stay in git history
forever.

Args: `$ARGUMENTS`

## Path conventions

- `references/...` and `scripts/...` are relative to **this skill's own
  directory**. Where a path must go into a subagent prompt (subagents don't
  inherit this skill's cwd), substitute `<SKILL_DIR>` with this skill's
  absolute directory first.
- `<REPO_ROOT>` is the checkout the diff came from: `git rev-parse
  --show-toplevel` for a local or PR diff, or the Phase 1a clone path for a
  fork-branch review.

References (role agents load the ones their prompt names):

- `references/repo_conventions.md` — the contract: plugin-not-fork, the two
  modes and the import guard idiom, where each kind of change belongs, the
  patch-module conventions, headers, what pre-commit already enforces, version
  pin sources, test layout.
- `references/rules_shared.md` — 39 rules, every diff.
- `references/rules_aot.md` — 8 rules, AoT overlay.
- `references/rules_pyt.md` — 13 rules, PyT (eager) overlay.
- `scripts/run_review_greps.sh` — mechanical pre-pass, run in Phase 3.

The worker contract — anchoring gate, carve-outs, YAML schema — lives in the
`vllm-qaic-code-review` agent (`skills_studio/agents/vllm-qaic-code-review.md`;
activate it with `make claude` or `make codex` from the repo root). Rules 1–34
of the shared set and all AoT/PyT rules were mined from this repo's human
review history; shared rules 35–39 restate ground rules from
`skills_studio/AGENTS.md`.

## Workflow (6 phases)

### Phase 1 — Capture the diff

Pick the first form that applies:

| Arg form | Command |
|---|---|
| empty | `git --no-pager diff --no-color HEAD` |
| `<N>` (digits) | `gh pr diff <N> --repo qualcomm/vllm-qaic`, plus `gh pr view <N> --repo qualcomm/vllm-qaic --json title,body,baseRefName,headRefName,isDraft` |
| a GitHub branch link — `https://github.com/<owner>/<repo>/tree/<branch>`, or a repo URL plus a branch name given in prose | Phase 1a |
| a commit or range (`<sha>`, `<base>..<head>`) | `git --no-pager diff --no-color <range>` |
| a path | `git --no-pager diff --no-color HEAD -- <path>` |

Note `--runs=2` if present (Phase 4). Don't pass `-w`: a whitespace-only rewrite
of upstream-derived code is itself an `upstream.minimize_diff` signal. Never
`gh pr checkout` — it mutates the user's checkout. If `gh` is missing or not
authenticated, fall back to `git fetch origin pull/<N>/head:pr<N>` and
`git diff origin/main...pr<N>`, and say the PR title/body were not available.

If the diff is empty, say so and stop.

**Where the diff goes.** Save it to a temporary file such as
`${TMPDIR:-/tmp}/vqreview/<tag>.diff` — the grep pass reads it, and agents
read it instead of a re-pasted copy. Never write a diff or review file
*inside* a checkout: CI checkouts are read-only, and a stray file dirties
`git status`. If the diff is ≤ ~50 KB, also inline it in each agent prompt;
if larger, give agents the file path.

#### Phase 1a — Cloning a fork + branch

A fixed procedure, run the same way every time:

1. **Parse** `<owner>`, `<repo>`, `<branch>`. A `/tree/<branch>` segment can
   contain `/` (`dev/v0.30.0`) — take everything after `/tree/`.
2. **Location.** `${TMPDIR:-/tmp}/vllm-qaic-review/<owner>-<repo>`, **outside**
   any existing checkout. Never clone into or under the user's own working
   clone. State the path before cloning.
3. **Reuse, don't re-clone.** If that directory is already a git repo, `git
   fetch origin` there instead.
4. **Clone.** `git clone --no-single-branch git@github.com:<owner>/<repo>.git
   <path>`; fall back to `https://github.com/<owner>/<repo>.git` if SSH isn't
   set up.
5. **Checkout.** `git fetch origin <branch>`, then `git checkout <branch>`
   (`git switch -c <branch> origin/<branch>` if no local branch exists).
6. **Diff** against the fork's default branch (`git symbolic-ref
   refs/remotes/origin/HEAD`, normally `main`) with three-dot syntax:
   `git --no-pager diff --no-color origin/<default>...<branch>`.
7. **Sanity-check the range.** Run `git log --oneline origin/<default>...<branch>`.
   If it visibly bundles independent concerns (two version bumps, a rebase
   plus a feature), print the commits and ask which range to review instead of
   picking one.

### Phase 2 — Shape verdict (before reading code)

```bash
awk '/^diff --git /{f=$4; sub(/^b\//,"",f)} /^[+-]/ && !/^(\+\+\+|---) /{c[f]++} END{for (k in c) print c[k], k}' <diff> | sort -rn | head -40
```

One line for the report. These are signals for where to look, not findings —
anything that matters becomes a finding only through a rule in Phase 4.

- Any `vllm/` path → `patch.no_direct_upstream_edit` will fire; say so up front.
- Generated or runtime artifacts in the file list (`*.onnx`, QPCs,
  `scheduler_output/`, `*.log`, `__pycache__/`) → `hygiene.no_committed_artifacts`.
- A version bump bundled with feature work, or a rebase bundled with anything
  → say "split this". A bump diff that also deletes upstream code is
  `upstream.no_deletion_on_bump` territory.
- A new `vllm_qaic/patch/patch_*.py` → the patch rules (2–8) carry the review;
  read the escalation order in `repo_conventions.md` before accepting it.
- A change described as single-mode that lands in `platform_base.py`,
  `worker/worker.py`, or `worker/model_runner.py` outside the mode-specific
  classes → expect a cross-mode hazard.
- A PR whose title or draft state says WIP / do-not-merge → state it; the
  verdict can't be better than `CHANGES REQUIRED` while the author says it isn't
  ready.

### Phase 3 — Mechanical grep pass

```bash
bash <SKILL_DIR>/scripts/run_review_greps.sh <diff> <REPO_ROOT> > ${TMPDIR:-/tmp}/vqreview/<tag>.greps
```

The script prints `path:line: <added line>` hits under headers that name the
candidate rule IDs, plus the changed-file list, the version-pin values from
every source (with `MISMATCH` / `consistent`), and added lines per file.
Capture the output verbatim; Phase 4 agents get it. A hit is a **lead**, not a
finding — the agent still applies the rule, its carve-outs, and the anchoring
gate. An empty section is evidence the trigger is absent, not proof: rules
like `state.parallel_paths_updated` have no grep and are judged from the diff.

### Phase 4 — Route, then fan out role agents in parallel

**Route.** Classify every changed file:

| Path pattern | Bucket |
|---|---|
| `QaicWorkerAoT` / `QaicModelRunnerAoT` hunks; `vllm_qaic/model_loader/qaic.py`, `qaic_session_np.py`, `spec_decode/**`, `distributed/kv_transfer/**`, `patch/patch_graph_pickler.py`, `patch/patch_rejection_sampler.py` | **aot** |
| `QaicWorkerPyt` / `QaicModelRunnerPyt` hunks; `vllm_qaic/ops/**`, `vllm_qaic/_custom_ops.py` | **pyt** |
| `vllm_qaic/platform_base.py`, `platform.py`, `envs.py`, other `patch/**`, `utils/**`, `attention/**`, `executor/**`, `quantization/**`, base `QaicWorker` hunks | **shared** |
| `vllm/**` (upstream) | **shared** |
| `tests/**`, `examples/**`, `docs/**`, `ci_scripts/**`, `setup.py`, `requirements/**`, `scripts/**`, `docker/**` | **shared** (plus **pyt** under an `eager/` test path) |
| `skills_studio/**`, `*.md`, `Makefile`, dotfiles | **shared** — no mode signal, but publicly visible |

Then pick roles:

- Only **aot** paths → `shared` + `aot`.
- Only **pyt** paths → `shared` + `pyt`.
- Both, **or** the grep pass shows a mode branch (`is_aot`, `enforce_eager`)
  added in shared code → `shared` + `aot` + `pyt`, and say *"Diff straddles
  shared code; reviewing both modes."*
- Only docs, assistant assets, or other no-mode-signal files → `shared` alone.
  There is no mode question to ask; the `leak.*` rules still run.

If routing is genuinely ambiguous — a new code file whose mode you can't infer,
or a shared hunk where you can't tell which mode the changed branch serves —
don't guess: ask the user, listing the files, with the options review as AoT /
review as PyT / review both. Print the routing decision before fanning out.

**Fan out.** Spawn every selected role in **one** message so they run
concurrently, `subagent_type: "vllm-qaic-code-review"`. Each prompt:

```text
[ROLE] <shared|aot|pyt>
[RULE CHECKLIST] <SKILL_DIR>/references/rules_<role>.md — read it; attest every row in order.
[CONVENTIONS] <SKILL_DIR>/references/repo_conventions.md — read it before judging.
[REPO CHECKOUT PATH] <REPO_ROOT>
[GREP LEADS]
<the Phase 3 output, verbatim>
[DIFF TO REVIEW]
<the diff inline, or its file path>
```

`[REPO CHECKOUT PATH]` lets the build rules read version-pin files the diff
didn't touch, and lets every role check a claimed convention against the tree.

Default is one run per role. Under `--runs=2`, spawn each role twice on the
identical diff.

**Precision contract** (the agent contract enforces it; restate it if you
review in the main thread). A review is judged on finding every real issue
*and* never shipping a junk comment — a false positive teaches the author to
ignore the rest. A `fail` ships only when it has a `quoted_line` that is a
literal substring of the diff **and** a concrete consequence: the mode where
the plugin crashes on import, the config string that is misread, the upstream
symbol that is silently replaced, the file that disagrees on a pin, the path
that won't exist on another machine. Never fail on: a rule outside the role's
file; style that ruff or the other pre-commit hooks already enforce
(`repo_conventions.md#what-pre-commit-already-enforces-dont-re-review`); the
plugin's deliberate mirroring of upstream names and structure; the intended
AoT/PyT capability asymmetry; or a convention you haven't confirmed in the
tree. One root cause, one finding.

### Phase 5 — Merge

One more agent, `subagent_type: "vllm-qaic-code-review"`, `[ROLE] merger`, with
every role's YAML plus `[FULL DIFF]`, and these six steps pasted into its
prompt. The merger:

1. **Re-verifies every anchor** — drops any `fail` whose `quoted_line` is not
   a literal substring of the diff. Hard gate.
2. **Orders `leak.*` first, then `affects_other_mode: true`.** A leak is the
   one class a follow-up commit can't fix, because the value stays in history.
   Cross-mode findings are the bug class a single-mode reviewer misses and the
   reason both overlays exist.
3. **Dedupes** across roles by `(file, line, rule)`, and keeps one finding
   when `mode.lazy_package_import` and `pyt.forbidden_import` hit the same line.
4. **Under `--runs=2` only**, unions the runs by `(file, line, rule)` and
   attaches `votes: k/2`. Votes annotate confidence; they don't gate the verdict.
5. **Tallies coverage** per role from the `pass` / `fail` / `na` counts.
6. **Aggregates the verdict:**

| Findings | Verdict |
|---|---|
| Any `BLOCKING` | `CHANGES REQUIRED` |
| Only `non-blocking` | `APPROVED WITH COMMENTS` |
| None | `APPROVED` |

### Phase 6 — Output

Use exactly this. Keep the Leaks and Cross-mode sections even when empty
(write `none`); omit an empty BLOCKING or Non-blocking section.

```text
Shape: N files, +A / -D. <one-line shape verdict>
Routing: <files> → roles <shared, aot, pyt>
Coverage: shared 39/39 (2 failed, 30 n/a) · aot 8/8 (1 failed, 5 n/a)

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
<CHANGES REQUIRED | APPROVED WITH COMMENTS | APPROVED>. <one-line reason>
```

Always print the coverage line — a silently skipped rule has to be visible.

## Finding form (the bar)

Every finding names the file and patched-file line, the rule ID, one sentence
on what goes wrong, the quoted diff line, and a one-sentence fix. A leak
finding names the *class* of leak and where it is — never the value — so the
review is itself safe to paste into the public PR.

Example:

```text
- `vllm_qaic/worker/model_runner.py:41` **mode.lazy_package_import** — module-scope QEfficient import in a file both modes load; plugin import fails in an eager-only environment.
  > from QEfficient import QEFFAutoModelForCausalLM
  Fix: move the import into the `QaicModelRunnerAoT` path, or wrap it in `try/except ImportError` like `platform_base.py`.
```

"This could be cleaner" with no rule and no consequence is not a finding.

## Pitfalls

- **The diff is ground truth, not the description.** A PR body that says
  "AoT only" doesn't make a shared-code hunk AoT only.
- **Read both sides of the diff.** Removing an `is_aot` guard, a `timeout=`, or
  a patch's mode gate is as dangerous as adding the wrong one.
- **Check the tree before citing it.** Symbols move; `repo_conventions.md`
  anchors to symbols, not line numbers, and the tree wins on disagreement.
- **Push back on bundled scope.** A version bump plus a feature, or a rebase
  plus a refactor, is unreviewable as one diff — say "split this".
- **Don't re-review pre-commit.** Ruff, typos, markdownlint, shellcheck, mypy,
  the Python license header, and the forbidden-import hooks fail CI by
  themselves.
- **Don't redo agent work in the main thread.** Cite the merger's findings;
  don't re-derive them.

## Iteration

If the user disputes a finding, re-read the cited line and the rule before
retracting or standing by it. If new commits land, re-run Phases 1–5 on the new
diff and say which findings were resolved, which remain, and which are new.

## When you're done

Print the review to chat. Saving it to a file needs the user's go-ahead (a
temporary directory is fine; the repo is not). Posting it to the PR — `gh pr
review`, `gh pr comment` — is a public action and needs explicit confirmation.

Then offer to apply fixes. On yes, make one precise edit per finding and print
one confirmation line each, leaving the changes uncommitted. Never fix and
report in the same breath — the user sees the review before anything changes.
