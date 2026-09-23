---
name: vllm-qaic-rebase
description: Rebase the vllm-qaic plugin (and the paired disaggregated-serving repo) from one upstream vLLM release to a newer one. Use when the user asks to move vllm-qaic to a new vLLM version, bump the vLLM pin, or fix breakage after a vLLM upgrade. Keeps local changes minimal, adapts the plugin's monkeypatches and worker subclasses to upstream API changes, and validates with the offline example plus the AOT/PYT fast tests.
---

# vLLM-QAIC Rebase

## Purpose

Move the vllm-qaic plugin from the vLLM version it currently targets to a newer
one, with as few local changes as possible.

The plugin sits on top of upstream vLLM. It subclasses and monkeypatches parts of
vLLM to run on QAIC hardware. When vLLM changes between releases, those hooks can
break. This skill is the step-by-step way to find and fix that breakage.

Goal for every rebase:

- the version pins point to the new release
- the plugin's patches and subclasses still match upstream
- the offline example runs end-to-end
- the AOT (and PYT) fast tests pass

## When To Use

- The user wants vllm-qaic to target a newer vLLM release.
- A vLLM upgrade broke install, import, the example, or the tests.
- Not for adding features or models unrelated to a version bump.

## Environment

- The environment is user-provided. Do not create it.
- The user creates a conda env (usually a separate one for AOT and for PYT),
  then the plugin install script fills it in.
- Ask the user for the env and how to activate it before running any
  python/pip/pytest command.
- Never claim success from install or import alone. The example must run and the
  fast tests must pass (or every non-pass must be explained).

## Workflow

Do the steps in order. Commit in small, single-purpose, signed-off commits.

### Step 1 - Bump versions, dependencies, and docs

- Update the vLLM version everywhere it appears: package version, the build /
  source pin, and any version constant in the install scripts.
- Diff upstream vLLM's main requirements file (old release vs new) and port the
  changes into the plugin's dependency files. Keep any version deliberately held
  back locally (for example, a dtype/quant library pinned to avoid a torch
  upgrade), and keep its comment.
- Keep the lint / pre-commit pin in sync with upstream, and check that shared
  pre-commit hook versions still match.
- Update the docs: replace old vLLM (and torch, if changed) version numbers in
  the docs folder. This is easy to forget.

### Step 2 - Adapt the plugin to upstream API changes

This is the main work. Compare each place the plugin hooks into vLLM against the
new release, and fix what drifted. Look at every module the plugin subclasses,
monkeypatches, or imports a symbol from. Common kinds of drift:

- A base class the plugin subclasses gains or renames an argument. Mirror it in
  the QAIC subclass.
- An internal method the plugin unpacks returns more or fewer values. Update
  every place that unpacks it.
- A symbol the plugin imports moved to a new module (or a module became a
  package). Update the import; re-exported names usually still work.
- The new release adds a check against a type/enum/Literal that the plugin only
  extended on its config class. Also rebind the module-level symbol the check
  reads, in every module that imported it directly. Extend a Literal as one
  flattened Literal (not a union), so CLI choice lists show all values.
- A code path assumes a device backend (for example Triton kernels) that is not
  active on QAIC CPU. Skip it when unavailable, or add a plain fallback.
- A compatibility shim must run before the module that needs it is imported.
  Apply such shims at plugin startup, not lazily.

### Step 3 - Handle the transformers / QEfficient dependency

The model-compilation layer (QEfficient) pins one exact transformers version
and can lag behind upstream renames. The plugin's looser floor may let pip pull
a newer, incompatible transformers, which then breaks at import or compile time.

- Install first, then check the transformers version actually resolved.
- If the compilation layer needs changes that are only on an unmerged branch,
  install transformers at the version that layer supports, at the environment
  level. Do not commit a pin to an unmerged branch.
- Guard framework imports with a fallback import (try the new name, fall back to
  the old) instead of exact version checks.

### Step 4 - Validate

Run these in order, fixing as you go:

1. Reinstall the plugin. If the build reads its version from git, make sure the
   release tag format is one the build tool can parse.
2. Import-smoke every patch, worker, model-loader, and ops module. This catches
   moved or renamed imports quickly.
3. Run the offline example end-to-end on a free QAIC device.
4. Run the AOT fast tests through the CI harness (it sets up the device pool).
5. Run the PYT fast tests the same way, after the AOT ones.
When a test fails, fix the narrowest scope first, re-run just that, then widen.

Finding free devices and reading SDK/torch versions are QAIC basics; use the
cloud-ai skills for those details.

### Step 5 - Update the paired disagg repo

Disaggregated serving lives in a separate repo that tracks the plugin's version.

- Apply the same kind of import fixes there (symbols vLLM moved or renamed).
- Bump its version to match the plugin, and keep its lint / pre-commit pin in
  sync.
- It is a separate review track, so its change is committed and reviewed on its
  own, not as part of the plugin PR.

### Step 6 - Prepare the PR handoff

- Write a clear PR description grouped by what changed: version bumps, API-drift
  fixes, dependency changes, test/CI changes, docs.
- Save it to a file so the user can copy it.
- Do not open the PR. Hand off the description and the validation evidence for
  the user to submit.

## If Something Fails - Quick Reference

- Build fails on the version -> the release tag format does not match what the
  build tool expects.
- Import error when the engine starts -> a vLLM symbol moved or was renamed;
  search the plugin (and the disagg repo) for the old import path.
- Assertion on a QAIC-specific config value -> a new vLLM check reads a
  module-level symbol the plugin only extended on a class.
- CLI rejects a valid choice -> the extended type is a union; make it one
  flattened Literal instead.
- Test fails on changed behavior -> update the test to the new contract. If one
  case out of many fails, check whether the test uses random inputs without a
  seed (a flake) before treating it as a real break.
- Spec-decode / kernel test error about inspecting a kernel -> the CPU backend
  it needs is not installed; skip when unavailable, or install the backend.
- Device activation / resource error -> the hardware is busy, not a code bug;
  retry on free devices.
- Compatibility shim not taking effect (e.g. a torch-version stub) -> it ran too
  late; apply it at plugin startup, before the module that needs it is imported.
- Subclass init fails with an unexpected keyword argument -> a vLLM base class
  added a parameter; accept and forward it in the QAIC subclass.
- Unpack error (too many / too few values) -> a vLLM internal method changed how
  many values it returns; fix every unpack site.
- Missing-attribute error on the model runner -> a base attribute became a
  module-level constant upstream; set it explicitly in the QAIC subclass.
- Framework class import fails (e.g. a renamed model class) -> add a fallback
  import; if the compile layer needs a specific transformers version, pin it at
  the environment level.
- torch_qaic import fails with an undefined-symbol error -> the installed torch
  does not match what the SDK's torch_qaic was built against; align the torch
  version to the SDK's expectation (see cloud-ai skills).
- Streaming/output test fails on one random case -> likely an unseeded
  random-input flake; confirm by re-running before treating it as a real break.

## Rules To Follow

- Prefer what upstream does. Add a workaround only where QAIC truly needs one.
- Keep workarounds at the edges (config, imports, CLI), not inside model math.
- Prefer checking for a feature over checking a version number.
- Keep each commit small, focused, and signed off. Do not mix unrelated
  cleanups into a rebase commit.
- Do not change comments or docstrings unless the change needs it.
- Remove any temporary workaround once the proper fix is in; note when it can go.

## Done When

- The offline example runs end-to-end.
- The AOT and PYT fast tests pass, or every non-pass is explained (skipped by
  design, hardware busy, or a confirmed flake).
- Version pins, dependencies, and docs all point to the new release.
- The paired disagg repo change is prepared.
- A PR description is written and handed off (the agent does not open the PR).
