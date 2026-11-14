---
name: vllm-qaic-release-notes
description: Draft release notes for a vllm-qaic release. Use when the user asks to prepare, generate, or update release notes for a vllm-qaic tag or release branch, or to add a section to docs/docs/community/release_notes.md. Collects the commits and version-pin changes between two refs with a script, then drafts a section in the repo's established format with every claim tied to a PR number. Writes only the working tree; never commits, tags, or publishes a GitHub release.
---

# vLLM-QAIC Release Notes

## Purpose

Produce a reviewable draft of one release-notes section, in the same format as the
existing sections of `docs/docs/community/release_notes.md`, for a human to edit and
publish.

Facts come from a script, not from memory. Prose comes from the PR titles only.

## When To Use

- The user wants release notes for a new vllm-qaic tag or `release/*` branch.
- The user wants the Version Compatibility table or the "Changes since" list refreshed.
- Not for publishing a GitHub release, tagging, or committing. Stop at the working tree.

## Inputs To Settle First

Ask the user for anything not stated. Do not guess.

1. **New ref**: the tag or branch being released (for example `release/r1.23.4`).
2. **Previous ref**: the tag the changelog is measured from (for example `v1.23.0`).
3. **Release name**: the heading, normally `vX.Y.Z (vLLM A.B.C)` so the vLLM version is
   in the name. Take the vLLM version from `VLLM_VERSION` in the script output.
4. **Earlier sections**: check whether `release_notes.md` already has a section for a
   release that this one includes (for example the same vLLM version on an older
   tag). If so, ask whether to add an "includes everything in vX below" line. Do not
   merge sections without being asked.

## Workflow

1. Make sure both refs exist locally (`git fetch` the remote that holds them if not).
2. Run the collector and read all of its output:

   ```
   python3 -I skills_studio/skills/vllm-qaic-release-notes/scripts/release_context.py \
       <repo-path> <previous-ref> <new-ref>
   ```

   It prints the commits (PR, author, handle, subject), the version pins at both refs
   with a Changed column, and the raw +/- lines of the pin files.
3. Resolve GitHub handles. The script fills the handle column from the commit's
   `users.noreply.github.com` email, then from `references/handles.md`, else `unknown`.
   - For any `unknown`, and if `gh` is available, take the PR author from
     `gh api repos/<owner>/<repo>/pulls/<n>` or from the generate-notes API
     (`POST repos/<owner>/<repo>/releases/generate-notes`).
   - If a handle is still unknown, write the PR line as `by <git author name>` with no
     `@`, and list those authors in your report.
   - Add every newly confirmed author to `references/handles.md` (only with the user's
     go-ahead, since it is a file outside `release_notes.md`) and mention rows whose
     Source is `inferred`.
   - Never invent a handle.
4. Draft the section (format below) and insert it **above** the previous release's
   section in `docs/docs/community/release_notes.md`.
5. Run the checks below, fix what fails, then report.

## Output Format

Follow the existing sections. Use these headings in this order and omit any that are
empty:

```
## <name>

<one-paragraph intro: what kind of release, what it builds on, SDK/QEfficient it targets>

### New Features
### Bug Fixes
### PyTorch (JIT) Mode
### Build and Dependencies

### Version Compatibility
<table + "Changes since <previous>:" list>

### Pull Requests
- <title> by @<handle> in https://github.com/<owner>/<repo>/pull/<n>

**Full Changelog**: https://github.com/<owner>/<repo>/compare/<previous>...<new>

### Contributors
```

Rules for the bullets:

- One bullet per user-visible change, ending in `(#PR)`. A bullet may cite several PRs
  when they are one change.
- Section by what the PR title says, not by guess. If a title fits two sections, use
  the first one it names and say in your report which PRs had two candidates. A title that does not clearly say
  which section it belongs to goes under "Build and Dependencies" if it is
  tooling or packaging, otherwise list it in your report as unclassified. Do not
  invent a description of its effect.
- Reword a title only to make it read as a sentence. Dropping a leading tag such as
  `[PyT]`, `[AOT]`, `docs:` or `scripts:` is allowed. Do not add behaviour, impact,
  or motivation that the title does not state.
- A title that says fix, patch, prevent, stop, or avoid describes a bug fix: put it
  under Bug Fixes even when it does not say what was wrong. Only titles that say
  nothing about the kind of change (for example "Remove X", "Increase Y") are
  unclassified.
- PRs that only touch CI, formatting, or tests are listed in the Pull Requests list
  and may be left out of the grouped bullets.
- The Version Compatibility table has six columns: vllm-qaic release name, vLLM,
  Apps SDK, QEfficient, torch (AOT), torch (PYT). Fill them from the script's pin
  table (`VLLM_VERSION`, Apps SDK, `QEFF_BRANCH`, `TORCH_VERSION_AOT`,
  `TORCH_VERSION_PYT`). Write vLLM with a leading `v`, as the existing tables do.
  Never fill a cell from memory.
- The six columns are only the table. The list under it is separate: paste the
  script's "Ready-made block" (the `Changes since` list) verbatim. It covers every
  changed pin the script tracks, including pins that have no table column
  (torchvision, `mistral_common`, `requires-python`, the `spd` extra, the SDK path).
  Do not shorten or filter it. Do not add an "Unchanged" line or section.
- The Apps SDK value comes from `get_qaic_sdk_version()` in `setup.py`, which is an
  exact string and not a minimum. Say so in your report if the previous table used
  ">=".
- Contributors are the distinct commit authors, written with the display names the
  script prints from `references/handles.md`, or the git author name when the script
  marks one as not in the map.
- The compare link uses the refs from the script's "Compare URL refs" line, which has
  any remote prefix (`upstream/`, `origin/`) removed so the link resolves on GitHub.
- The Pull Requests list is oldest first. For a commit that cites two PR numbers
  (a backport), link the last number.
- Leave a Known Issues section out unless the user gives you the content.

## Checks Before Reporting

- Every `#N` in the grouped bullets appears in the script's commit table.
- Every PR in the commit table appears in the Pull Requests list exactly once.
- Every version in the table matches the script output.
- Every row marked `yes` in the script's pin table appears in "Changes since". Count
  them: the number of list items must equal the number of `yes` rows.
- The heading uses the release name and the compare link uses the script's
  "Compare URL refs" line (no `upstream/` or `origin/` prefix), and the markdown renders
  (blank line before each list, no stray indentation).
- `git status` shows `docs/docs/community/release_notes.md` modified and no other tracked file changed (apart from `references/handles.md` if step 3 updated it).

## Reporting

Say the changes are uncommitted. List separately: handles you inferred, PRs you could
not classify, any ">=" vs exact-version question for the Apps SDK, and anything in the
script output you did not use.

## Hard Rules

- Do not run `git commit`, `git tag`, `git push`, or `gh release create`.
- Do not edit files other than `release_notes.md` (and `references/handles.md`, only as
  step 3 describes).
- Do not add a claim, version, or handle that is not in the script output, the PR
  titles, or an earlier section of the notes.
