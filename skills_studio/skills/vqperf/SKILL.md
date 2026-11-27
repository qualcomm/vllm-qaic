---
name: vqperf
description: Compare vllm-qaic serving performance between two commits with the vqperf tool in tools/vqperf. Use when the user asks whether a change regressed or improved throughput or latency, wants an A/B perf comparison of two SHAs, branches, or tags, wants to benchmark a PR against main, or wants to re-read or re-judge an existing vqperf result. Builds one conda env per commit, runs vllm serve plus vllm bench serve on QAIC devices, and reports per-metric verdicts against a noise band. Optional PyT-only torch/LRT profiling.
---

# vqperf - Commit-to-Commit Perf Comparison

## Purpose

Answer "did this commit make serving faster or slower?" with numbers that hold
up. vqperf takes a base and a new vllm-qaic commit, builds a separate conda
environment for each one, runs upstream vLLM's serving benchmark against both,
and writes a report. The report lists which metrics moved and whether the
change is larger than the run-to-run noise.

The full reference is `tools/vqperf/README.md`. This skill is the short path
for running the tool and reading its result.

## When To Use

- The user wants to know if a commit, branch, or PR changed serving
  performance.
- The user wants to benchmark one vllm-qaic version against another on the same
  host.
- The user has an existing vqperf `--out` directory and wants it explained or
  re-judged with a different threshold.
- Not for a single-commit benchmark with no baseline. Run `vllm bench serve`
  directly for that.
- Not for accuracy or functional testing.

## Before You Start

Ask the user for anything below that they haven't given:

- **Base and new refs.** Any SHA, branch, or tag that `git rev-parse` resolves
  in `--fork`. Results are labeled with the resolved full SHA.
- **Mode.** `aot` (default) or `pyt`. Profiling needs `pyt`.
- **Configs.** A CSV of model and workload rows. Start from
  `tools/vqperf/configs/example.csv`, or write a new one with the user. The
  required columns are `config_id`, `model`, `max_model_len` and
  `num_prompts`. The README has the full column list.
- **Devices.** The QAIC device IDs to use (`--devices 0` or `--devices 0,1`).
  Use `qaic-util` to confirm they are free (see the cloud-ai skills).
  vqperf locks its devices against other vqperf runs only. It cannot see other
  jobs using them, and a busy device makes the numbers useless.
- **Fork.** Only if the commits are not in `qualcomm/vllm-qaic`. Pass
  `--fork owner/repo`, a git URL, or a local path.

Host requirements: `micromamba`, `mamba` or `conda` on `PATH`, plus `git` and
Python 3.10+. Under `--workdir` (default `./.vqperf`) there must be disk space
for two full environments and the model weights. vqperf is standard-library
only; nothing needs to be installed to run it.

vqperf builds its own environments. Do not ask the user for one and do not
activate one; run it with the host's `python3`.

## Workflow

Run every command from the repository root. `./tools/vqperf/vqperf.sh` is the
entry point.

### Step 1 - Validate the configs

```bash
./tools/vqperf/vqperf.sh validate \
    --configs <configs.csv> --mode <aot|pyt>
```

This needs no hardware or network. It prints the exact `vllm serve` and
`vllm bench serve` command for every row. Fix any error before going further.
Common errors:

- `random_input_len + random_output_len > max_model_len`
- a duplicate `config_id`
- an unknown column
- a flag in `extra_server_args` or `extra_client_args` that duplicates a typed
  column
- a `profile_mode` row with `--mode aot`

### Step 2 - Dry run

```bash
./tools/vqperf/vqperf.sh run \
    --base-sha <base> --new-sha <new> \
    --configs <configs.csv> --mode <aot|pyt> --dry-run
```

Show the user the plan: the resolved SHAs, the cells, and the order.

### Step 3 - Run

```bash
./tools/vqperf/vqperf.sh run \
    --base-sha <base> --new-sha <new> \
    --configs <configs.csv> --mode <aot|pyt> \
    --repeats 3 --devices <ids> \
    --reuse-env --keep-going \
    --out <out-dir>
```

- Always use `--repeats 3` or more. With fewer repeats there is no noise
  estimate, and vqperf cannot tell a regression from noise.
- `--reuse-env` skips the environment build when an earlier run built the
  same SHA with the same mode, Python version and install scripts.
- `--keep-going` finishes the other cells when one fails.
- Pick an explicit `--out` so the result is easy to find and resume.
- A run can take hours. Building the two environments takes tens of minutes,
  and each server start includes model compilation (`--server-timeout`
  defaults to 45 minutes). Run it in the background and check on it; do not
  block on it.
- To resume an interrupted run, repeat the same command with the same `--out`
  and add `--reuse-results`.

### Step 4 - Read the result

Open `<out-dir>/summary.md` first. It names the two commits and the host,
then lists regressions and improvements.

| Verdict | Meaning |
|---------|---------|
| `REGRESSION` / `IMPROVEMENT` | The median moved by at least `--threshold-pct` (default 5%) in the bad / good direction, and by more than the 2-sigma noise band. |
| `NOISY` | The move passed the threshold but is inside the noise band. Not confirmed. Rerun with more repeats, or check for host load. |
| `NEUTRAL` | The move is smaller than the threshold. |
| `UNKNOWN` | The metric is missing on one side (e.g. that side failed), or the base median is 0. |
| `INFO` | Workload metrics (`completed`, `duration`, token totals). Never a regression. If they differ, the two sides did not run the same work. Say so. |

The noise band needs 3 or more repeats on **both** sides. With fewer,
`REGRESSION` and `IMPROVEMENT` are judged on the threshold alone. Every row
then carries a "no noise estimate" warning, and `summary.md` opens with a
banner. In that case, describe these verdicts to the user as unconfirmed, not
as regressions.

When you report back, give the user the regressions and improvements with
their base value, new value and delta. Also say how many repeats ran.

`comparison.csv` holds every (config, metric) row. `per_run.csv` holds every
measured run. `manifest.json` records the exact args, SHAs, host, and the
package versions in each environment.

To re-judge with a different threshold, without hardware:

```bash
./tools/vqperf/vqperf.sh report --out <out-dir> --threshold-pct 3
```

### Step 5 - Profile (optional, PyT only)

Profile only when the user wants to know *why* a metric moved. Add
`--profile torch`, `--profile lrt` or `--profile both`, or set a
`profile_mode` column per row. vqperf runs a separate profiling pass after the
measured passes, so profiling never skews `comparison.csv`. Add
`--profile-only` to skip the measured cells.

- `torch` passes `--profiler-config.*` to `vllm serve`. vLLM 0.23+ accepts
  torch profiler settings only this way, and does not read
  `VLLM_TORCH_PROFILER_DIR`. The output is `profile_ops_diff.csv`, which lists
  the ops that moved. These are **host-side CPU times only**:
  `QaicWorkerPyt.profile()` records CPU activity, so a device-only regression
  will not show up.
- `lrt` sets `VLLM_TORCH_QAIC_PROFILER_DIR` for device traces under
  `profile/<config_id>/<side>-<sha12>/lrt/`. Use it when the change is on the
  device side.
- vqperf rejects profiling in AoT mode at parse time, because the AoT worker
  does not support it.

### Step 6 - Clean up (when asked)

```bash
./tools/vqperf/vqperf.sh clean --workdir <workdir> --sha <sha-12+chars>
./tools/vqperf/vqperf.sh clean --workdir <workdir> --all
```

Environments are large but slow to rebuild. Ask before deleting them.
`--sha` matches the first 12 hex characters, so pass at least 12.

## If Something Fails - Quick Reference

Exit codes: `0` clean, `2` bad config or flags, `3` provisioning failed,
`4` a cell failed, `5` devices locked by another vqperf run, `130`
interrupted. Reports are still written for `4` and `130`.

- Exit `2` -> run `validate` and fix the CSV or flags it names.
- Exit `3` -> read `<out>/provision/<sha12>/` for the clone, conda or
  `install.sh` log. Usual causes are a ref that is not in `--fork`, no conda
  tool on `PATH`, or a dependency that failed to install for that commit.
- Exit `4` -> find the failed cell in `summary.md` or `manifest.json`, then
  read `raw/<config_id>/<side>-<sha12>/r<N>/server.log` first and `bench.log`
  second.
    - A server that never became healthy is usually a compile error, an OOM, a
    busy device, or `--server-timeout` being too short for a large model.
    - If only one side fails, the new commit may have broken that config. That
    is a finding in itself; report it.
- Exit `5` -> another vqperf run holds those devices. Wait for it or pick other
  `--devices`. Do not delete the lock files under `<workdir>`.
- Profiling pass is `unsupported` -> that side's vLLM does not accept
  `--profiler-config`. Its measured cells still count.
- Many `NOISY` verdicts -> the run-to-run spread is wide. Check for other
  jobs on the host and the devices, then rerun with `--repeats 5`.
- `INFO` metrics differ between the sides -> the sides did not do the same
  work, e.g. a different number of completed requests or output tokens.
  Explain this before trusting any latency or throughput verdict.

## Rules To Follow

- Compare like with like. Use the same host, devices, configs and mode for
  both sides. vqperf enforces this within a run, so do not compare across
  separate runs.
- Keep the default `--order interleave`, so slow drift (thermals, host load)
  hits both commits equally.
- Report the median and the verdict that vqperf gives. Do not re-derive
  verdicts from single runs.
- Do not edit files under `--out` by hand. Use `report` to regenerate them.
- Do not commit results, environments or `.vqperf/` to the repository.

## Done When

- `summary.md` exists for the requested commits, with 3 or more repeats per
  side.
- Every failed or unsupported cell is explained from its logs.
- The user has the regressions and improvements, with values and deltas, and
  the `--out` path.
