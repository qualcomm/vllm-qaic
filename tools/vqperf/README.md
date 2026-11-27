<!--
Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
SPDX-License-Identifier: BSD-3-Clause-Clear
-->

# vqperf

Commit-to-commit serving performance comparison for vllm-qaic.

Given a base and a new vllm-qaic commit, vqperf builds one conda environment
per commit, runs upstream vLLM's serving benchmark against each, and writes a
report that says which metrics moved and whether the movement is larger than
the noise.

## Requirements

- A host with QAIC devices that no other job is using. vqperf locks the
  devices it uses, but it cannot see processes started outside vqperf.
- `micromamba`, `mamba` or `conda` on `PATH`.
- `git`, with access to the repository given by `--fork` (default
  `https://github.com/qualcomm/vllm-qaic.git`).
- Python 3.10+ to run vqperf itself. It uses only the standard library and
  needs no install.
- Disk space for two full vllm-qaic environments plus model weights, under
  `--workdir`.

## Quick start

All commands run from the repository root. `vqperf.sh` adds `tools/vqperf`
to `sys.path`, so there is nothing to install.

```bash
# 1. Check the CSV and see the exact server/client commands.
./tools/vqperf/vqperf.sh validate \
    --configs tools/vqperf/configs/example.csv --mode pyt

# 2. Print the plan without building anything.
./tools/vqperf/vqperf.sh run \
    --base-sha 7cb4f35 --new-sha 4c7bc66 \
    --configs tools/vqperf/configs/example.csv \
    --mode pyt --dry-run

# 3. Run it. Use 3+ repeats; with fewer there is no noise estimate.
./tools/vqperf/vqperf.sh run \
    --base-sha 7cb4f35 --new-sha 4c7bc66 \
    --configs tools/vqperf/configs/example.csv \
    --mode pyt --repeats 3 --devices 0 \
    --reuse-env --keep-going \
    --out /tmp/vqperf/run1

# 4. Read the result.
less /tmp/vqperf/run1/summary.md
```

`--base-sha` and `--new-sha` accept anything `git rev-parse` resolves in
`--fork`: a full or short SHA, a branch, or a tag. If the fork doesn't have
the ref, vqperf falls back to `qualcomm/vllm-qaic`. Results are always
labeled with the resolved 40-character SHA, never with a moving ref like
`main`.

The example CSV has a `profile_mode=torch` row, so it only validates with
`--mode pyt`. Profiling is PyT-only; see [Profiling](#profiling).

## Commands

| Command | Purpose |
|---------|---------|
| `vqperf run` | Build both environments, benchmark every cell, write the report. |
| `vqperf validate` | Check a config CSV and print the resolved `vllm serve` / `vllm bench serve` commands. Needs no hardware or network. |
| `vqperf report` | Rebuild every report file from an existing `--out` directory, e.g. with a different `--threshold-pct`. Needs no hardware. |
| `vqperf clean` | Remove cached checkouts and environments from `--workdir`. |

Each command lists its flags with `--help`
(`./tools/vqperf/vqperf.sh run --help`).

### `run` options

#### What to compare

| Flag | Default | Meaning |
|------|---------|---------|
| `--base-sha` | required | Baseline commit (SHA, branch or tag). |
| `--new-sha` | required | Candidate commit. |
| `--fork` | `qualcomm/vllm-qaic` | Where to fetch commits: `owner/repo`, a git URL, or a local path. |
| `--configs` | required | Config CSV; see [Configuration CSV](#configuration-csv). |
| `--mode` | `aot` | `aot` or `pyt`; passed to `scripts/install.sh`. |
| `--python` | `3.12` | Python version for each conda environment. |

#### How to run

| Flag | Default | Meaning |
|------|---------|---------|
| `--repeats` | `1` | Measured runs per (config, commit). Use 3 or more. |
| `--order` | `interleave` | `interleave` alternates base/new for each config and repeat, so slow drift (thermals, host load) affects both commits equally. `grouped` runs every base cell first. |
| `--warmup-requests` | `0` | Unmeasured requests sent to each fresh server before the measured benchmark. |
| `--devices` | `0` | Comma-separated QAIC device IDs. Exported as `QAIC_VISIBLE_DEVICES` and locked for the whole run. |
| `--server-timeout` | `2700` | Seconds to wait for `vllm serve` to report healthy, including compilation. |
| `--bench-timeout` | `3600` | Seconds allowed for one `vllm bench serve` run. |
| `--keep-going` | off | Continue past a failed cell. By default the first failure stops the run; reports are still written. |
| `--dry-run` | off | Validate and print the plan, then exit. |

#### Caching and resuming

| Flag | Default | Meaning |
|------|---------|---------|
| `--workdir` | `./.vqperf` | Holds the git mirror, per-SHA checkouts, conda envs, the shared HF/QEfficient caches, and device locks. Reuse it across runs. |
| `--out` | `./results/<timestamp>-<mode>` | Output directory for logs, raw results and reports. |
| `--reuse-env` | off | Reuse an environment built by an earlier run if its cache key (SHA + mode + Python + hash of `install.sh` and `utility.sh`) still matches. Without this flag, every run rebuilds both environments. |
| `--reuse-results` | off | Skip cells that already have a successful result for the same SHA in `--out`. Use it with the same `--out` to resume an interrupted run. |

#### Reporting and profiling

| Flag | Default | Meaning |
|------|---------|---------|
| `--threshold-pct` | `5.0` | Smallest relative change, in percent, that can count as a `REGRESSION` or `IMPROVEMENT`. |
| `--profile` | `off` | Default `profile_mode` (`off`, `torch`, `lrt`, `both`) for rows that don't set one. PyT only. |
| `--no-profile` | off | Force profiling off, overriding the CSV. |
| `--profile-only` | off | Run only the profiling passes, without measured cells. |
| `--profile-top-ops` | `30` | Number of ops listed in the profile op diff. |
| `-v`, `--verbose` | off | Debug logging. |

`validate` takes `--configs`, `--mode`, `--profile` and `--no-profile`.
`report` takes `--out`, `--threshold-pct` and `--profile-top-ops`. `clean`
takes `--workdir` plus `--all` (remove the whole workdir) or `--sha <sha>`.
`--sha` matches on the first 12 hex characters, so pass at least 12.

## What a run does

1. Resolves both refs to full SHAs through one bare mirror, so both checkouts
   share a single fetch.
2. Creates a conda environment (`conda create -p <prefix>`) per commit and
   runs `scripts/install.sh {aot|pyt}` inside it. The two environments build
   in parallel.
3. Works out how to call the benchmark *in each environment*: `vllm bench
   serve`, then `python -m vllm.entrypoints.cli.main bench serve`, then the
   legacy script. The two commits may pin different vLLM versions, so each
   side is checked separately. It also checks whether that vLLM accepts
   `--profiler-config`.
4. For each cell: starts `vllm serve`, waits for `/health`, optionally sends
   warmup requests, runs the client, stops the server, and reads the result
   JSON.
5. Writes the report files into `--out`.

vqperf manages the server itself because `vllm bench serve` is only a
client. It takes an advisory `flock` on each QAIC device for the whole run.
Devices are exclusive, so cells run one at a time even though the
environments build in parallel. If another vqperf run holds the lock, the run
exits with code 5.

## Configuration CSV

One row per cell. Lines starting with `#` and blank lines are ignored. See
`configs/example.csv`.

**Required:** `config_id` (letters, digits, `.`, `_`, `-`; unique),
`model`, `max_model_len`, `num_prompts`.

**Server columns** (passed to `vllm serve`):

| Column | Default | Flag |
|--------|---------|------|
| `model` | required | positional model argument |
| `max_model_len` | required | `--max-model-len` |
| `max_num_seqs` | `8` | `--max-num-seqs` |
| `quantization` | unset | `--quantization` |
| `kv_cache_dtype` | unset | `--kv-cache-dtype` |
| `tensor_parallel_size` | `1` | `--tensor-parallel-size` |
| `long_prefill_token_threshold` | unset | `--long-prefill-token-threshold` |
| `extra_server_args` | empty | Any other flags, shell-quoted, e.g. `--enable-prefix-caching --block-size 32` |
| `env` | empty | Extra server environment variables as `KEY=VALUE;KEY2=VALUE2` |

**Client columns** (passed to `vllm bench serve`):

| Column | Default | Flag |
|--------|---------|------|
| `num_prompts` | required | `--num-prompts` |
| `dataset_name` | `random` | `--dataset-name` |
| `dataset_path` | unset | `--dataset-path` |
| `random_input_len` | `1024` | `--random-input-len` (random dataset only) |
| `random_output_len` | `128` | `--random-output-len` (random dataset only) |
| `random_range_ratio` | `0.0` | `--random-range-ratio` (random dataset only) |
| `request_rate` | `inf` | `--request-rate` |
| `max_concurrency` | unset | `--max-concurrency` |
| `burstiness` | `1.0` | `--burstiness` |
| `ignore_eos` | `true` | `--ignore-eos` |
| `percentile_metrics` | `ttft,tpot,itl,e2el` | `--percentile-metrics` |
| `metric_percentiles` | `50,90,99` | `--metric-percentiles` |
| `tokenizer` | unset | `--tokenizer` |
| `extra_client_args` | empty | Any other client flags, shell-quoted |

**Profiling columns** (PyT only): `profile_mode` (`off`, `torch`, `lrt`,
`both`), plus the tuning knobs `profile_num_prompts` (default: the smaller of
`num_prompts` and 16), `profile_wait_iterations`, `profile_warmup_iterations`,
`profile_active_iterations`, `profile_delay_iterations`,
`profile_max_iterations`, `profile_with_stack`, `profile_record_shapes`,
`profile_with_memory`, `profile_with_flops`, `profile_detailed_annotation`
and `profile_ignore_frontend`. Each maps to the `--profiler-config.*` field of
the same name.

**Labels:** any column starting with `x_` is copied into the report as a
label and never reaches a command line.

Validation is strict and happens before any environment is built:

- Unknown columns are rejected.
- These are hard errors: a context-window overflow
  (`random_input_len + random_output_len > max_model_len` for the random
  dataset), a duplicate `config_id`, and an `extra_server_args` /
  `extra_client_args` flag that duplicates a typed column.
- Booleans accept `true/false`, `yes/no`, `on/off` and `1/0`.

## Output

```text
<out>/
  summary.md            # start here
  comparison.csv        # one row per (config, metric): base, new, delta, verdict
  comparison_wide.csv   # one row per config, metrics as columns
  per_run.csv           # every measured cell, every metric
  manifest.json         # args, host, resolved SHAs, env versions, cell status
  profile_index.csv     # profiling passes and their trace dirs (if profiling)
  profile_ops.csv       # per-op host times (torch profiling)
  profile_ops_diff.csv  # ops that moved between base and new (torch profiling)
  provision/<sha12>/    # install logs and git info per commit
  raw/<config_id>/<side>-<sha12>/r<N>/   # server.log, bench.log, result.json
  profile/<config_id>/<side>-<sha12>/    # server.log, torch/, lrt/
```

When a cell fails, read its `server.log` and `bench.log` under `raw/` first.

## Reading the report

`summary.md` starts with the two commits and the host, then lists regressions
and improvements. The verdicts are `REGRESSION`, `IMPROVEMENT`, `NEUTRAL`,
`NOISY`, `UNKNOWN` and `INFO`.

The point estimate is the **median** across repeats, not the mean, so one slow
first run doesn't move it. A change counts as a `REGRESSION` or
`IMPROVEMENT` if it is at least the threshold (`--threshold-pct`, default 5%)
and larger than a 2-sigma noise band. A change that passes the threshold but
stays inside the noise band is `NOISY`. The noise band needs **at least 3
repeats on both sides**. With fewer, verdicts use the threshold alone, so
vqperf can't tell a regression from noise. In that case every row carries a
"no noise estimate" warning, and `summary.md` opens with a banner saying so.

Metrics that describe the workload rather than performance (`completed`,
`duration`, token totals) are reported as `INFO` and can never be a
regression. If they differ, the two sides didn't run the same work.

To re-judge an existing run with a different threshold:

```bash
./tools/vqperf/vqperf.sh report --out /tmp/vqperf/run1 --threshold-pct 3
```

## Profiling

`--profile {off,torch,lrt,both}`, or a `profile_mode` column, adds a separate
profiling pass per (config, side) after all measured passes. Profiling skews
timing, so its numbers never feed `comparison.csv`.

**Profiling requires `--mode pyt`.** In AoT mode, `QaicWorkerAoT.profile()`
only logs `profile is not supported with QAIC AoT mode` and returns, so
`/start_profile` answers 200 and produces nothing. The LRT hook
(`optional_qaic_profiling`) exists only in `QaicModelRunnerPyt`. vqperf
rejects AoT profiling when it parses the CSV, so you don't find out from an
empty directory. AoT performance is still compared; only the per-op breakdown
is missing.

- **`torch`** (host profiling) starts the server with `--profiler-config.*`,
  the only way vLLM v0.23+ accepts torch profiler settings. vqperf never sets
  the removed `VLLM_TORCH_PROFILER_DIR`. If an environment's vLLM doesn't
  accept `--profiler-config`, that side's profiling pass is marked
  `unsupported`. The output is `profiler_out_<rank>.txt` op tables plus Chrome
  traces. vqperf parses the tables into `profile_ops.csv` and
  `profile_ops_diff.csv`, which name the ops that moved. Two upstream defaults
  are changed: `with_stack` is off (stack tracing skews timing the most) and
  warmup is 1 iteration.
- **`lrt`** (device tracing) sets `VLLM_TORCH_QAIC_PROFILER_DIR`, the variable
  `QaicModelRunnerPyt` reads via `vllm_qaic/envs.py`. It isn't request-scoped:
  every forward pass from server start writes a trace, so the profiling pass
  sends only `profile_num_prompts` requests (at most 16 by default).

The torch profiler records `CPU` activity only, because
`QaicWorkerPyt.profile()` requests `["CPU"]`. The op tables are therefore
**host-side times**. A regression that is entirely on the device won't show up
in them; use `lrt` for that. The CSVs and summary repeat this caveat.

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Clean run |
| `2` | Bad configuration (CSV or flags) |
| `3` | Provisioning failed (clone, conda, `install.sh`) |
| `4` | A benchmark cell failed |
| `5` | Requested devices are locked by another vqperf run |
| `130` | Interrupted. Partial reports are still written. |

## Tests

```bash
cd tools/vqperf && python3 -m pytest tests -q
```

The tests need only pytest, and none of them touch hardware, the network, or
conda.
