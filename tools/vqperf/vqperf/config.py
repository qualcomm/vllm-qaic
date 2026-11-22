# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Benchmark-configuration CSV: schema, coercion and validation.

Every rule here exists to fail in milliseconds rather than after a 20-minute
QPC compile. Validation collects *all* errors in a file before raising, so a
broken matrix is fixed in one pass.
"""

from __future__ import annotations

import csv
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .util import VqperfError

CONFIG_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

PROFILE_MODES = ("off", "torch", "lrt", "both")

#: Columns that are only meaningful when profiling is enabled.
PROFILE_TUNING_COLUMNS = (
    "profile_num_prompts",
    "profile_wait_iterations",
    "profile_warmup_iterations",
    "profile_active_iterations",
    "profile_delay_iterations",
    "profile_max_iterations",
    "profile_with_stack",
    "profile_record_shapes",
    "profile_with_memory",
    "profile_with_flops",
    "profile_detailed_annotation",
    "profile_ignore_frontend",
)

KNOWN_COLUMNS = (
    "config_id",
    "model",
    "tokenizer",
    "max_model_len",
    "max_num_seqs",
    "quantization",
    "kv_cache_dtype",
    "tensor_parallel_size",
    "long_prefill_token_threshold",
    "extra_server_args",
    "dataset_name",
    "dataset_path",
    "random_input_len",
    "random_output_len",
    "random_range_ratio",
    "num_prompts",
    "request_rate",
    "max_concurrency",
    "burstiness",
    "ignore_eos",
    "percentile_metrics",
    "metric_percentiles",
    "extra_client_args",
    "env",
    "profile_mode",
    *PROFILE_TUNING_COLUMNS,
)

TRUE_WORDS = {"1", "true", "yes", "y", "on"}
FALSE_WORDS = {"0", "false", "no", "n", "off"}


class ConfigError(VqperfError):
    """One or more problems in the configuration CSV."""


@dataclass
class ProfileSpec:
    """Resolved ``ProfilerConfig`` inputs for one configuration.

    Field defaults mirror ``vllm/config/profiler.py`` except where noted in
    :mod:`vqperf.profiler`.
    """

    mode: str = "off"
    num_prompts: int | None = None
    wait_iterations: int = 0
    warmup_iterations: int = 1
    active_iterations: int = 5
    delay_iterations: int = 0
    max_iterations: int = 0
    with_stack: bool = False
    record_shapes: bool = False
    with_memory: bool = False
    with_flops: bool = False
    detailed_annotation: bool = False
    ignore_frontend: bool | None = None

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def wants_torch(self) -> bool:
        return self.mode in ("torch", "both")

    @property
    def wants_lrt(self) -> bool:
        return self.mode in ("lrt", "both")

    def resolved_ignore_frontend(self) -> bool:
        """Auto-enable ``ignore_frontend`` when a delay/limit window is set.

        Upstream warns that front-end profiling does not track iterations and
        captures the entire range, which would swamp the window the user asked
        for. Explicit CSV values always win.
        """
        if self.ignore_frontend is not None:
            return self.ignore_frontend
        return self.delay_iterations > 0 or self.max_iterations > 0


@dataclass
class BenchConfig:
    """One row of the configuration CSV."""

    config_id: str
    model: str
    max_model_len: int
    num_prompts: int
    tokenizer: str | None = None
    max_num_seqs: int = 8
    quantization: str | None = None
    kv_cache_dtype: str | None = None
    tensor_parallel_size: int = 1
    long_prefill_token_threshold: int | None = None
    extra_server_args: list[str] = field(default_factory=list)
    dataset_name: str = "random"
    dataset_path: str | None = None
    random_input_len: int = 1024
    random_output_len: int = 128
    random_range_ratio: float = 0.0
    request_rate: str = "inf"
    max_concurrency: int | None = None
    burstiness: float = 1.0
    ignore_eos: bool = True
    percentile_metrics: str = "ttft,tpot,itl,e2el"
    metric_percentiles: str = "50,90,99"
    extra_client_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    profile: ProfileSpec = field(default_factory=ProfileSpec)
    labels: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    row_number: int = 0

    # -- argv construction -------------------------------------------------

    def server_args(self, *, include_extras: bool = True) -> list[str]:
        argv = [
            self.model,
            "--max-model-len",
            str(self.max_model_len),
            "--max-num-seqs",
            str(self.max_num_seqs),
        ]
        if self.quantization:
            argv += ["--quantization", self.quantization]
        if self.kv_cache_dtype:
            argv += ["--kv-cache-dtype", self.kv_cache_dtype]
        if self.tensor_parallel_size != 1:
            argv += ["--tensor-parallel-size", str(self.tensor_parallel_size)]
        if self.long_prefill_token_threshold is not None:
            argv += [
                "--long-prefill-token-threshold",
                str(self.long_prefill_token_threshold),
            ]
        return argv + (self.extra_server_args if include_extras else [])

    def client_args(
        self, *, num_prompts: int | None = None, include_extras: bool = True
    ) -> list[str]:
        argv = [
            "--model",
            self.model,
            "--dataset-name",
            self.dataset_name,
            "--num-prompts",
            str(num_prompts if num_prompts is not None else self.num_prompts),
            "--request-rate",
            self.request_rate,
            "--burstiness",
            str(self.burstiness),
            "--percentile-metrics",
            self.percentile_metrics,
            "--metric-percentiles",
            self.metric_percentiles,
        ]
        if self.tokenizer:
            argv += ["--tokenizer", self.tokenizer]
        if self.dataset_path:
            argv += ["--dataset-path", self.dataset_path]
        if self.dataset_name == "random":
            argv += [
                "--random-input-len",
                str(self.random_input_len),
                "--random-output-len",
                str(self.random_output_len),
                "--random-range-ratio",
                str(self.random_range_ratio),
            ]
        if self.max_concurrency is not None:
            argv += ["--max-concurrency", str(self.max_concurrency)]
        if self.ignore_eos:
            argv += ["--ignore-eos"]
        return argv + (self.extra_client_args if include_extras else [])

    @property
    def effective_profile_prompts(self) -> int:
        if self.profile.num_prompts is not None:
            return self.profile.num_prompts
        return min(self.num_prompts, 16)


# ---------------------------------------------------------------------------
# coercion helpers
# ---------------------------------------------------------------------------


def _as_int(raw: str, column: str) -> int:
    try:
        return int(raw)
    except ValueError:
        raise _bad(column, raw, "an integer") from None


def _as_float(raw: str, column: str) -> float:
    try:
        return float(raw)
    except ValueError:
        raise _bad(column, raw, "a number") from None


def _as_bool(raw: str, column: str) -> bool:
    low = raw.strip().lower()
    if low in TRUE_WORDS:
        return True
    if low in FALSE_WORDS:
        return False
    raise _bad(column, raw, "a boolean (true/false)")


def _bad(column: str, raw: str, expected: str) -> ConfigError:
    return ConfigError(f"column {column!r}: {raw!r} is not {expected}")


def _as_rate(raw: str, column: str) -> str:
    low = raw.strip().lower()
    if low == "inf":
        return "inf"
    value = _as_float(raw, column)
    if value <= 0:
        raise _bad(column, raw, "a positive rate or 'inf'")
    return raw.strip()


def _split_env(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ConfigError(f"column 'env': {item!r} is not KEY=VALUE")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _flag_names(argv: list[str]) -> set[str]:
    return {tok.split("=", 1)[0] for tok in argv if tok.startswith("-")}


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def parse_configs(
    path: Path,
    *,
    mode: str,
    default_profile_mode: str = "off",
    force_profile_off: bool = False,
) -> list[BenchConfig]:
    """Parse and validate the configuration CSV.

    ``mode`` is the vllm-qaic install mode (``aot``/``pyt``); it gates the
    profiling columns, since AoT has no working profiler (see profiler.py).
    """
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            line
            for line in handle.read().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not rows:
        raise ConfigError(f"config file is empty: {path}")

    reader = csv.DictReader(rows)
    if reader.fieldnames is None:
        raise ConfigError(f"config file has no header: {path}")

    header = [name.strip() for name in reader.fieldnames]
    unknown = [
        name
        for name in header
        if name not in KNOWN_COLUMNS and not name.startswith("x_")
    ]
    if unknown:
        raise ConfigError(
            f"unknown column(s) in {path}: {', '.join(unknown)}. "
            f"Prefix passthrough label columns with 'x_'."
        )

    configs: list[BenchConfig] = []
    errors: list[str] = []
    seen_ids: set[str] = set()

    for offset, raw_row in enumerate(reader, start=2):
        row = {
            (key or "").strip(): (value or "").strip() for key, value in raw_row.items()
        }
        try:
            config = _build_config(
                row,
                row_number=offset,
                mode=mode,
                default_profile_mode=default_profile_mode,
                force_profile_off=force_profile_off,
            )
        except ConfigError as exc:
            errors.append(f"row {offset}: {exc}")
            continue
        if config.config_id in seen_ids:
            errors.append(f"row {offset}: duplicate config_id {config.config_id!r}")
            continue
        seen_ids.add(config.config_id)
        configs.append(config)

    if errors:
        raise ConfigError(
            f"{len(errors)} problem(s) in {path}:\n  " + "\n  ".join(errors)
        )
    if not configs:
        raise ConfigError(f"no configuration rows in {path}")
    return configs


def _build_config(
    row: dict[str, str],
    *,
    row_number: int,
    mode: str,
    default_profile_mode: str,
    force_profile_off: bool,
) -> BenchConfig:
    def get(column: str) -> str:
        return row.get(column, "")

    for required in ("config_id", "model", "max_model_len", "num_prompts"):
        if not get(required):
            raise ConfigError(f"missing required column {required!r}")

    config_id = get("config_id")
    if not CONFIG_ID_RE.match(config_id):
        raise ConfigError(
            f"config_id {config_id!r} must match {CONFIG_ID_RE.pattern} "
            f"(it is used as a directory name)"
        )

    warnings: list[str] = []
    config = BenchConfig(
        config_id=config_id,
        model=get("model"),
        max_model_len=_as_int(get("max_model_len"), "max_model_len"),
        num_prompts=_as_int(get("num_prompts"), "num_prompts"),
        row_number=row_number,
        warnings=warnings,
    )

    if get("tokenizer"):
        config.tokenizer = get("tokenizer")
    if get("max_num_seqs"):
        config.max_num_seqs = _as_int(get("max_num_seqs"), "max_num_seqs")
    if get("quantization"):
        config.quantization = get("quantization")
    if get("kv_cache_dtype"):
        config.kv_cache_dtype = get("kv_cache_dtype")
    if get("tensor_parallel_size"):
        config.tensor_parallel_size = _as_int(
            get("tensor_parallel_size"), "tensor_parallel_size"
        )
    if get("long_prefill_token_threshold"):
        config.long_prefill_token_threshold = _as_int(
            get("long_prefill_token_threshold"), "long_prefill_token_threshold"
        )
    if get("extra_server_args"):
        config.extra_server_args = shlex.split(get("extra_server_args"))
    if get("dataset_name"):
        config.dataset_name = get("dataset_name")
    if get("dataset_path"):
        config.dataset_path = get("dataset_path")
    if get("random_input_len"):
        config.random_input_len = _as_int(get("random_input_len"), "random_input_len")
    if get("random_output_len"):
        config.random_output_len = _as_int(
            get("random_output_len"), "random_output_len"
        )
    if get("random_range_ratio"):
        config.random_range_ratio = _as_float(
            get("random_range_ratio"), "random_range_ratio"
        )
    if get("request_rate"):
        config.request_rate = _as_rate(get("request_rate"), "request_rate")
    if get("max_concurrency"):
        config.max_concurrency = _as_int(get("max_concurrency"), "max_concurrency")
    if get("burstiness"):
        config.burstiness = _as_float(get("burstiness"), "burstiness")
    if get("ignore_eos"):
        config.ignore_eos = _as_bool(get("ignore_eos"), "ignore_eos")
    if get("percentile_metrics"):
        config.percentile_metrics = get("percentile_metrics")
    if get("metric_percentiles"):
        config.metric_percentiles = get("metric_percentiles")
    if get("extra_client_args"):
        config.extra_client_args = shlex.split(get("extra_client_args"))
    if get("env"):
        config.env = _split_env(get("env"))

    config.labels = {
        key: value for key, value in row.items() if key.startswith("x_") and value
    }
    config.profile = _build_profile(
        row,
        mode=mode,
        default_profile_mode=default_profile_mode,
        force_profile_off=force_profile_off,
    )

    _validate(config, mode=mode)
    return config


def _build_profile(
    row: dict[str, str],
    *,
    mode: str,
    default_profile_mode: str,
    force_profile_off: bool,
) -> ProfileSpec:
    requested = (row.get("profile_mode") or "").strip().lower()
    if requested and requested not in PROFILE_MODES:
        raise ConfigError(
            f"column 'profile_mode': {requested!r} must be one of "
            f"{', '.join(PROFILE_MODES)}"
        )

    set_tuning = [col for col in PROFILE_TUNING_COLUMNS if (row.get(col) or "").strip()]

    if force_profile_off:
        spec = ProfileSpec(mode="off")
        return spec

    spec = ProfileSpec(mode=requested or default_profile_mode)

    # Mirror upstream's "only applicable when" strictness: tuning knobs with
    # profiling disabled are a mistake, not a harmless no-op.
    if not spec.enabled and set_tuning:
        raise ConfigError(
            f"{', '.join(set_tuning)} set but profile_mode is 'off'. "
            f"Set profile_mode (or --profile) to torch/lrt/both, or drop these "
            f"columns."
        )

    if spec.enabled and mode == "aot":
        reason = (
            "QaicWorkerAoT.profile() is a no-op that only logs "
            "'profile is not supported with QAIC AoT mode'"
            if spec.wants_torch
            else "the LRT profiling hook (optional_qaic_profiling) exists only "
            "in QaicModelRunnerPyt"
        )
        raise ConfigError(
            f"profile_mode={spec.mode!r} is not supported with --mode aot: {reason}. "
            f"Use --mode pyt for profiling, or profile_mode=off to compare "
            f"AoT performance without attribution."
        )

    def num(column: str, attr: str) -> None:
        raw = (row.get(column) or "").strip()
        if raw:
            setattr(spec, attr, _as_int(raw, column))

    def flag(column: str, attr: str) -> None:
        raw = (row.get(column) or "").strip()
        if raw:
            setattr(spec, attr, _as_bool(raw, column))

    num("profile_num_prompts", "num_prompts")
    num("profile_wait_iterations", "wait_iterations")
    num("profile_warmup_iterations", "warmup_iterations")
    num("profile_active_iterations", "active_iterations")
    num("profile_delay_iterations", "delay_iterations")
    num("profile_max_iterations", "max_iterations")
    flag("profile_with_stack", "with_stack")
    flag("profile_record_shapes", "record_shapes")
    flag("profile_with_memory", "with_memory")
    flag("profile_with_flops", "with_flops")
    flag("profile_detailed_annotation", "detailed_annotation")
    flag("profile_ignore_frontend", "ignore_frontend")

    # Upstream ProfilerConfig: active_iterations has ge=1, the rest ge=0.
    if spec.active_iterations < 1:
        raise ConfigError("profile_active_iterations must be >= 1")
    for attr in (
        "wait_iterations",
        "warmup_iterations",
        "delay_iterations",
        "max_iterations",
    ):
        if getattr(spec, attr) < 0:
            raise ConfigError(f"profile_{attr} must be >= 0")
    if spec.num_prompts is not None and spec.num_prompts < 1:
        raise ConfigError("profile_num_prompts must be >= 1")
    return spec


def _validate(config: BenchConfig, *, mode: str) -> None:
    """Cross-field rules. Hard errors only where a run would be meaningless."""
    if config.random_output_len < 1:
        raise ConfigError("random_output_len must be >= 1")

    if config.dataset_name == "random":
        needed = config.random_input_len + config.random_output_len
        if config.max_model_len < needed:
            raise ConfigError(
                f"max_model_len ({config.max_model_len}) < random_input_len + "
                f"random_output_len ({config.random_input_len} + "
                f"{config.random_output_len} = {needed}). The server would reject "
                f"or truncate requests, and truncation differences between commits "
                f"produce meaningless deltas."
            )
    if config.dataset_name != "random" and not config.dataset_path:
        config.warnings.append(
            f"dataset_name={config.dataset_name!r} without dataset_path; "
            f"the client may need a path"
        )

    if not config.ignore_eos:
        config.warnings.append(
            "ignore_eos=false: output length becomes model-dependent, so a decode "
            "regression is indistinguishable from the model emitting EOS earlier"
        )

    if config.request_rate == "inf" and config.max_concurrency is None:
        config.warnings.append(
            "request_rate=inf with no max_concurrency measures saturation "
            "throughput; TTFT/ITL percentiles from this row are queueing-dominated "
            "and weak regression signals"
        )

    if config.num_prompts < config.max_num_seqs:
        config.warnings.append(
            f"num_prompts ({config.num_prompts}) < max_num_seqs "
            f"({config.max_num_seqs}); the engine never reaches full batch"
        )

    # Duplicate-flag detection: if extras set something a typed column also
    # sets, precedence would be ambiguous and silently version-dependent.
    server_typed = _flag_names(config.server_args(include_extras=False)[1:])
    clash = _flag_names(config.extra_server_args) & server_typed
    if clash:
        raise ConfigError(
            f"extra_server_args duplicates typed column flag(s): "
            f"{', '.join(sorted(clash))}. Use the typed column instead."
        )

    client_typed = _flag_names(config.client_args(include_extras=False))
    clash = _flag_names(config.extra_client_args) & client_typed
    if clash:
        raise ConfigError(
            f"extra_client_args duplicates typed column flag(s): "
            f"{', '.join(sorted(clash))}. Use the typed column instead."
        )

    for reserved in ("--profiler-config", "--host", "--port"):
        if any(tok.startswith(reserved) for tok in config.extra_server_args):
            raise ConfigError(
                f"extra_server_args must not set {reserved}; vqperf owns it "
                f"(use the profile_* columns for profiling)"
            )
    for reserved in ("--save-result", "--result-dir", "--result-filename", "--profile"):
        if any(tok.startswith(reserved) for tok in config.extra_client_args):
            raise ConfigError(
                f"extra_client_args must not set {reserved}; vqperf owns it"
            )


def describe(config: BenchConfig) -> dict[str, Any]:
    """Flat, report-friendly view of the knobs that identify a configuration."""
    return {
        "config_id": config.config_id,
        "model": config.model,
        "max_model_len": config.max_model_len,
        "max_num_seqs": config.max_num_seqs,
        "quantization": config.quantization or "",
        "kv_cache_dtype": config.kv_cache_dtype or "",
        "dataset_name": config.dataset_name,
        "random_input_len": config.random_input_len,
        "random_output_len": config.random_output_len,
        "num_prompts": config.num_prompts,
        "request_rate": config.request_rate,
        "max_concurrency": (
            config.max_concurrency if config.max_concurrency is not None else ""
        ),
        "profile_mode": config.profile.mode,
        **config.labels,
    }
