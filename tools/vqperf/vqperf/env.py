# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Per-commit conda environments: creation, install.sh, and capability probes."""

from __future__ import annotations

import hashlib
import json
import logging
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from .profiler import ProfilerStyle
from .repo import Checkout
from .util import VqperfError, run, which

logger = logging.getLogger("vqperf.env")

CONDA_CANDIDATES = ("micromamba", "mamba", "conda")

PROBE_PACKAGES = (
    "vllm",
    "vllm_qaic",
    "torch",
    "torch_qaic",
    "transformers",
    "qefficient",
)


@dataclass
class BenchInvocation:
    """How to invoke the serving benchmark in one environment."""

    argv: list[str]
    form: str  # cli | module | script

    def with_args(self, args: list[str]) -> list[str]:
        return [*self.argv, *args]


@dataclass
class Environment:
    """A provisioned environment for one commit."""

    sha: str
    prefix: Path
    mode: str
    python_version: str
    checkout: Checkout
    cache_key: str
    versions: dict[str, str] = field(default_factory=dict)
    bench: BenchInvocation | None = None
    serve_argv: list[str] = field(default_factory=list)
    profiler_style: ProfilerStyle = "unavailable"

    @property
    def vllm_version(self) -> str:
        return self.versions.get("vllm", "unknown")


class EnvManager:
    def __init__(
        self,
        workdir: Path,
        *,
        mode: str,
        python_version: str,
        out_dir: Path,
    ) -> None:
        self.workdir = workdir
        self.mode = mode
        self.python_version = python_version
        self.out_dir = out_dir
        self.conda = self._find_conda()

    @staticmethod
    def _find_conda() -> str:
        for candidate in CONDA_CANDIDATES:
            path = which(candidate)
            if path:
                logger.debug("using %s for environment management", path)
                return path
        raise VqperfError(
            "no conda/mamba/micromamba found on PATH; vqperf needs one to create "
            "per-commit environments"
        )

    # -- cache key ---------------------------------------------------------

    def cache_key(self, checkout: Checkout) -> str:
        install_sh = checkout.path / "scripts" / "install.sh"
        utility_sh = checkout.path / "scripts" / "utility.sh"
        digest = hashlib.sha256()
        for path in (install_sh, utility_sh):
            if path.exists():
                digest.update(path.read_bytes())
        payload = {
            "sha": checkout.sha,
            "mode": self.mode,
            "python": self.python_version,
            "scripts": digest.hexdigest(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[
            :16
        ]

    def prefix_for(self, checkout: Checkout) -> Path:
        return (
            self.workdir
            / "envs"
            / f"{self.mode}-py{self.python_version}-{checkout.short}"
        )

    # -- provisioning ------------------------------------------------------

    def provision(self, checkout: Checkout, *, reuse: bool = False) -> Environment:
        prefix = self.prefix_for(checkout)
        key = self.cache_key(checkout)
        sentinel = prefix / ".vqperf-provisioned.json"
        log_dir = self.out_dir / "provision" / checkout.short
        log_dir.mkdir(parents=True, exist_ok=True)

        if reuse and sentinel.exists():
            try:
                recorded = json.loads(sentinel.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                recorded = {}
            if recorded.get("cache_key") == key:
                logger.info("reusing environment %s", prefix)
                return self._finalize(checkout, prefix, key, log_dir)
            logger.info("environment %s is stale; rebuilding", prefix)

        if prefix.exists():
            # A partial build must never look complete; start clean.
            import shutil

            shutil.rmtree(prefix)

        logger.info(
            "creating env for %s (%s, py%s)",
            checkout.short,
            self.mode,
            self.python_version,
        )
        run(
            [
                self.conda,
                "create",
                "-y",
                "-p",
                str(prefix),
                f"python={self.python_version}",
            ],
            log_path=log_dir / "conda-create.log",
            check=True,
        )

        self._assert_prefix(prefix, log_dir)

        logger.info("running scripts/install.sh %s for %s", self.mode, checkout.short)
        install_script = checkout.path / "scripts" / "install.sh"
        if not install_script.exists():
            raise VqperfError(
                f"{install_script} not found; commit {checkout.short} does not look "
                f"like a vllm-qaic checkout"
            )
        run(
            self._conda_run(prefix)
            + ["bash", "-lc", f"./scripts/install.sh {shlex.quote(self.mode)}"],
            cwd=checkout.path,
            log_path=log_dir / "install.log",
            check=True,
        )

        sentinel.write_text(
            json.dumps(
                {"cache_key": key, "sha": checkout.sha, "mode": self.mode}, indent=2
            ),
            encoding="utf-8",
        )
        return self._finalize(checkout, prefix, key, log_dir)

    def _conda_run(self, prefix: Path) -> list[str]:
        return [self.conda, "run", "-p", str(prefix), "--no-capture-output"]

    def _assert_prefix(self, prefix: Path, log_dir: Path) -> None:
        """Confirm ``conda run`` really targets this prefix.

        scripts/install.sh resolves its interpreter from VIRTUAL_ENV or
        ``command -v python``. If PATH does not land inside the prefix, the
        install would silently target another interpreter -- a hard failure is
        far better than an install into the wrong env discovered hours later.
        """
        probe = run(
            self._conda_run(prefix) + ["python", "-c", "import sys; print(sys.prefix)"],
            capture=True,
            log_path=log_dir / "prefix-probe.log",
        )
        reported = probe.stdout.strip().splitlines()[-1].strip() if probe.ok else ""
        if Path(reported).resolve() != prefix.resolve():
            raise VqperfError(
                f"'conda run -p {prefix}' resolves python to {reported!r}, not the "
                f"target prefix. scripts/install.sh would install into the wrong "
                f"interpreter."
            )

    # -- probes ------------------------------------------------------------

    def _finalize(
        self, checkout: Checkout, prefix: Path, key: str, log_dir: Path
    ) -> Environment:
        env = Environment(
            sha=checkout.sha,
            prefix=prefix,
            mode=self.mode,
            python_version=self.python_version,
            checkout=checkout,
            cache_key=key,
        )
        env.versions = self.probe_versions(prefix, log_dir)
        env.bench = self.resolve_bench(prefix, log_dir)
        env.serve_argv = self.resolve_serve(prefix)
        env.profiler_style = self.probe_profiler_style(prefix, log_dir)
        logger.info(
            "env %s ready: vllm=%s vllm_qaic=%s torch=%s profiler=%s",
            checkout.short,
            env.versions.get("vllm", "?"),
            env.versions.get("vllm_qaic", "?"),
            env.versions.get("torch", "?"),
            env.profiler_style,
        )
        return env

    def probe_versions(self, prefix: Path, log_dir: Path) -> dict[str, str]:
        code = (
            "import json\n"
            "from importlib.metadata import version, PackageNotFoundError\n"
            f"names = {list(PROBE_PACKAGES)!r}\n"
            "out = {}\n"
            "for name in names:\n"
            "    try:\n"
            "        out[name] = version(name)\n"
            "    except PackageNotFoundError:\n"
            "        pass\n"
            "print(json.dumps(out))\n"
        )
        result = run(
            self._conda_run(prefix) + ["python", "-c", code],
            capture=True,
            log_path=log_dir / "versions.log",
        )
        versions: dict[str, str] = {}
        if result.ok:
            for line in reversed(result.stdout.strip().splitlines()):
                try:
                    versions = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

        freeze = run(
            self._conda_run(prefix) + ["python", "-m", "pip", "freeze"],
            capture=True,
        )
        if freeze.ok:
            (log_dir / "pip-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
        return versions

    def resolve_bench(self, prefix: Path, log_dir: Path) -> BenchInvocation:
        """Resolve how to run the serving benchmark in this environment.

        Upstream moved ``benchmarks/benchmark_serving.py`` to ``vllm bench
        serve``; at v0.30.0 the old path is a stub that prints DEPRECATED and
        exits 1. Resolving per environment keeps the measurement code matched to
        the vllm each commit actually pins.
        """
        cli = self._conda_run(prefix) + ["vllm", "bench", "serve", "--help"]
        if run(cli, capture=True, log_path=log_dir / "probe-bench-cli.log").ok:
            return BenchInvocation(
                self._conda_run(prefix) + ["vllm", "bench", "serve"], "cli"
            )

        module = self._conda_run(prefix) + [
            "python",
            "-m",
            "vllm.entrypoints.cli.main",
            "bench",
            "serve",
            "--help",
        ]
        if run(module, capture=True, log_path=log_dir / "probe-bench-module.log").ok:
            return BenchInvocation(
                self._conda_run(prefix)
                + ["python", "-m", "vllm.entrypoints.cli.main", "bench", "serve"],
                "module",
            )

        script = self._legacy_script(prefix)
        if script is not None:
            return BenchInvocation(
                self._conda_run(prefix) + ["python", str(script)], "script"
            )

        raise VqperfError(
            f"cannot find a serving benchmark in {prefix}: 'vllm bench serve' and "
            f"the CLI module both failed, and no usable benchmark_serving.py exists"
        )

    def _legacy_script(self, prefix: Path) -> Path | None:
        """Locate a pre-CLI ``benchmark_serving.py``, skipping the stub."""
        code = (
            "import pathlib, vllm\n"
            "root = pathlib.Path(vllm.__file__).resolve().parent.parent\n"
            "print(root / 'benchmarks' / 'benchmark_serving.py')\n"
        )
        probe = run(self._conda_run(prefix) + ["python", "-c", code], capture=True)
        if not probe.ok:
            return None
        candidate = Path(probe.stdout.strip().splitlines()[-1].strip())
        if not candidate.exists():
            return None
        head = candidate.read_text(encoding="utf-8", errors="replace").splitlines()[:40]
        if any("DEPRECATED" in line for line in head):
            logger.debug("%s is a deprecation stub; ignoring", candidate)
            return None
        return candidate

    def resolve_serve(self, prefix: Path) -> list[str]:
        return self._conda_run(prefix) + ["vllm", "serve"]

    def probe_profiler_style(self, prefix: Path, log_dir: Path) -> ProfilerStyle:
        """Detect whether this vllm accepts ``--profiler-config``.

        Plain ``vllm serve --help`` lists only the config *groups*, not the
        individual flags, so it never contains ``--profiler-config``. The
        keyword form ``--help=profiler-config`` prints the matching arguments.
        There is no env-var fallback: ``VLLM_TORCH_PROFILER_DIR`` was removed
        upstream, so without the flag host profiling is unavailable.
        """
        help_result = run(
            self._conda_run(prefix) + ["vllm", "serve", "--help=profiler-config"],
            capture=True,
            timeout=300,
            log_path=log_dir / "probe-serve-help.log",
        )
        if help_result.ok and "--profiler-config" in help_result.stdout:
            return "profiler_config"
        return "unavailable"
