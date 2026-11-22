# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""``vllm serve`` lifecycle for one benchmark cell.

``vllm bench serve`` is a client only, so vqperf owns the server. One server
per (commit x configuration) cell -- never reused -- because max-model-len,
max-num-seqs and quantization are server-side engine args.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import BenchConfig
from .env import Environment
from .util import (
    VqperfError,
    format_argv,
    free_port,
    tail_file,
    terminate_tree,
    wait_for_port_release,
)

logger = logging.getLogger("vqperf.server")


class ServerStartError(VqperfError):
    pass


class VllmServer:
    """Context manager around one ``vllm serve`` process."""

    def __init__(
        self,
        env: Environment,
        config: BenchConfig,
        *,
        log_path: Path,
        devices: str,
        extra_args: list[str] | None = None,
        extra_env: dict[str, str] | None = None,
        cache_dir: Path,
        timeout_s: float = 2700.0,
        host: str = "127.0.0.1",
    ) -> None:
        self.env = env
        self.config = config
        self.log_path = log_path
        self.devices = devices
        self.extra_args = extra_args or []
        self.extra_env = extra_env or {}
        self.cache_dir = cache_dir
        self.timeout_s = timeout_s
        self.host = host
        self.port = free_port()
        self.process: subprocess.Popen | None = None
        self.started_at = 0.0
        self.ready_after_s: float | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _build_env(self) -> dict[str, str]:
        hf_home = self.cache_dir / "hf"
        qeff_home = self.cache_dir / "qeff"
        hf_home.mkdir(parents=True, exist_ok=True)
        qeff_home.mkdir(parents=True, exist_ok=True)
        env = {
            "QAIC_VISIBLE_DEVICES": self.devices,
            # Shared across both commits so a model download and a compatible
            # QPC are paid for once, not per commit.
            "HF_HOME": os.environ.get("HF_HOME", str(hf_home)),
            "QEFF_HOME": os.environ.get("QEFF_HOME", str(qeff_home)),
        }
        env.update(self.config.env)
        env.update(self.extra_env)
        return env

    def _argv(self) -> list[str]:
        return [
            *(self.env.serve_argv),
            *self.config.server_args(),
            "--host",
            self.host,
            "--port",
            str(self.port),
            *self.extra_args,
        ]

    def __enter__(self) -> VllmServer:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def start(self, retry_on_port_clash: bool = True) -> None:
        argv = self._argv()
        env = self._build_env()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        (self.log_path.parent / "server.cmd.txt").write_text(
            format_argv(argv, env=env) + "\n", encoding="utf-8"
        )
        logger.info(
            "starting server for %s on port %s", self.config.config_id, self.port
        )
        # Not a context manager: this handle is the server's stdout for as
        # long as it runs, and is closed in stop().
        handle = open(self.log_path, "wb")  # noqa: SIM115
        full_env = dict(os.environ)
        full_env.update(env)
        self.started_at = time.monotonic()
        self.process = subprocess.Popen(  # noqa: S603
            argv,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=full_env,
        )
        try:
            self._wait_ready()
        except ServerStartError:
            log = tail_file(self.log_path, 60)
            self.stop()
            if retry_on_port_clash and "address already in use" in log.lower():
                logger.warning("port %s was taken; retrying once", self.port)
                self.port = free_port()
                self.start(retry_on_port_clash=False)
                return
            raise
        finally:
            handle.close()

    def _wait_ready(self) -> None:
        """Poll /health, then confirm /v1/models serves the expected model.

        The long default timeout exists because the first AoT run for a model
        compiles a QPC (documented as 3-10 minutes, longer for big models).
        """
        deadline = time.monotonic() + self.timeout_s
        last_error = ""
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise ServerStartError(
                    f"server exited with rc={self.process.returncode} before becoming "
                    f"healthy\nlog: {self.log_path}\n{tail_file(self.log_path)}"
                )
            try:
                with urllib.request.urlopen(  # noqa: S310
                    f"{self.base_url}/health", timeout=10
                ) as response:
                    if response.status == 200:
                        self._check_model()
                        self.ready_after_s = time.monotonic() - self.started_at
                        logger.info(
                            "server ready after %.0fs (%s)",
                            self.ready_after_s,
                            self.config.config_id,
                        )
                        return
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                last_error = str(exc)
            time.sleep(3.0)

        raise ServerStartError(
            f"server for {self.config.config_id} not healthy within "
            f"{self.timeout_s:.0f}s (last error: {last_error})\n"
            f"log: {self.log_path}\n{tail_file(self.log_path)}"
        )

    def _check_model(self) -> None:
        try:
            with urllib.request.urlopen(  # noqa: S310
                f"{self.base_url}/v1/models", timeout=30
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ServerStartError(f"/v1/models unreachable: {exc}") from exc

        served = [item.get("id", "") for item in payload.get("data", [])]
        if self.config.model not in served:
            logger.warning(
                "server reports models %s, expected %s; continuing with the "
                "configured name",
                served,
                self.config.model,
            )

    def post(self, path: str, timeout: float = 120.0) -> int:
        """POST with an empty body; returns the HTTP status."""
        request = urllib.request.Request(  # noqa: S310
            f"{self.base_url}{path}", data=b"", method="POST"
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return int(response.status)

    def stop(self) -> None:
        if self.process is None:
            return
        logger.debug("stopping server on port %s", self.port)
        terminate_tree(self.process)
        self.process = None
        if not wait_for_port_release(self.port):
            logger.warning(
                "port %s still bound after teardown; a worker may have survived",
                self.port,
            )
