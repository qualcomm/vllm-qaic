# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Git: fork resolution, a shared mirror, and per-SHA checkouts."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .util import VqperfError, run, short_sha

logger = logging.getLogger("vqperf.repo")

DEFAULT_FORK = "qualcomm/vllm-qaic"
UPSTREAM_FORK = "qualcomm/vllm-qaic"

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_OWNER_REPO_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def resolve_fork_url(fork: str | None) -> str:
    """Accept ``owner/repo``, a full git URL, or a local path."""
    if not fork:
        return f"https://github.com/{DEFAULT_FORK}.git"
    if _OWNER_REPO_RE.match(fork):
        return f"https://github.com/{fork}.git"
    if "://" in fork or fork.startswith("git@"):
        return fork
    path = Path(fork).expanduser()
    if path.exists():
        return str(path.resolve())
    raise VqperfError(
        f"--fork {fork!r} is neither 'owner/repo', a git URL, nor an existing path"
    )


@dataclass
class Checkout:
    """A detached checkout of one fully-resolved SHA."""

    sha: str
    path: Path
    remote_url: str
    subject: str
    committed_at: str

    @property
    def short(self) -> str:
        return short_sha(self.sha)


class RepoManager:
    """Owns one bare mirror so both SHAs share a single object fetch."""

    def __init__(self, workdir: Path, fork_url: str) -> None:
        self.workdir = workdir
        self.fork_url = fork_url
        self.mirror = workdir / "mirror.git"
        self._fetched_remotes: list[str] = []

    # -- mirror ------------------------------------------------------------

    def ensure_mirror(self) -> None:
        log_dir = self.workdir / "logs"
        if not (self.mirror / "HEAD").exists():
            logger.info("cloning mirror from %s", self.fork_url)
            self.mirror.parent.mkdir(parents=True, exist_ok=True)
            run(
                ["git", "clone", "--mirror", self.fork_url, str(self.mirror)],
                log_path=log_dir / "mirror-clone.log",
                check=True,
            )
            self._fetched_remotes.append(self.fork_url)
            return

        logger.info("fetching mirror %s", self.mirror)
        run(
            [
                "git",
                "--git-dir",
                str(self.mirror),
                "remote",
                "set-url",
                "origin",
                self.fork_url,
            ],
            log_path=log_dir / "mirror-seturl.log",
            check=True,
        )
        run(
            [
                "git",
                "--git-dir",
                str(self.mirror),
                "fetch",
                "--prune",
                "origin",
                "+refs/*:refs/*",
            ],
            log_path=log_dir / "mirror-fetch.log",
            check=True,
        )
        self._fetched_remotes.append(self.fork_url)

    def _add_upstream_fallback(self) -> str:
        """Fetch qualcomm/vllm-qaic as an extra remote and report its URL."""
        url = f"https://github.com/{UPSTREAM_FORK}.git"
        if url == self.fork_url or url in self._fetched_remotes:
            return url
        logger.info("SHA not in fork; fetching fallback remote %s", url)
        git_dir = ["git", "--git-dir", str(self.mirror)]
        existing = run(git_dir + ["remote"], capture=True)
        if "vqperf-upstream" not in existing.stdout.split():
            run(
                git_dir + ["remote", "add", "vqperf-upstream", url],
                capture=True,
                check=True,
            )
        run(
            git_dir
            + [
                "fetch",
                "vqperf-upstream",
                "+refs/heads/*:refs/remotes/vqperf-upstream/*",
            ],
            log_path=self.workdir / "logs" / "mirror-fetch-upstream.log",
            check=True,
        )
        self._fetched_remotes.append(url)
        return url

    # -- resolution --------------------------------------------------------

    def resolve(self, ref: str) -> tuple[str, str]:
        """Resolve a ref to ``(full_sha, remote_url_that_supplied_it)``.

        A moving ref such as ``main`` must never label a result, so everything
        downstream keys off the returned 40-char SHA.
        """
        git_dir = ["git", "--git-dir", str(self.mirror)]
        for remote_url in (self.fork_url, None):
            if remote_url is None:
                remote_url = self._add_upstream_fallback()
                if remote_url == self.fork_url:
                    break
            result = run(
                git_dir + ["rev-parse", "--verify", f"{ref}^{{commit}}"],
                capture=True,
            )
            if result.ok:
                sha = result.stdout.strip().splitlines()[-1].strip()
                if _SHA_RE.match(sha):
                    return sha, remote_url
        raise VqperfError(
            f"ref {ref!r} not found in {self.fork_url} nor in "
            f"https://github.com/{UPSTREAM_FORK}.git"
        )

    # -- checkout ----------------------------------------------------------

    def checkout(self, sha: str, remote_url: str) -> Checkout:
        dest = self.workdir / "src" / short_sha(sha)
        log_dir = self.workdir / "logs"
        if not (dest / ".git").exists():
            logger.info("checking out %s into %s", short_sha(sha), dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            run(
                ["git", "clone", "--no-checkout", str(self.mirror), str(dest)],
                log_path=log_dir / f"clone-{short_sha(sha)}.log",
                check=True,
            )
        run(
            [
                "git",
                "-c",
                "advice.detachedHead=false",
                "-C",
                str(dest),
                "checkout",
                "--detach",
                "--force",
                sha,
            ],
            log_path=log_dir / f"checkout-{short_sha(sha)}.log",
            check=True,
        )

        status = run(["git", "-C", str(dest), "status", "--porcelain"], capture=True)
        if status.stdout.strip():
            raise VqperfError(
                f"checkout {dest} is dirty after detaching at {sha}:\n{status.stdout}"
            )

        info = run(
            ["git", "-C", str(dest), "log", "-1", "--format=%H%x09%cI%x09%s"],
            capture=True,
            check=True,
        )
        full_sha, committed_at, subject = info.stdout.strip().split("\t", 2)
        return Checkout(
            sha=full_sha,
            path=dest,
            remote_url=remote_url,
            subject=subject,
            committed_at=committed_at,
        )

    def git_info_text(self, checkout: Checkout) -> str:
        return (
            f"sha: {checkout.sha}\n"
            f"remote: {checkout.remote_url}\n"
            f"committed_at: {checkout.committed_at}\n"
            f"subject: {checkout.subject}\n"
            f"path: {checkout.path}\n"
        )
