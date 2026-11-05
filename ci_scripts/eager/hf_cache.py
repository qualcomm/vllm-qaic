# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Shared HF cache cleanup for the eager CI runners."""

import os


def _path_owned_by_current_user(path) -> bool:
    """Owned by the uid running this process; unreadable counts as not ours."""
    try:
        return os.stat(path).st_uid == os.getuid()
    except OSError:
        return False


def _revision_owned_by_current_user(revision) -> bool:
    """A revision is ours only if its snapshot and every backing blob are.

    Blobs matter because they hold the bytes deletion frees: a revision can
    reference a blob another user downloaded, which delete_revisions() would
    free once no surviving revision needs it.
    """
    return _path_owned_by_current_user(revision.snapshot_path) and all(
        _path_owned_by_current_user(f.blob_path) for f in revision.files
    )


def delete_hf_checkpoint(model_name: str) -> None:
    """Delete this user's cached revisions of model_name, freeing disk between runs.

    Uses the cache from HF_HOME / HF_HUB_CACHE, so nothing outside it is touched.
    Deletes only revisions this user downloaded -- the cache is often shared and
    group-writable, and the flag authorises evicting your own downloads, not a
    checkpoint someone else relies on. Skips local paths, runs whether or not the
    model succeeded, and never raises, since it runs during teardown.
    """
    if os.path.isdir(model_name):
        print(f"[cleanup] '{model_name}' is a local path - not deleting")
        return

    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        cached = [
            rev
            for repo in cache.repos
            if repo.repo_type == "model" and repo.repo_id == model_name
            for rev in repo.revisions
        ]
        deletable = [
            rev.commit_hash for rev in cached if _revision_owned_by_current_user(rev)
        ]

        if len(deletable) != len(cached):
            print(
                f"[cleanup] keeping {len(cached) - len(deletable)} revision(s) of "
                f"'{model_name}' not downloaded by this user"
            )
        if not deletable:
            print(f"[cleanup] no deletable cache entry for '{model_name}'")
            return

        strategy = cache.delete_revisions(*deletable)
        freed = strategy.expected_freed_size_str
        strategy.execute()
        print(f"[cleanup] deleted checkpoint '{model_name}' (freed {freed})")

    except Exception as exc:  # noqa: BLE001 - teardown must not mask the run
        print(f"[cleanup] could not delete '{model_name}': {exc}")
