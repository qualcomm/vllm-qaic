#!/bin/bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
#
# Container entrypoint for vllm-qaic images.
#
# If USER_UID/USER_GID are provided (via `docker run -e`), creates a
# non-root user matching the caller's host uid/gid, grants it passwordless
# sudo, and optionally joins it to a "qaic" group matching QAIC_GID (the
# gid of the host's qaic device group) so it can access /dev/accel. The
# container command then runs as that user instead of root.
#
# If USER_UID/USER_GID are not set, runs the command as root directly.
#
# Usage (typically set via docker run -e, not passed manually):
#   docker run -e USER_UID=$(id -u) -e USER_GID=$(id -g) \
#              -e QAIC_GID=$(getent group qaic | cut -d: -f3 || echo '') \
#              ...

set -euo pipefail

if [ -z "${USER_UID:-}" ] || [ -z "${USER_GID:-}" ]; then
    exec "$@"
fi

DEFAULT_NAME="user"

# The base image may already ship a group/user at this gid/uid (e.g. Ubuntu's
# default uid/gid 1000) — reuse it under its existing name rather than erroring.
EXISTING_GROUP="$(getent group "${USER_GID}" | cut -d: -f1 || true)"
if [ -z "${EXISTING_GROUP}" ]; then
    groupadd -g "${USER_GID}" "${DEFAULT_NAME}"
fi

EXISTING_USER="$(getent passwd "${USER_UID}" | cut -d: -f1 || true)"
if [ -n "${EXISTING_USER}" ]; then
    USERNAME="${EXISTING_USER}"
else
    USERNAME="${DEFAULT_NAME}"
    useradd -m -u "${USER_UID}" -g "${USER_GID}" -s /bin/bash "${USERNAME}"
fi

# Passwordless sudo, primarily for debugging/pip install inside the container.
usermod -aG sudo "${USERNAME}"
echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Join the qaic group (matching the host's qaic gid) if one was provided.
# The QAIC runtime looks up the group by name ("qaic"), not by gid, so a
# group literally named "qaic" must exist — even if that gid is already
# taken by another group in the base image (-o allows the duplicate gid).
if [ -n "${QAIC_GID:-}" ]; then
    if ! getent group qaic >/dev/null; then
        groupadd -o -g "${QAIC_GID}" qaic
    fi
    usermod -aG qaic "${USERNAME}"
fi

USERHOME="/home/${USERNAME}"
mkdir -p "${USERHOME}"
chown -R "${USER_UID}:${USER_GID}" "${USERHOME}"

exec runuser -u "${USERNAME}" -- "$@"
