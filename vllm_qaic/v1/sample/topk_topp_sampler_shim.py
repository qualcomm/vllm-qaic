# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/v1/sample/ops/topk_topp_sampler.py

"""Compatibility wrapper for the centralized AOT top-k/top-p patch.

The production patch is registered from :mod:`vllm_qaic.patch` at plugin
initialization time. Keep this import path for local experiment scripts that
still call ``topk_topp_sampler_shim.install()``.
"""

from __future__ import annotations

from vllm_qaic.patch.patch_topk_topp_sampler import install

__all__ = ["install"]
