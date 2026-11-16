# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Expose shared disaggregated-serving fixtures to this test subtree."""

from .ec_fixtures import (
    _ec_disagg_test_config as _ec_disagg_test_config,
    ec_disagg_server as ec_disagg_server,
)
from .fixtures import (
    _disagg_test_config as _disagg_test_config,
    disagg_server as disagg_server,
    kv_handoff_server as kv_handoff_server,
    kv_port as kv_port,
)
