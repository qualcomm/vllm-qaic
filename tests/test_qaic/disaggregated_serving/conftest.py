# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""CLI options for the hardware-dependent disagg+SpD E2E suite.

Scoped to this directory (not tests/conftest.py) because these options are
specific to test_spd_disagg.py's spd_test_config fixture and are irrelevant
to the rest of the vllm-qaic test suite, which only needs
--test-device-group (see tests/conftest.py).

These tests are hardware-dependent and are not run as part of the
non-hardware verification gate; see docs/qaic/disagg_spd_port.md for the
follow-up commands to run them on a reserved device group.
"""


def pytest_addoption(parser):
    parser.addoption("--model-name", type=str, default=None)
    parser.addoption("--seq-len", type=int, default=None)
    parser.addoption("--ctx-len", type=int, default=None)
    parser.addoption("--decode-bsz", type=int, default=None)
    parser.addoption("--device-id", type=int, default=None)
    parser.addoption("--prefill-devices", type=str, default=None)
    parser.addoption("--decode-devices", type=str, default=None)
    parser.addoption("--prefill-max-num-seqs", type=int, default=2)
    parser.addoption("--client-request-timeout", type=int, default=60)
    parser.addoption("--disaggregated-startup-timeout", type=int, default=1200)
    parser.addoption("--disaggregated-server-port", type=int, default=8081)
    parser.addoption("--kv-transfer-port", type=int, default=5656)
    parser.addoption("--speculative-model", type=str, default=None)
    parser.addoption("--speculative-method", type=str, default=None)
    parser.addoption("--num-speculative-tokens", type=int, default=None)
    parser.addoption("--ngram-prompt-lookup-max", type=int, default=None)
    parser.addoption("--ngram-prompt-lookup-min", type=int, default=None)
