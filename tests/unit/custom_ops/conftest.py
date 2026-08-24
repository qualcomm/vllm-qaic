# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import os
import sys


def _resolve_qaic_visible_device() -> str:
    """Mirror device_unit_and_e2e/conftest.py's --device-id pool parsing so
    this module honors whatever device pool the run was invoked with,
    instead of a hardcoded id pair that may not exist on this host. Read
    straight off sys.argv since the env var must be set before torch_qaic
    is imported, before any pytest fixture is resolvable.
    """
    for i, arg in enumerate(sys.argv):
        if arg == "--device-id" and i + 1 < len(sys.argv):
            return sys.argv[i + 1].split(",")[-1]
        if arg.startswith("--device-id="):
            return arg.split("=", 1)[1].split(",")[-1]
    return "0"


# Must be set before torch_qaic is imported
os.environ.setdefault("QAIC_VISIBLE_DEVICES", _resolve_qaic_visible_device())

import pytest

try:
    import torch_qaic  # noqa: F401
    _QAIC_AVAILABLE = True
except Exception:
    _QAIC_AVAILABLE = False


def pytest_addoption(parser):
    """Register all device CLI options so pytest accepts them when run from this subdirectory."""
    options = [
        ("--model-name",   str,  None,   "HuggingFace model name for on-device tests"),
        ("--seq-len",      int,  None,   "Prefill sequence length"),
        ("--ctx-len",      int,  None,   "Context (max model) length"),
        ("--decode-bsz",   int,  None,   "Decode batch size"),
        ("--dtype",        str,  None,   "Quantization dtype (e.g. mxfp6)"),
        ("--kv-dtype",     str,  None,   "KV-cache dtype (e.g. mxint8)"),
        ("--device-group", int,  None,   "Number of devices in the device group"),
        ("--device-id",    int,  None,   "Starting device ID"),
        ("--lora-path",    str,  None,   "Path to a pre-downloaded LoRA adapter"),
        ("--override-qaic-config", str, None, "Override QAIC config as JSON string"),
    ]
    for name, typ, default, help_text in options:
        try:
            parser.addoption(name, type=typ, default=default, help=help_text)
        except ValueError:
            pass  # already registered

    try:
        parser.addoption(
            "--mode", type=str, default="auto",
            choices=["auto", "aot", "eager"],
            help="Inference mode for on-device tests",
        )
    except ValueError:
        pass


def pytest_configure(config):
    config.addinivalue_line("markers", "qaic: requires a QAIC device (torch_qaic)")


@pytest.fixture(scope="session")
def qaic_device():
    if not _QAIC_AVAILABLE:
        pytest.skip("torch_qaic not available — skipping hardware kernel test")
    return "qaic:0"


@pytest.fixture
def vllm_config():
    """Provide a minimal VllmConfig context for tests that instantiate CustomOp subclasses."""
    from unittest.mock import patch
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_qaic.platform_base import QaicPlatform
    # VllmConfig.__post_init__ calls check_and_update_config which requires a full
    # ModelConfig. Patch it out so we can build a bare VllmConfig for unit tests.
    with patch.object(QaicPlatform, "check_and_update_config"):
        with set_current_vllm_config(VllmConfig()):
            yield
