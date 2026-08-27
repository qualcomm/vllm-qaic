# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for vllm_qaic.patch.patch_utils.

Tests that importing the patch module registers the QAIC-specific
"mxint8" KV-cache dtype in vLLM's STR_DTYPE_TO_TORCH_DTYPE mapping.
No hardware required.

Coverage areas
--------------
1. "mxint8" key is present in STR_DTYPE_TO_TORCH_DTYPE after import
2. "mxint8" maps to torch.int8
3. Existing standard dtype mappings are not disturbed
"""

import pytest
import torch

# Import the patch module to trigger the registration
import vllm_qaic.patch.patch_utils  # noqa: F401

from vllm.utils import torch_utils


class TestMxint8DtypeRegistration:
    """
    vllm_qaic.patch.patch_utils registers "mxint8" → torch.int8 in
    vllm.utils.torch_utils.STR_DTYPE_TO_TORCH_DTYPE so that vLLM's
    KV-cache dtype validation accepts the QAIC-specific "mxint8" string.
    """

    def test_mxint8_key_present(self):
        """'mxint8' must be present in STR_DTYPE_TO_TORCH_DTYPE after import."""
        assert "mxint8" in torch_utils.STR_DTYPE_TO_TORCH_DTYPE

    def test_mxint8_maps_to_int8(self):
        """'mxint8' must map to torch.int8."""
        assert torch_utils.STR_DTYPE_TO_TORCH_DTYPE["mxint8"] == torch.int8

    def test_standard_float16_not_disturbed(self):
        """Standard 'float16' mapping must not be affected by the patch."""
        assert torch_utils.STR_DTYPE_TO_TORCH_DTYPE.get("float16") == torch.float16

    def test_standard_bfloat16_not_disturbed(self):
        """Standard 'bfloat16' mapping must not be affected by the patch."""
        assert torch_utils.STR_DTYPE_TO_TORCH_DTYPE.get("bfloat16") == torch.bfloat16

    def test_standard_float32_not_disturbed(self):
        """Standard 'float32' mapping must not be affected by the patch."""
        assert torch_utils.STR_DTYPE_TO_TORCH_DTYPE.get("float32") == torch.float32

    def test_mxint8_is_integer_dtype(self):
        """torch.int8 must be an integer dtype (not floating point)."""
        dtype = torch_utils.STR_DTYPE_TO_TORCH_DTYPE["mxint8"]
        assert not dtype.is_floating_point

    def test_mxint8_dtype_itemsize(self):
        """torch.int8 must be 1 byte (8-bit integer)."""
        dtype = torch_utils.STR_DTYPE_TO_TORCH_DTYPE["mxint8"]
        assert torch.tensor(0, dtype=dtype).element_size() == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
