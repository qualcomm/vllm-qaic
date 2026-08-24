# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Hardware tests for the QAIC rms_norm_hexagon kernel and its Python wrappers.

These call the compiled HVX kernel (via torch_qaic / vllm_qaic) on real QAIC
silicon — there is no CPU fallback to exercise instead. Skipped automatically
on hosts without torch_qaic via the qaic_device fixture defined below.

The pure-Python numerics that validate the FP32 reference algorithm itself
(no hardware involved) live in
vllm-qaic/tests/unit/custom_ops/test_rms_norm_kernel.py.
"""

import os
import sys


def _resolve_qaic_visible_device() -> str:
    """Mirror conftest.py's --device-id pool parsing (the same CLI
    convention `device_group`/`device_groups` resolve against) so this
    kernel test honors whatever device pool the run was actually invoked
    with, instead of a hardcoded id pair that may not exist on this host.
    Read straight off sys.argv because pytest fixtures aren't resolvable
    yet at module-import time, which is required here (env var must be set
    before torch_qaic is imported). Takes the pool's last id to avoid
    colliding with device_group/device_groups fixture-based tests, which
    acquire from the front of the pool. Falls back to conftest.py's own
    default ([0]) when --device-id wasn't passed.
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
import torch
import torch.nn.functional as F

try:
    import torch_qaic  # noqa: F401
    _QAIC_AVAILABLE = True
except Exception:
    _QAIC_AVAILABLE = False


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


def _rms_norm_ref(
    attn_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 reference: residual = attn_out + x; dst = residual / rms(residual) * weight."""
    residual = attn_out.float() + x.float()
    mean_sq = (residual * residual).mean(dim=-1, keepdim=True)
    inv_rms = 1.0 / torch.sqrt(mean_sq + epsilon)
    dst = residual * inv_rms * weight.float()
    return dst, residual


def _cast_ref(
    dst_ref: torch.Tensor,
    residual_ref: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    return dst_ref.to(dtype), residual_ref.to(dtype)


def _make_inputs(M: int, N: int, dtype: torch.dtype, seed: int = 42):
    torch.manual_seed(seed)
    attn_out = torch.randn(M, N, dtype=dtype) * 0.1
    x        = torch.randn(M, N, dtype=dtype) * 0.1
    weight   = torch.randn(N,    dtype=dtype)
    return attn_out, x, weight


class TestHexagonKernelHW:
    """Calls rms_norm_hexagon on real QAIC hardware and compares against the FP32 reference.
    Skipped automatically on CPU-only hosts via conftest.py.
    """

    @pytest.fixture(autouse=True)
    def _import_kernel(self):
        from vllm_qaic._custom_ops import rms_norm_hexagon
        self._kernel = rms_norm_hexagon

    def _run(self, attn_out, x, weight, epsilon, qaic_device):
        normed, new_res = self._kernel(
            attn_out.to(qaic_device),
            x.to(qaic_device),
            weight.to(qaic_device),
            epsilon,
        )
        return normed.cpu(), new_res.cpu()

    @pytest.mark.parametrize("N", [64, 128, 256, 512, 1024, 3584, 4096])
    @pytest.mark.parametrize("M", [1, 2, 4, 16, 128])
    def test_fp16_shapes(self, M, N, qaic_device):
        """dst and residual match FP32 reference across common prefill/decode shapes."""
        attn_out, x, weight = _make_inputs(M, N, torch.float16)
        dst_ref, res_ref = _cast_ref(*_rms_norm_ref(attn_out, x, weight, 1e-6), torch.float16)
        dst_got, res_got = self._run(attn_out, x, weight, 1e-6, qaic_device)
        torch.testing.assert_close(dst_got, dst_ref, atol=5e-3, rtol=1e-2, msg=f"dst mismatch M={M} N={N}")
        torch.testing.assert_close(res_got, res_ref, atol=5e-3, rtol=1e-2, msg=f"residual mismatch M={M} N={N}")

    @pytest.mark.parametrize("scale", [1.0, 64.0, 256.0])
    def test_fp16_large_activations(self, scale, qaic_device):
        """Kernel must not produce NaN/Inf for large activations.
        Regression test for the overflow bug fixed in kernel.cpp (squares via QF32, not FP16).
        """
        M, N = 128, 4096
        torch.manual_seed(0)
        attn_out = torch.randn(M, N, dtype=torch.float16) * scale
        x        = torch.randn(M, N, dtype=torch.float16) * scale
        weight   = torch.ones(N, dtype=torch.float16)
        dst_got, res_got = self._run(attn_out, x, weight, 1e-6, qaic_device)
        assert not dst_got.isnan().any(), f"dst NaN at scale={scale}"
        assert not dst_got.isinf().any(), f"dst Inf at scale={scale}"
        assert not res_got.isnan().any(), f"residual NaN at scale={scale}"
        dst_ref, _ = _cast_ref(*_rms_norm_ref(attn_out, x, weight, 1e-6), torch.float16)
        torch.testing.assert_close(dst_got, dst_ref, atol=5e-3, rtol=1e-2, msg=f"dst mismatch at scale={scale}")

    @pytest.mark.parametrize("epsilon", [1e-5, 1e-6])
    def test_epsilon_variants(self, epsilon, qaic_device):
        """Both common epsilon values produce correct results."""
        M, N = 16, 3584
        attn_out, x, weight = _make_inputs(M, N, torch.float16)
        dst_ref, res_ref = _cast_ref(*_rms_norm_ref(attn_out, x, weight, epsilon), torch.float16)
        dst_got, res_got = self._run(attn_out, x, weight, epsilon, qaic_device)
        torch.testing.assert_close(dst_got, dst_ref, atol=5e-3, rtol=1e-2)
        torch.testing.assert_close(res_got, res_ref, atol=5e-3, rtol=1e-2)

    def test_decode_shapes(self, qaic_device):
        """Typical decode shapes (M=1,2,4) for Qwen2.5-VL-7B and Qwen3-8B match reference."""
        for M in [1, 2, 4]:
            for N in [3584, 4096]:
                attn_out, x, weight = _make_inputs(M, N, torch.float16, seed=M)
                dst_ref, res_ref = _cast_ref(*_rms_norm_ref(attn_out, x, weight, 1e-6), torch.float16)
                dst_got, res_got = self._run(attn_out, x, weight, 1e-6, qaic_device)
                torch.testing.assert_close(dst_got, dst_ref, atol=5e-3, rtol=1e-2, msg=f"dst mismatch M={M} N={N}")
                torch.testing.assert_close(res_got, res_ref, atol=5e-3, rtol=1e-2, msg=f"residual mismatch M={M} N={N}")

    def test_no_nan_all_zeros(self, qaic_device):
        """All-zero input must not NaN (epsilon guards the divide-by-zero)."""
        M, N = 4, 64
        attn_out = torch.zeros(M, N, dtype=torch.float16)
        x        = torch.zeros(M, N, dtype=torch.float16)
        weight   = torch.ones(N, dtype=torch.float16)
        dst_got, res_got = self._run(attn_out, x, weight, 1e-5, qaic_device)
        assert not dst_got.isnan().any(), "dst NaN for all-zero input"
        assert not dst_got.isinf().any(), "dst Inf for all-zero input"


class TestLayernormWrappers:
    """Tests QAicRMSNorm / QAicGemmaRMSNorm — the Python wrappers vLLM model runners call."""

    @pytest.fixture(autouse=True)
    def _vllm_config(self, vllm_config):
        pass

    @pytest.fixture(autouse=True)
    def _import_wrappers(self):
        from vllm_qaic.ops.layernorm import QAicRMSNorm, QAicGemmaRMSNorm
        self.QAicRMSNorm = QAicRMSNorm
        self.QAicGemmaRMSNorm = QAicGemmaRMSNorm

    @pytest.mark.parametrize("N", [3584, 4096])
    @pytest.mark.parametrize("M", [1, 16, 128])
    def test_qaic_rms_norm_with_residual(self, M, N, qaic_device):
        """forward_oot(x, residual) returns (normed, residual) matching the FP32 reference."""
        norm = self.QAicRMSNorm(N, eps=1e-6)
        norm.weight.data = torch.ones(N, dtype=torch.float16)
        torch.manual_seed(0)
        x        = torch.randn(M, N, dtype=torch.float16).to(qaic_device)
        residual = torch.randn(M, N, dtype=torch.float16).to(qaic_device)
        normed, new_res = norm.forward_oot(x, residual)
        normed  = normed.cpu()
        new_res = new_res.cpu()
        x_cpu = x.cpu()
        r_cpu = residual.cpu()
        dst_ref, res_ref = _cast_ref(
            *_rms_norm_ref(r_cpu, x_cpu, norm.weight.data.cpu(), 1e-6), torch.float16
        )
        torch.testing.assert_close(normed,  dst_ref, atol=5e-3, rtol=1e-2, msg=f"normed mismatch M={M} N={N}")
        torch.testing.assert_close(new_res, res_ref, atol=5e-3, rtol=1e-2, msg=f"residual mismatch M={M} N={N}")

    @pytest.mark.parametrize("N", [3584, 4096])
    def test_qaic_rms_norm_no_residual(self, N, qaic_device):
        """forward_oot(x) without residual returns plain F.rms_norm result."""
        norm = self.QAicRMSNorm(N, eps=1e-6)
        norm.weight.data = torch.ones(N, dtype=torch.float16)
        torch.manual_seed(1)
        x = torch.randn(16, N, dtype=torch.float16)
        out = norm.forward_oot(x)
        ref = F.rms_norm(x, [N], norm.weight.data, 1e-6)
        torch.testing.assert_close(out, ref, atol=5e-3, rtol=1e-2)

    @pytest.mark.parametrize("N", [3584, 4096])
    def test_qaic_gemma_rms_norm_with_residual(self, N, qaic_device):
        """GemmaRMSNorm with residual produces no NaN (effective weight = 1 + 0 = 1)."""
        norm = self.QAicGemmaRMSNorm(N, eps=1e-6)
        norm.weight.data = torch.zeros(N, dtype=torch.float16)
        torch.manual_seed(2)
        M = 16
        x        = torch.randn(M, N, dtype=torch.float16).to(qaic_device)
        residual = torch.randn(M, N, dtype=torch.float16).to(qaic_device)
        normed, new_res = norm.forward_oot(x, residual)
        assert not normed.cpu().isnan().any(),  "normed NaN in GemmaRMSNorm"
        assert not new_res.cpu().isnan().any(), "residual NaN in GemmaRMSNorm"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
