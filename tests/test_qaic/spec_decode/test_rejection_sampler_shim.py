# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Unit tests for the eager-mode speculative-decoding sampler shim.

Run with ``pytest -s`` so QAIC runtime diagnostics remain visible::

    .venv_eager/bin/python -m pytest -s \
        tests/test_qaic/spec_decode/test_rejection_sampler_shim.py -v
"""

import os
import subprocess
import sys
import textwrap
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_qaic.v1.sample.rejection_sampler_shim import (
    _GridLaunchable,
    _sample_recovered_tokens_kernel_pyt,
    generate_uniform_probs,
    install,
)


def _has_qualcomm_triton_backend() -> bool:
    try:
        from triton.backends import backends

        return "qcom_hexagon_backend" in backends
    except Exception:
        return False


def _run_selector_subprocess(
    implementation: str | None, source: str
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    if implementation is None:
        environment.pop("VLLM_QAIC_REJECTION_SAMPLER_IMPL", None)
    else:
        environment["VLLM_QAIC_REJECTION_SAMPLER_IMPL"] = implementation
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=False,
        cwd=os.getcwd(),
        env=environment,
        text=True,
        capture_output=True,
    )


def _available_devices() -> list[str]:
    devices = ["cpu"]
    try:
        import torch_qaic  # noqa: F401

        if torch.qaic.device_count() > 0:
            devices.append("qaic")
    except Exception:
        pass
    return devices


@pytest.fixture(params=_available_devices())
def device(request) -> torch.device:
    return torch.device(request.param)


@pytest.fixture
def qaic_device() -> torch.device:
    try:
        import torch_qaic  # noqa: F401

        if torch.qaic.device_count() > 0:
            return torch.device("qaic")
    except Exception:
        pass
    pytest.skip("Requires a visible QAIC device and torch_qaic")


def test_grid_launchable_forwards_args_and_discards_grid():
    seen = []

    def fn(first, second, keyword=None):
        seen.append((first, second, keyword))
        return "result"

    wrapped = _GridLaunchable(fn)

    assert wrapped[(4,)](1, 2, keyword=3) == "result"
    assert wrapped(10, 20, keyword=30) == "result"
    assert seen == [(1, 2, 3), (10, 20, 30)]
    assert wrapped.__name__ == "fn"


@pytest.mark.skipif(
    not _has_qualcomm_triton_backend(),
    reason="Qualcomm Triton backend is not installed",
)
@pytest.mark.parametrize("probability_dtype", [torch.float32, torch.float16])
def test_triton_random_kernel_handoffs_pytorch_recovered_ids_on_qaic(
    qaic_device, probability_dtype
):
    """Exercise the retained PyTorch producer with the Triton consumer."""
    from vllm_qaic.v1.sample.rejection_sampler_triton import (
        get_qaic_triton_kernels,
    )

    max_spec_len = 3
    vocab_size = 8
    draft_cpu = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    cu_tokens_cpu = torch.tensor([0, 2, 5], dtype=torch.int32)
    bonus_cpu = torch.tensor([7, 6, 5], dtype=torch.int32)
    is_greedy_cpu = torch.tensor([True, False, False], dtype=torch.bool)
    uniform_cpu = torch.ones(5, dtype=torch.float32)
    target_cpu = torch.full((5, vocab_size), 0.01, dtype=probability_dtype)
    for token_index, token_id in enumerate(draft_cpu.tolist()):
        target_cpu[token_index, token_id] = 0.75
        target_cpu[token_index, (token_id + 1) % vocab_size] = 0.9
    inv_q_cpu = torch.ones((3, vocab_size), dtype=torch.float32)

    expected_recovered = torch.tensor([2, 3, 4, 5, 6], dtype=torch.int32)
    expected_output = torch.tensor(
        [[-1, -1, -1, -1], [2, -1, -1, -1], [4, -1, -1, -1]],
        dtype=torch.int32,
    )
    draft = draft_cpu.to(qaic_device).contiguous()
    cu_tokens = cu_tokens_cpu.to(qaic_device)
    target_probs = target_cpu.to(qaic_device)
    recovered = torch.full((5,), -1, dtype=torch.int32, device=qaic_device)
    output = torch.full(
        (3, max_spec_len + 1), -1, dtype=torch.int32, device=qaic_device
    )
    _sample_recovered_tokens_kernel_pyt(
        recovered,
        cu_tokens,
        draft,
        None,
        target_probs,
        inv_q_cpu.to(qaic_device),
        vocab_size,
        BLOCK_SIZE=8,
        NO_DRAFT_PROBS=True,
        USE_FP64_GUMBEL=False,
    )

    triton_kernels = get_qaic_triton_kernels()
    triton_kernels.rejection_random_sample_kernel[(3,)](
        output,
        cu_tokens,
        draft,
        None,
        target_probs,
        bonus_cpu.to(qaic_device),
        recovered,
        uniform_cpu.to(qaic_device),
        is_greedy_cpu.to(qaic_device),
        max_spec_len,
        vocab_size,
        None,
        NO_DRAFT_PROBS=True,
        SYNTHETIC_MODE=False,
    )
    torch.qaic.synchronize()

    assert draft.dtype == torch.int32
    assert draft.device.type == "qaic"
    assert draft.is_contiguous()
    assert recovered.dtype == torch.int32
    assert recovered.device.type == "qaic"
    assert recovered.is_contiguous()
    assert torch.equal(recovered.cpu(), expected_recovered)
    assert output.dtype == torch.int32
    assert output.device.type == "qaic"
    assert torch.equal(output.cpu(), expected_output)


def test_recovered_tokens_mask_draft_column(device):
    target_probs = torch.tensor(
        [[0.1, 0.1, 0.9, 0.6, 0.2], [0.9, 0.2, 0.3, 0.1, 0.4]],
        device=device,
    )
    output = torch.zeros(2, dtype=torch.int32, device=device)

    _sample_recovered_tokens_kernel_pyt(
        output,
        torch.tensor([1, 2], device=device),
        torch.tensor([2, 0], device=device),
        None,
        target_probs,
        torch.ones((2, 5), device=device),
        5,
        NO_DRAFT_PROBS=True,
        USE_FP64_GUMBEL=False,
    )

    assert output.cpu().tolist() == [3, 4]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_recovered_tokens_masks_int64_boundary_ids(device, dtype, noncontiguous):
    target_probs = torch.tensor(
        [
            [0.9, 0.8, 0.1, 0.2, 0.3],
            [0.5, 0.1, 0.2, 0.3, 0.95],
            [0.1, 0.2, 0.9, 0.8, 0.3],
        ],
        dtype=dtype,
    )
    if noncontiguous:
        target_probs = target_probs.t().contiguous().t()

    output = torch.zeros(3, dtype=torch.int32, device=device)
    _sample_recovered_tokens_kernel_pyt(
        output,
        torch.tensor([1, 3], device=device),
        torch.tensor([0, 4, 2], dtype=torch.int64, device=device),
        None,
        target_probs.to(device),
        torch.ones((2, 5), dtype=dtype, device=device),
        5,
        NO_DRAFT_PROBS=True,
        USE_FP64_GUMBEL=False,
    )

    assert output.device.type == device.type
    assert output.cpu().tolist() == [1, 0, 3]


def test_recovered_tokens_uses_clamped_target_minus_draft(device):
    output = torch.zeros(1, dtype=torch.int32, device=device)

    _sample_recovered_tokens_kernel_pyt(
        output,
        torch.tensor([1], device=device),
        torch.tensor([0], device=device),
        torch.tensor([[0.0, 0.5, 0.1, 0.0]], device=device),
        torch.tensor([[0.1, 0.2, 0.7, 0.0]], device=device),
        torch.ones((1, 4), device=device),
        4,
        NO_DRAFT_PROBS=False,
        USE_FP64_GUMBEL=False,
    )

    assert output.cpu().tolist() == [2]


def test_recovered_tokens_skips_zero_draft_requests(device):
    output = torch.full((3,), -7, dtype=torch.int32, device=device)

    _sample_recovered_tokens_kernel_pyt(
        output,
        torch.tensor([1, 1, 3], device=device),
        torch.tensor([0, 4, 2], dtype=torch.int64, device=device),
        None,
        torch.tensor(
            [
                [0.9, 0.8, 0.1, 0.2, 0.3],
                [0.5, 0.1, 0.2, 0.3, 0.95],
                [0.1, 0.2, 0.9, 0.8, 0.3],
            ],
            device=device,
        ),
        torch.ones((3, 5), device=device),
        5,
        NO_DRAFT_PROBS=True,
        USE_FP64_GUMBEL=False,
    )

    assert output.cpu().tolist() == [1, 0, 3]


def test_generate_uniform_probs_uses_float32(device):
    probabilities = generate_uniform_probs(6, [2, 4], {}, device)

    assert probabilities.shape == (6,)
    assert probabilities.dtype == torch.float32
    assert bool(((probabilities.cpu() >= 0) & (probabilities.cpu() <= 1)).all())


def test_generate_uniform_probs_honors_generators():
    first = torch.Generator(device="cpu").manual_seed(1234)
    second = torch.Generator(device="cpu").manual_seed(1234)

    first_probs = generate_uniform_probs(4, [4], {0: first}, torch.device("cpu"))
    second_probs = generate_uniform_probs(4, [4], {0: second}, torch.device("cpu"))

    assert torch.equal(first_probs, second_probs)


def test_native_temperature_matches_cpu_reference(device):
    import vllm.v1.sample.sampler as sampler_module

    logits = torch.tensor(
        [[2.0, 4.0], [1.0, 3.0], [5.0, 7.0]],
        dtype=torch.float32,
    )
    temperatures = torch.tensor([0.0, 0.5, 2.0], dtype=torch.float32)
    expected_temperatures = torch.where(
        temperatures < sampler_module._SAMPLING_EPS,
        torch.ones_like(temperatures),
        temperatures,
    )
    expected = logits / expected_temperatures.unsqueeze(dim=1)

    actual = logits.to(device).clone()
    sampler_module.Sampler.apply_temperature(
        actual,
        temperatures.to(device),
        all_random=False,
    )

    assert actual.device.type == device.type
    assert torch.equal(actual.cpu(), expected)


def test_native_top_k_matches_cpu_reference(device):
    import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_module

    logits = torch.tensor(
        [[0.1, 0.3, 0.2, 0.4], [1.0, 0.5, 0.7, 0.9]],
        dtype=torch.float32,
    )
    top_k = torch.tensor([2, 3], dtype=torch.int64)
    expected = topk_topp_module.apply_top_k_top_p_pytorch(logits.clone(), top_k, None)

    actual = topk_topp_module.apply_top_k_top_p_pytorch(
        logits.to(device).clone(), top_k.to(device), None
    )

    assert actual.device.type == device.type
    assert torch.equal(actual.cpu(), expected)


def test_native_mixed_greedy_random_selection_matches_cpu_reference(device):
    import vllm.v1.sample.sampler as sampler_module

    class FixedTopKTopPSampler(torch.nn.Module):
        def __init__(self, sampled: torch.Tensor):
            super().__init__()
            self.register_buffer("sampled", sampled)

        def forward(self, logits, generators, top_k, top_p):
            return self.sampled, None

    logits = torch.tensor([[0.1, 0.7, 0.2], [0.4, 0.3, 0.9]], dtype=torch.float32)
    temperatures = torch.tensor([0.0, 0.5], dtype=torch.float32)
    random_sampled = torch.tensor([0, 1], dtype=torch.int64)
    expected = torch.tensor([1, 1], dtype=torch.int64)
    sampling_metadata = SimpleNamespace(
        all_greedy=False,
        all_random=False,
        temperature=temperatures.to(device),
        logitsprocs=SimpleNamespace(argmax_invariant=[]),
        generators={},
        top_k=None,
        top_p=None,
    )
    sampler = sampler_module.Sampler()
    sampler.topk_topp_sampler = FixedTopKTopPSampler(random_sampled.to(device))

    actual, processed_logprobs = sampler.sample(
        logits.to(device).clone(), sampling_metadata
    )

    assert processed_logprobs is None
    assert actual.device.type == device.type
    assert torch.equal(actual.cpu(), expected)


def test_native_top_p_matches_cpu_predicate(device):
    import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_module

    logits = torch.tensor(
        [[-2.0, -1.0, 0.0, 0.5, 1.0], [-1.5, -0.5, 0.0, 0.7, 1.5]],
        dtype=torch.float16,
        device=device,
    )
    top_p = torch.tensor([0.6, 0.8], dtype=torch.float32, device=device)
    expected = topk_topp_module.apply_top_k_top_p_pytorch(
        logits.cpu().clone(), None, top_p.cpu()
    )
    actual = topk_topp_module.apply_top_k_top_p_pytorch(logits.clone(), None, top_p)

    assert actual.device.type == device.type
    assert torch.equal(actual.cpu(), expected.cpu())


def test_install_patches_rejection_sampler_idempotently():
    import vllm.v1.sample.rejection_sampler as rejection_sampler
    import vllm.v1.sample.sampler as sampler_module
    import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_module
    import vllm_qaic.v1.sample.rejection_sampler_shim as shim
    from vllm_qaic.v1.sample import rejection_sampler_triton as triton_sampler

    original_apply_temperature = sampler_module.Sampler.apply_temperature
    original_sample = sampler_module.Sampler.sample
    original_top_k_top_p = topk_topp_module.apply_top_k_top_p_pytorch
    install()
    assert shim._selected_implementation() == "triton"
    assert rejection_sampler.expand_kernel is triton_sampler.expand_kernel
    assert (
        rejection_sampler.rejection_greedy_sample_kernel
        is triton_sampler.rejection_greedy_sample_kernel
    )
    assert (
        rejection_sampler.rejection_random_sample_kernel
        is triton_sampler.rejection_random_sample_kernel
    )
    assert (
        rejection_sampler.sample_recovered_tokens_kernel
        is shim.sample_recovered_tokens_kernel
    )
    assert rejection_sampler.generate_uniform_probs is shim.generate_uniform_probs
    assert sampler_module.Sampler.apply_temperature is original_apply_temperature
    assert sampler_module.Sampler.sample is original_sample
    assert topk_topp_module.apply_top_k_top_p_pytorch is original_top_k_top_p

    install()
    assert rejection_sampler.expand_kernel is triton_sampler.expand_kernel
    for deleted_kernel in (
        "_expand_kernel_pyt",
        "_rejection_greedy_sample_kernel_pyt",
        "_rejection_random_sample_kernel_pyt",
    ):
        assert not hasattr(shim, deleted_kernel)


@pytest.mark.skipif(
    not _has_qualcomm_triton_backend(),
    reason="Qualcomm Triton backend is not installed",
)
@pytest.mark.parametrize("implementation", [None, "hybrid"])
def test_default_and_legacy_selector_patch_triton_objects_in_subprocess(
    implementation,
):
    result = _run_selector_subprocess(
        implementation,
        """
        import vllm.v1.sample.rejection_sampler as rejection_sampler
        import vllm_qaic.v1.sample.rejection_sampler_shim as shim
        from vllm_qaic.v1.sample import rejection_sampler_triton as triton_sampler

        shim.install()
        assert shim._selected_implementation() == "triton"
        assert rejection_sampler.expand_kernel is triton_sampler.expand_kernel
        assert (rejection_sampler.rejection_greedy_sample_kernel
                is triton_sampler.rejection_greedy_sample_kernel)
        assert (rejection_sampler.rejection_random_sample_kernel
                is triton_sampler.rejection_random_sample_kernel)
        assert (rejection_sampler.sample_recovered_tokens_kernel
                is shim.sample_recovered_tokens_kernel)
        assert rejection_sampler.generate_uniform_probs is shim.generate_uniform_probs
        """,
    )

    assert result.returncode == 0, result.stderr


def test_pytorch_selector_fails_actionably_in_subprocess():
    result = _run_selector_subprocess(
        "pytorch",
        """
        import vllm_qaic.v1.sample.rejection_sampler_shim as shim

        shim.install()
        """,
    )

    assert result.returncode != 0
    assert "pytorch is not a valid QAIC PYT setting" in result.stderr
    assert "legacy alias" in result.stderr


def test_invalid_selector_fails_actionably_in_subprocess():
    result = _run_selector_subprocess(
        "invalid",
        """
        import vllm_qaic.v1.sample.rejection_sampler_shim as shim

        shim.install()
        """,
    )

    assert result.returncode != 0
    assert "must be unset or 'hybrid'" in result.stderr


def test_legacy_alias_can_be_selected_after_default_install_in_subprocess():
    result = _run_selector_subprocess(
        None,
        """
        import os
        import vllm_qaic.v1.sample.rejection_sampler_shim as shim

        shim.install()
        os.environ["VLLM_QAIC_REJECTION_SAMPLER_IMPL"] = "hybrid"
        shim.install()
        """,
    )

    assert result.returncode == 0, result.stderr


def test_default_selector_reports_unavailable_backend(monkeypatch):
    import vllm_qaic.v1.sample.rejection_sampler_shim as shim

    unavailable_backend = ModuleType("vllm_qaic.v1.sample.rejection_sampler_triton")

    def get_qaic_triton_kernels():
        raise RuntimeError("Qualcomm Triton backend is unavailable")

    unavailable_backend.get_qaic_triton_kernels = get_qaic_triton_kernels
    monkeypatch.setitem(
        sys.modules,
        "vllm_qaic.v1.sample.rejection_sampler_triton",
        unavailable_backend,
    )
    monkeypatch.delenv("VLLM_QAIC_REJECTION_SAMPLER_IMPL", raising=False)
    monkeypatch.setattr(shim, "_shim_installed", False)
    monkeypatch.setattr(shim, "_shim_mode", None)

    with pytest.raises(RuntimeError, match="qcom_hexagon_backend"):
        shim.install()
