"""Unit tests for the eager-mode speculative-decoding sampler shim.

Run with ``pytest -s`` so QAIC runtime diagnostics remain visible::

    .venv_eager/bin/python -m pytest -s \
        tests/test_qaic/spec_decode/test_rejection_sampler_shim.py -v
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_qaic.v1.sample.rejection_sampler_shim import (
    _GridLaunchable,
    _expand_kernel_pyt,
    _rejection_greedy_sample_kernel_pyt,
    _rejection_random_sample_kernel_pyt,
    _sample_recovered_tokens_kernel_pyt,
    generate_uniform_probs,
    install,
)

PLACEHOLDER_TOKEN_ID = -1


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


def test_expand_kernel_broadcasts_and_replaces(device):
    inputs = torch.tensor([0, 5, 7], device=device)
    cumulative_tokens = torch.tensor([2, 5, 6], device=device)
    output = torch.zeros(6, dtype=torch.long, device=device)

    _expand_kernel_pyt(
        output,
        inputs,
        cumulative_tokens,
        replace_from=0,
        replace_to=99,
    )

    assert output.cpu().tolist() == [99, 99, 5, 5, 5, 7]


def test_expand_kernel_empty_batch(device):
    output = torch.zeros(0, dtype=torch.long, device=device)
    inputs = torch.zeros(0, dtype=torch.long, device=device)
    cumulative_tokens = torch.zeros(0, dtype=torch.long, device=device)

    _expand_kernel_pyt(output, inputs, cumulative_tokens, 0, 0)

    assert output.cpu().tolist() == []


def _output(batch_size: int, max_spec_len: int, device: torch.device):
    return torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )


def test_greedy_accepts_all_and_appends_bonus(device):
    output = _output(1, 3, device)
    cumulative_tokens = torch.tensor([2], device=device)
    draft_tokens = torch.tensor([5, 5], device=device)
    target_argmax = torch.tensor([5, 5], device=device)
    bonus_tokens = torch.tensor([99], device=device)

    _rejection_greedy_sample_kernel_pyt(
        output,
        cumulative_tokens,
        draft_tokens,
        target_argmax,
        bonus_tokens,
        None,
        3,
        None,
        None,
        SYNTHETIC_MODE=False,
    )

    assert output.cpu().tolist() == [[5, 5, 99, PLACEHOLDER_TOKEN_ID]]


def test_greedy_stops_at_first_mismatch(device):
    output = _output(1, 3, device)
    cumulative_tokens = torch.tensor([2], device=device)
    draft_tokens = torch.tensor([7, 8], device=device)
    target_argmax = torch.tensor([9, 8], device=device)
    bonus_tokens = torch.tensor([99], device=device)

    _rejection_greedy_sample_kernel_pyt(
        output,
        cumulative_tokens,
        draft_tokens,
        target_argmax,
        bonus_tokens,
        None,
        3,
        None,
        None,
        SYNTHETIC_MODE=False,
    )

    assert output.cpu().tolist() == [[9, -1, -1, -1]]


def test_greedy_handles_multiple_requests(device):
    output = _output(2, 3, device)
    cumulative_tokens = torch.tensor([2, 4], device=device)
    draft_tokens = torch.tensor([5, 5, 7, 8], device=device)
    target_argmax = torch.tensor([5, 5, 9, 8], device=device)
    bonus_tokens = torch.tensor([99, 88], device=device)

    _rejection_greedy_sample_kernel_pyt(
        output,
        cumulative_tokens,
        draft_tokens,
        target_argmax,
        bonus_tokens,
        None,
        3,
        None,
        None,
        SYNTHETIC_MODE=False,
    )

    assert output.cpu().tolist() == [
        [5, 5, 99, -1],
        [9, -1, -1, -1],
    ]


def test_random_accepts_all_when_uniform_is_zero(device):
    output = _output(1, 3, device)
    cumulative_tokens = torch.tensor([2], device=device)
    draft_tokens = torch.tensor([1, 2], device=device)
    target_probs = torch.full((2, 4), 0.5, device=device)
    bonus_tokens = torch.tensor([99], device=device)
    recovered_tokens = torch.zeros(2, dtype=torch.int32, device=device)
    uniform_probs = torch.zeros(2, device=device)
    is_greedy = torch.tensor([False], device=device)

    _rejection_random_sample_kernel_pyt(
        output,
        cumulative_tokens,
        draft_tokens,
        None,
        target_probs,
        bonus_tokens,
        recovered_tokens,
        uniform_probs,
        is_greedy,
        3,
        4,
        None,
        NO_DRAFT_PROBS=True,
        SYNTHETIC_MODE=False,
    )

    assert output.cpu().tolist() == [[1, 2, 99, -1]]


def test_random_recovers_after_rejection(device):
    output = _output(1, 3, device)
    cumulative_tokens = torch.tensor([2], device=device)
    draft_tokens = torch.tensor([1, 2], device=device)
    target_probs = torch.full((2, 4), 0.5, device=device)
    bonus_tokens = torch.tensor([99], device=device)
    recovered_tokens = torch.tensor([42, 43], device=device)
    uniform_probs = torch.ones(2, device=device)
    is_greedy = torch.tensor([False], device=device)

    _rejection_random_sample_kernel_pyt(
        output,
        cumulative_tokens,
        draft_tokens,
        None,
        target_probs,
        bonus_tokens,
        recovered_tokens,
        uniform_probs,
        is_greedy,
        3,
        4,
        None,
        NO_DRAFT_PROBS=True,
        SYNTHETIC_MODE=False,
    )

    assert output.cpu().tolist() == [[42, -1, -1, -1]]


def test_random_skips_greedy_requests(device):
    output = _output(1, 3, device)
    cumulative_tokens = torch.tensor([2], device=device)
    draft_tokens = torch.tensor([1, 2], device=device)
    target_probs = torch.full((2, 4), 0.5, device=device)
    bonus_tokens = torch.tensor([99], device=device)
    recovered_tokens = torch.zeros(2, dtype=torch.int32, device=device)
    uniform_probs = torch.zeros(2, device=device)
    is_greedy = torch.tensor([True], device=device)

    _rejection_random_sample_kernel_pyt(
        output,
        cumulative_tokens,
        draft_tokens,
        None,
        target_probs,
        bonus_tokens,
        recovered_tokens,
        uniform_probs,
        is_greedy,
        3,
        4,
        None,
        NO_DRAFT_PROBS=True,
        SYNTHETIC_MODE=False,
    )

    assert output.cpu().tolist() == [[-1, -1, -1, -1]]


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
    output = torch.full((1,), -7, dtype=torch.int32, device=device)

    _sample_recovered_tokens_kernel_pyt(
        output,
        torch.tensor([0, 1], device=device),
        torch.tensor([3], device=device),
        None,
        torch.tensor([[0.5, 0.2, 0.9, 0.1]], device=device),
        torch.ones((2, 4), device=device),
        4,
        NO_DRAFT_PROBS=True,
        USE_FP64_GUMBEL=False,
    )

    assert output.cpu().tolist() == [2]


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
    expected = topk_topp_module.apply_top_k_top_p_pytorch(
        logits.clone(), top_k, None
    )

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

    logits = torch.tensor(
        [[0.1, 0.7, 0.2], [0.4, 0.3, 0.9]], dtype=torch.float32
    )
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


def test_top_p_patch_preserves_cpu_predicate_for_qaic(device):
    if device.type != "qaic":
        pytest.skip("Requires QAIC eager CPU fallback")

    import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_module

    def apply_top_p_with_cpu_predicate(
        logits: torch.Tensor, top_p: torch.Tensor
    ) -> torch.Tensor:
        logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = torch.le(
            probs_sum.cpu(), (1 - top_p.unsqueeze(dim=1)).cpu()
        ).to(device=logits.device)
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))
        return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)

    install()
    logits = torch.tensor(
        [[-2.0, -1.0, 0.0, 0.5, 1.0], [-1.5, -0.5, 0.0, 0.7, 1.5]],
        dtype=torch.float16,
        device=device,
    )
    top_p = torch.tensor([0.6, 0.8], dtype=torch.float32, device=device)
    expected = apply_top_p_with_cpu_predicate(logits.clone(), top_p)
    actual = topk_topp_module.apply_top_k_top_p_pytorch(
        logits.clone(), None, top_p
    )

    assert getattr(topk_topp_module, "_qaic_top_p_comparison_patched", False)
    assert torch.equal(actual.cpu(), expected.cpu())


def test_install_patches_rejection_sampler_idempotently():
    import vllm.v1.sample.rejection_sampler as rejection_sampler
    import vllm.v1.sample.sampler as sampler_module
    import vllm_qaic.v1.sample.rejection_sampler_shim as shim

    original_apply_temperature = sampler_module.Sampler.apply_temperature
    original_sample = sampler_module.Sampler.sample
    install()
    assert rejection_sampler.expand_kernel is shim.expand_kernel
    assert (
        rejection_sampler.rejection_greedy_sample_kernel
        is shim.rejection_greedy_sample_kernel
    )
    assert (
        rejection_sampler.rejection_random_sample_kernel
        is shim.rejection_random_sample_kernel
    )
    assert (
        rejection_sampler.sample_recovered_tokens_kernel
        is shim.sample_recovered_tokens_kernel
    )
    assert rejection_sampler.generate_uniform_probs is shim.generate_uniform_probs
    assert sampler_module.Sampler.apply_temperature is original_apply_temperature
    assert sampler_module.Sampler.sample is original_sample

    install()
    assert rejection_sampler.expand_kernel is shim.expand_kernel
