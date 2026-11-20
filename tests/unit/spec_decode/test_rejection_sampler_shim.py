# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Unit tests for the eager-mode speculative-decoding rejection-sampler shim.

The shim only rebinds ``generate_uniform_probs`` (fp32); every other rejection
kernel is upstream's own ``@triton.jit`` kernel running on QAIC tensors. The
``*_on_qaic`` tests below launch those upstream kernels directly on hardware and
compare against a PyTorch CPU reference.

Run with ``pytest -s`` so QAIC runtime diagnostics remain visible. The Hexagon
toolchain env vars (``HEXAGON_TOOLS``, ``HEXAGON_SDK_ROOT``) must be exported
for the Triton-compiling tests::

    .venv_eager/bin/python -m pytest -s \
        tests/unit/spec_decode/test_rejection_sampler_shim.py -v
"""

from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm  # noqa: F401  (must precede vllm_qaic: circular import)
from vllm_qaic.v1.sample.rejection_sampler_shim import (
    generate_uniform_probs,
    install,
)


def _has_qualcomm_triton_backend() -> bool:
    try:
        from triton.backends import backends

        return "qcom_hexagon_backend" in backends
    except Exception:
        return False


requires_qualcomm_triton = pytest.mark.skipif(
    not _has_qualcomm_triton_backend(),
    reason="Qualcomm Triton backend is not installed",
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


def test_generate_uniform_probs_uses_float32(device):
    probabilities = generate_uniform_probs(6, [2, 4], {}, device)

    assert probabilities.shape == (6,)
    assert probabilities.dtype == torch.float32
    # Strictly positive: u == 0 would accept zero-probability draft tokens.
    assert bool(((probabilities.cpu() > 0) & (probabilities.cpu() < 1)).all())


def test_generate_uniform_probs_never_returns_zero(monkeypatch):
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.zeros(*args[0]))

    probabilities = generate_uniform_probs(3, [3], {}, torch.device("cpu"))

    assert bool((probabilities > 0).all())


def test_generate_uniform_probs_honors_generators():
    # Generators are keyed by request index; zero-draft rows occupy no slots
    # and must not consume RNG from any generator.
    cpu = torch.device("cpu")
    first = generate_uniform_probs(
        5,
        [2, 0, 3],
        {
            0: torch.Generator(device="cpu").manual_seed(1),
            2: torch.Generator(device="cpu").manual_seed(2),
        },
        cpu,
    )

    expected_first = torch.empty(2).uniform_(
        generator=torch.Generator(device="cpu").manual_seed(1)
    )
    expected_last = torch.empty(3).uniform_(
        generator=torch.Generator(device="cpu").manual_seed(2)
    )
    assert torch.equal(first[:2], expected_first)
    assert torch.equal(first[2:], expected_last)


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


def test_install_rebinds_only_generate_uniform_probs(monkeypatch):
    import vllm.v1.sample.rejection_sampler as rejection_sampler
    import vllm.v1.sample.sampler as sampler_module
    import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_module
    import vllm_qaic.v1.sample.rejection_sampler_shim as shim

    monkeypatch.setattr(shim, "_shim_installed", False)
    kernel_names = (
        "expand_kernel",
        "rejection_greedy_sample_kernel",
        "rejection_random_sample_kernel",
        "sample_recovered_tokens_kernel",
    )
    originals = {name: getattr(rejection_sampler, name) for name in kernel_names}
    original_apply_temperature = sampler_module.Sampler.apply_temperature
    original_sample = sampler_module.Sampler.sample
    original_top_k_top_p = topk_topp_module.apply_top_k_top_p_pytorch
    monkeypatch.setattr(
        rejection_sampler,
        "generate_uniform_probs",
        rejection_sampler.generate_uniform_probs,
    )

    install()
    assert rejection_sampler.generate_uniform_probs is shim.generate_uniform_probs
    for name, original in originals.items():
        assert getattr(rejection_sampler, name) is original
    assert sampler_module.Sampler.apply_temperature is original_apply_temperature
    assert sampler_module.Sampler.sample is original_sample
    assert topk_topp_module.apply_top_k_top_p_pytorch is original_top_k_top_p

    install()  # idempotent
    assert rejection_sampler.generate_uniform_probs is shim.generate_uniform_probs


def test_require_qualcomm_triton_reports_unavailable_backend(monkeypatch):
    import vllm_qaic.v1.sample.rejection_sampler_shim as shim

    triton_backends = ModuleType("triton.backends")
    triton_backends.backends = {}
    monkeypatch.setitem(__import__("sys").modules, "triton.backends", triton_backends)

    with pytest.raises(RuntimeError, match="qcom_hexagon_backend"):
        shim.require_qualcomm_triton()


# ---------------------------------------------------------------------------
# Upstream (unshimmed) Triton kernels launched directly on QAIC tensors.
# ---------------------------------------------------------------------------

_MAX_SPEC_LEN = 5


def _random_inputs(vocab_size, batch_size, use_draft_probs, seed):
    generator = torch.Generator().manual_seed(seed)
    drafts = torch.randint(
        0, _MAX_SPEC_LEN + 1, (batch_size,), generator=generator
    ).tolist()
    # Mix zero-draft rows into the batch, but keep one full-length row and (when
    # possible) one single-draft row so the accept and bonus paths are reachable.
    drafts[0] = _MAX_SPEC_LEN
    if batch_size > 1:
        drafts[1] = 1
    num_tokens = sum(drafts)
    cu_num_draft_tokens = torch.tensor(drafts).cumsum(0).to(torch.int32)
    draft_token_ids = torch.randint(
        0, vocab_size, (num_tokens,), dtype=torch.int32, generator=generator
    )
    target_probs = torch.softmax(
        torch.randn(num_tokens, vocab_size, generator=generator) * 2, -1
    )
    draft_probs = (
        torch.softmax(torch.randn(num_tokens, vocab_size, generator=generator) * 2, -1)
        if use_draft_probs
        else None
    )
    # Boost every other draft token so both accept and reject paths are hit.
    for token in range(0, num_tokens, 2):
        draft = int(draft_token_ids[token])
        target_probs[token, draft] = max(float(target_probs[token, draft]), 0.5)
    target_probs = target_probs / target_probs.sum(-1, keepdim=True)
    return SimpleNamespace(
        drafts=drafts,
        num_tokens=num_tokens,
        cu=cu_num_draft_tokens,
        draft_ids=draft_token_ids,
        target_probs=target_probs,
        draft_probs=draft_probs,
        bonus=torch.randint(
            0, vocab_size, (batch_size,), dtype=torch.int64, generator=generator
        ),
        uniform=torch.rand(num_tokens, generator=generator).clamp_min(1e-6),
        inv_q=torch.empty(batch_size, vocab_size)
        .exponential_(generator=generator)
        .reciprocal(),
    )


def _draft_matrix(inputs):
    matrix = torch.full((len(inputs.drafts), _MAX_SPEC_LEN), -2, dtype=torch.int32)
    position = 0
    for row, count in enumerate(inputs.drafts):
        matrix[row, :count] = inputs.draft_ids[position : position + count]
        position += count
    return matrix


def _reference_recovered(inputs):
    recovered = torch.empty(inputs.num_tokens, dtype=torch.int32)
    position = 0
    for request, count in enumerate(inputs.drafts):
        for token in range(position, position + count):
            if inputs.draft_probs is not None:
                probs = (
                    inputs.target_probs[token] - inputs.draft_probs[token]
                ).clamp_min(0)
            else:
                probs = inputs.target_probs[token].clone()
                probs[inputs.draft_ids[token]] = 0
            recovered[token] = (probs * inputs.inv_q[request]).argmax()
        position += count
    return recovered


def _reference_random(inputs, recovered):
    batch_size = len(inputs.drafts)
    output = torch.full((batch_size, _MAX_SPEC_LEN + 1), -1, dtype=torch.int32)
    position = 0
    for request, count in enumerate(inputs.drafts):
        rejected = False
        for offset in range(count):
            token = position + offset
            draft = int(inputs.draft_ids[token])
            draft_prob = (
                float(inputs.draft_probs[token, draft])
                if inputs.draft_probs is not None
                else 1.0
            )
            target_prob = float(inputs.target_probs[token, draft])
            if draft_prob > 0 and target_prob / draft_prob >= float(
                inputs.uniform[token]
            ):
                output[request, offset] = draft
            else:
                output[request, offset] = recovered[token]
                rejected = True
                break
        if not rejected:
            output[request, count] = inputs.bonus[request]
        position += count
    return output


@requires_qualcomm_triton
def test_upstream_expand_kernel_on_qaic(qaic_device):
    import vllm.v1.sample.rejection_sampler as rejection_sampler

    cu_num_tokens = torch.tensor([2, 5, 6], dtype=torch.int32)
    values = torch.tensor([7, 8, 9], dtype=torch.int32)
    output = torch.empty(6, dtype=torch.int32, device=qaic_device)
    rejection_sampler._expand(
        output, values.to(qaic_device), cu_num_tokens.to(qaic_device), -1, -1
    )
    assert output.cpu().tolist() == [7, 7, 8, 8, 8, 9]


@requires_qualcomm_triton
def test_upstream_greedy_kernel_on_qaic(qaic_device):
    import vllm.v1.sample.rejection_sampler as rejection_sampler

    cu_num_tokens = torch.tensor([2, 5, 6], dtype=torch.int32)
    draft = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
    target_argmax = draft.clone()
    target_argmax[1] = 99  # reject request 0 at its second draft token
    bonus = torch.tensor([20, 21, 22], dtype=torch.int64)
    output = torch.full((3, _MAX_SPEC_LEN + 1), -1, dtype=torch.int32)
    output_device = output.to(qaic_device)
    rejection_sampler._rejection_greedy_sample(
        output_device,
        cu_num_tokens.to(qaic_device),
        draft.to(qaic_device),
        target_argmax.to(qaic_device),
        bonus.to(qaic_device),
        torch.ones(3, dtype=torch.bool).to(qaic_device),
        _MAX_SPEC_LEN,
        None,
        None,
    )
    expected = torch.full((3, _MAX_SPEC_LEN + 1), -1, dtype=torch.int32)
    expected[0, :2] = torch.tensor([10, 99])
    expected[1, :4] = torch.tensor([12, 13, 14, 21])
    expected[2, :2] = torch.tensor([15, 22])
    assert torch.equal(output_device.cpu(), expected)


@requires_qualcomm_triton
@pytest.mark.parametrize(
    "vocab_size,batch_size,use_draft_probs",
    [(32000, 8, False), (32000, 8, True), (128256, 16, True), (32000, 1, False)],
)
def test_upstream_random_and_recovered_kernels_on_qaic(
    qaic_device, vocab_size, batch_size, use_draft_probs
):
    import vllm.v1.sample.rejection_sampler as rejection_sampler

    inputs = _random_inputs(vocab_size, batch_size, use_draft_probs, seed=vocab_size)
    to_device = lambda t: None if t is None else t.to(qaic_device)  # noqa: E731

    recovered_ref = _reference_recovered(inputs)
    recovered = torch.empty(inputs.num_tokens, dtype=torch.int32, device=qaic_device)
    rejection_sampler._sample_recovered_tokens(
        recovered,
        to_device(inputs.cu),
        to_device(inputs.draft_ids),
        to_device(inputs.draft_probs),
        to_device(inputs.target_probs),
        to_device(inputs.inv_q),
        max_spec_len=_MAX_SPEC_LEN,
    )
    assert torch.equal(recovered.cpu(), recovered_ref)

    # Feed the reference recovered ids so the two kernels are checked independently.
    output = torch.full(
        (batch_size, _MAX_SPEC_LEN + 1), -1, dtype=torch.int32, device=qaic_device
    )
    rejection_sampler._rejection_random_sample(
        output,
        to_device(inputs.cu),
        to_device(inputs.draft_ids),
        to_device(inputs.draft_probs),
        to_device(inputs.target_probs),
        to_device(inputs.bonus),
        to_device(recovered_ref),
        to_device(inputs.uniform),
        torch.zeros(batch_size, dtype=torch.bool, device=qaic_device),
        _MAX_SPEC_LEN,
        None,
    )
    expected = _reference_random(inputs, recovered_ref)
    assert torch.equal(output.cpu(), expected)
    # Guard against a vacuous pass: the accept and bonus paths must be hit.
    # (A single-row batch has too few draws to guarantee either.)
    if batch_size > 1:
        assert (expected[:, :_MAX_SPEC_LEN] == _draft_matrix(inputs)).any()
        assert any(
            n > 0 and expected[i, n] == inputs.bonus[i]
            for i, n in enumerate(inputs.drafts)
        ), "no row reached the bonus-token path"


def test_qaic_rejects_float64_rand(qaic_device):
    """Documents why the shim exists: upstream draws fp64, which QAIC rejects."""
    # Independent of whether install() ran earlier in this process: upstream
    # draws fp64 directly on the device.
    with pytest.raises(RuntimeError, match="Unsupported datatype for QAIC"):
        torch.rand(4, dtype=torch.float64, device=qaic_device)


# ---------------------------------------------------------------------------
# End-to-end: upstream ``rejection_sample`` on QAIC vs a PyTorch CPU reference.
#
# Upstream has no CPU implementation: its Triton kernels launched on CPU tensors
# do not compute the right result (the output stays all placeholder ids), so the
# CPU side is a pure-PyTorch transcription of upstream's semantics. The QAIC
# side is the real upstream function with the shim installed; only its two
# noise sources are pinned: ``generate_uniform_probs`` (fixed fp32 uniforms)
# and the ``q.exponential_()`` draw in ``sample_recovered_tokens``.
# ---------------------------------------------------------------------------

# Logit offsets that make acceptance decisions insensitive to QAIC-vs-CPU
# softmax rounding: accepted drafts get the dominant target mass, rejected drafts
# get ~none, and the fixed uniform 0.5 sits far from either ratio.
_ACCEPT_LOGIT = 15.0
_REJECT_LOGIT = -20.0
_FIXED_UNIFORM = 0.5


def _row(drafts, greedy, accepted):
    assert 0 <= accepted <= drafts
    return SimpleNamespace(drafts=drafts, greedy=greedy, accepted=accepted)


G, R = True, False
_E2E_CASES = {
    # name: (rows as (num_drafts, greedy, num_accepted), vocab, use_draft_probs)
    "all_greedy": (
        [(3, G, 3), (3, G, 1), (2, G, 0), (1, G, 1)],
        32000,
        False,
    ),
    "all_random": ([(4, R, 2), (3, R, 0), (2, R, 2), (4, R, 4)], 32000, True),
    "zero_draft_rows": (
        [(0, R, 0), (3, R, 1), (0, G, 0), (2, G, 2), (0, R, 0)],
        32000,
        True,
    ),
    "mixed_greedy_random": (
        [(3, G, 1), (3, R, 3), (2, G, 2), (4, R, 0)],
        32000,
        True,
    ),
    "all_accepted_bonus": ([(3, R, 3), (2, G, 2), (4, R, 4)], 32000, True),
    "partial_rejection_recovered": (
        [(4, R, 1), (4, R, 2), (3, R, 0)],
        128256,
        True,
    ),
    "ngram_no_draft_probs": (
        [(3, R, 1), (3, R, 3), (2, G, 1), (0, R, 0), (4, R, 0)],
        128256,
        False,
    ),
}


def _build_e2e_case(rows, vocab_size, use_draft_probs, seed):
    rows = [_row(*row) for row in rows]
    generator = torch.Generator().manual_seed(seed)
    num_draft_tokens = [row.drafts for row in rows]
    num_tokens = sum(num_draft_tokens)
    draft_token_ids = torch.randint(
        0, vocab_size, (num_tokens,), dtype=torch.int32, generator=generator
    )
    target_logits = torch.randn(num_tokens, vocab_size, generator=generator) * 2
    draft_probs = (
        torch.softmax(torch.randn(num_tokens, vocab_size, generator=generator), -1)
        if use_draft_probs
        else None
    )
    # Uniforms for positions past the first rejection are drawn but unused.
    uniforms = torch.rand(num_tokens, generator=generator).clamp_min(1e-3)
    greedy_reject_tokens = {}
    position = 0
    for row in rows:
        for offset in range(min(row.accepted + 1, row.drafts)):
            token = position + offset
            draft = int(draft_token_ids[token])
            uniforms[token] = _FIXED_UNIFORM
            if offset < row.accepted:
                target_logits[token, draft] = _ACCEPT_LOGIT
            elif row.greedy:
                replacement = (draft + 1) % vocab_size
                target_logits[token, replacement] = _ACCEPT_LOGIT
                greedy_reject_tokens[token] = replacement
            else:
                target_logits[token, draft] = _REJECT_LOGIT
        position += row.drafts
    greedy = torch.tensor([row.greedy for row in rows])
    return SimpleNamespace(
        rows=rows,
        vocab_size=vocab_size,
        num_draft_tokens=num_draft_tokens,
        num_tokens=num_tokens,
        max_spec_len=max(num_draft_tokens),
        cu_num_draft_tokens=torch.tensor(num_draft_tokens).cumsum(0).to(torch.int32),
        draft_token_ids=draft_token_ids,
        draft_probs=draft_probs,
        target_logits=target_logits.contiguous(),
        bonus_token_ids=torch.randint(
            0, vocab_size, (len(rows), 1), dtype=torch.int64, generator=generator
        ),
        temperature=torch.where(greedy, 0.0, 1.0).to(torch.float32),
        all_greedy=bool(greedy.all()),
        all_random=not bool(greedy.any()),
        uniforms=uniforms,
        q=torch.empty(len(rows), vocab_size).exponential_(generator=generator),
        greedy_reject_tokens=greedy_reject_tokens,
    )


def _e2e_sampling_metadata(case, device):
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata

    batch_size = len(case.rows)
    zeros = torch.zeros(batch_size, device=device)
    return SamplingMetadata(
        temperature=None if case.all_greedy else case.temperature.to(device),
        all_greedy=case.all_greedy,
        all_random=case.all_random,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=zeros,
        presence_penalties=zeros,
        repetition_penalties=zeros,
        output_token_ids=[[] for _ in range(batch_size)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def _reference_rejection_sample(case):
    """PyTorch CPU transcription of upstream ``rejection_sample``."""
    target_probs = case.target_logits.softmax(dim=-1, dtype=torch.float32)
    inv_q = case.q.reciprocal()
    output = torch.full((len(case.rows), case.max_spec_len + 1), -1, dtype=torch.int32)
    position = 0
    for request, row in enumerate(case.rows):
        rejected = False
        for offset in range(row.drafts):
            token = position + offset
            draft = int(case.draft_token_ids[token])
            if row.greedy:
                chosen = int(case.target_logits[token].argmax())
                accepted = chosen == draft
            else:
                draft_prob = (
                    float(case.draft_probs[token, draft])
                    if case.draft_probs is not None
                    else 1.0
                )
                accepted = draft_prob > 0 and float(
                    target_probs[token, draft]
                ) / draft_prob >= float(case.uniforms[token])
                if accepted:
                    chosen = draft
                else:
                    if case.draft_probs is not None:
                        probs = (target_probs[token] - case.draft_probs[token]).clamp(
                            min=0
                        )
                    else:
                        probs = target_probs[token].clone()
                        probs[draft] = 0
                    scores = probs * inv_q[request]
                    top2 = scores.topk(2).values
                    # Guard against a near-tie the device could break differently.
                    assert float(top2[0] - top2[1]) > 1e-4 * float(top2[0])
                    chosen = int(scores.argmax())
            output[request, offset] = chosen
            if not accepted:
                rejected = True
                break
        if not rejected:
            output[request, row.drafts] = int(case.bonus_token_ids[request, 0])
        position += row.drafts
    return output


def _expected_from_spec(case, recovered_from=None):
    """Hand-derived output; rejected random positions read ``recovered_from``."""
    output = torch.full((len(case.rows), case.max_spec_len + 1), -1, dtype=torch.int32)
    position = 0
    for request, row in enumerate(case.rows):
        output[request, : row.accepted] = case.draft_token_ids[
            position : position + row.accepted
        ]
        if row.accepted == row.drafts:
            output[request, row.drafts] = int(case.bonus_token_ids[request, 0])
        elif row.greedy:
            output[request, row.accepted] = case.greedy_reject_tokens[
                position + row.accepted
            ]
        else:
            assert recovered_from is not None, "random rejection needs a reference"
            output[request, row.accepted] = recovered_from[request, row.accepted]
        position += row.drafts
    return output


def _install_shim_for_test(monkeypatch):
    """install(), with the upstream binding and install flag restored on exit."""
    import vllm.v1.sample.rejection_sampler as rejection_sampler
    import vllm_qaic.v1.sample.rejection_sampler_shim as shim

    monkeypatch.setattr(shim, "_shim_installed", False)
    monkeypatch.setattr(
        rejection_sampler,
        "generate_uniform_probs",
        rejection_sampler.generate_uniform_probs,
    )
    install()


def _run_upstream_rejection_sample_on_qaic(case, qaic_device, monkeypatch):
    import vllm.v1.sample.rejection_sampler as rejection_sampler
    import vllm_qaic.v1.sample.rejection_sampler_shim as shim

    _install_shim_for_test(monkeypatch)
    assert rejection_sampler.generate_uniform_probs is shim.generate_uniform_probs

    calls = {"uniform": 0, "exponential": 0}

    def fixed_uniform_probs(num_tokens, num_draft_tokens, generators, device):
        calls["uniform"] += 1
        assert num_tokens == case.num_tokens
        assert list(num_draft_tokens) == case.num_draft_tokens
        # Same dtype the shim produces on QAIC.
        return case.uniforms.to(device=device, dtype=torch.float32)

    original_exponential = torch.Tensor.exponential_

    def fixed_exponential(self, *args, **kwargs):
        if tuple(self.shape) != tuple(case.q.shape):
            return original_exponential(self, *args, **kwargs)
        calls["exponential"] += 1
        assert self.device.type == "qaic"
        return self.copy_(case.q.to(device=self.device, dtype=self.dtype))

    monkeypatch.setattr(
        rejection_sampler, "generate_uniform_probs", fixed_uniform_probs
    )
    monkeypatch.setattr(torch.Tensor, "exponential_", fixed_exponential)

    to_device = lambda t: None if t is None else t.to(qaic_device)  # noqa: E731
    output = rejection_sampler.rejection_sample(
        to_device(case.draft_token_ids),
        case.num_draft_tokens,
        case.max_spec_len,
        to_device(case.cu_num_draft_tokens),
        to_device(case.draft_probs),
        to_device(case.target_logits),
        to_device(case.bonus_token_ids),
        _e2e_sampling_metadata(case, qaic_device),
    )
    assert output.device.type == "qaic"
    assert output.dtype == torch.int32
    expected_calls = 0 if case.all_greedy else 1
    assert calls == {"uniform": expected_calls, "exponential": expected_calls}
    return output.cpu()


@requires_qualcomm_triton
@pytest.mark.parametrize("case_name", list(_E2E_CASES))
def test_upstream_rejection_sample_e2e_matches_cpu_reference_on_qaic(
    qaic_device, monkeypatch, case_name
):
    rows, vocab_size, use_draft_probs = _E2E_CASES[case_name]
    case = _build_e2e_case(rows, vocab_size, use_draft_probs, seed=len(case_name))

    reference = _reference_rejection_sample(case)
    actual = _run_upstream_rejection_sample_on_qaic(case, qaic_device, monkeypatch)

    assert torch.equal(actual, reference), f"qaic={actual}\ncpu={reference}"
    # The constructed acceptance pattern was actually realised.
    assert torch.equal(reference, _expected_from_spec(case, reference))
    if case_name in ("all_greedy", "all_accepted_bonus"):
        # Fully hand-computable: no recovered-token sampling involved.
        assert torch.equal(actual, _expected_from_spec(case))
    # A recovered token can never equal the rejected draft.
    position = 0
    for request, row in enumerate(case.rows):
        if row.accepted < row.drafts:
            rejected_draft = int(case.draft_token_ids[position + row.accepted])
            assert int(actual[request, row.accepted]) != rejected_draft
        position += row.drafts


@requires_qualcomm_triton
def test_upstream_rejection_sample_unpinned_noise_on_qaic(qaic_device, monkeypatch):
    """Real fp32 uniforms and exponential noise on QAIC: structural validity."""
    import vllm.v1.sample.rejection_sampler as rejection_sampler

    _install_shim_for_test(monkeypatch)
    rows, vocab_size, use_draft_probs = _E2E_CASES["mixed_greedy_random"]
    case = _build_e2e_case(rows, vocab_size, use_draft_probs, seed=7)
    to_device = lambda t: None if t is None else t.to(qaic_device)  # noqa: E731
    output = rejection_sampler.rejection_sample(
        to_device(case.draft_token_ids),
        case.num_draft_tokens,
        case.max_spec_len,
        to_device(case.cu_num_draft_tokens),
        to_device(case.draft_probs),
        to_device(case.target_logits),
        to_device(case.bonus_token_ids),
        _e2e_sampling_metadata(case, qaic_device),
    ).cpu()

    # The logits are built so every decision is insensitive to the uniform
    # draw (u in (0, 1) vs ratio ~1 or ~0), so the accept pattern is fixed.
    reference = _reference_rejection_sample(case)
    for request, row in enumerate(case.rows):
        assert torch.equal(
            output[request, : row.accepted], reference[request, : row.accepted]
        )
        emitted = output[request, row.accepted]
        assert 0 <= int(emitted) < vocab_size
        if row.greedy or row.accepted == row.drafts:
            assert int(emitted) == int(reference[request, row.accepted])
        assert bool((output[request, row.accepted + 1 :] == -1).all())
