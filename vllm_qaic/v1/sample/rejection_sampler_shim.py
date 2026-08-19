# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/v1/sample/rejection_sampler.py

"""Pure-PyTorch replacements for the four Triton kernels in
vllm.v1.sample.rejection_sampler, enabling speculative decoding in PYT
(eager) mode on QAIC where no active Triton driver is available.

In PYT mode, standard Triton 3.3.0 is installed but ``HAS_TRITON=False``
(vLLM checks for exactly 1 active GPU driver; QAIC does not register a
Triton backend → 0 active drivers → ``TritonPlaceholder`` is used).
``TritonPlaceholder`` makes ``@triton.jit`` a no-op decorator, so the
decorated functions remain plain Python callables.  vLLM then calls them
as ``kernel[(grid,)](*args)`` which tries to subscript a plain function →
``TypeError: 'function' object is not subscriptable``.

This shim wraps each PyTorch replacement in ``_GridLaunchable`` so that
the ``kernel[(grid,)](*args, CONSTEXPR=value)`` call syntax works
identically to the Triton call sites — no changes needed at the call sites.

The PyTorch replacements operate on tensors that are on the QAIC device
(they come from the model forward pass), so these ops dispatch through
torch_qaic and run on QAIC hardware.

Key behavioural notes:
- ``generate_uniform_probs``: uses ``torch.float32`` instead of
  ``torch.float64`` because QAIC does not support float64.
- ``sample_recovered_tokens_kernel``: ``inv_q`` is pre-computed by the
  ``sample_recovered_tokens`` wrapper before calling the kernel.
- ``USE_FP64_GUMBEL`` is always ``False`` on QAIC; the parameter is
  accepted but ignored.
- For ngram/suffix SpD: ``SYNTHETIC_MODE=False``, ``NO_DRAFT_PROBS=True``.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_qaic.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# _GridLaunchable: adapter for Triton grid-launch call syntax
# ---------------------------------------------------------------------------

class _GridLaunchable:
    """Makes a plain callable usable with Triton grid-launch syntax.

    Triton kernels are called as::

        kernel[(batch_size,)](*args, CONSTEXPR=value)

    where ``kernel[(batch_size,)]`` returns a callable that is immediately
    invoked with ``(*args, ...)``.  This adapter intercepts ``__getitem__``
    and returns a closure over the wrapped function, discarding the grid.
    The grid is unused because the PyTorch implementations process all
    requests in a single tensor expression (or a small Python loop over
    batch_size).
    """

    def __init__(self, fn):
        self._fn = fn
        self.__name__ = getattr(fn, "__name__", repr(fn))

    def __getitem__(self, _grid: Any):
        fn = self._fn

        def _launch(*args, **kwargs):
            return fn(*args, **kwargs)

        return _launch

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# 1. expand_kernel  (expand_batch_to_tokens helper)
#
# Upstream call:
#   expand_kernel[(batch_size,)](
#       expanded_x, x, cu_num_tokens,
#       replace_from, replace_to,
#       MAX_NUM_TOKENS=MAX_SPEC_LEN,
#   )
#
# Semantics: for each request i, broadcast x[i] (with replace_from→replace_to
# substitution) into output[cu[i-1]:cu[i]].
# ---------------------------------------------------------------------------

def _expand_kernel_pyt(
    output_ptr: torch.Tensor,        # [num_tokens]
    input_ptr: torch.Tensor,         # [batch_size]
    cu_num_tokens_ptr: torch.Tensor, # [batch_size]
    replace_from: int,
    replace_to: int,
    MAX_NUM_TOKENS: int = 128,       # constexpr, unused in PyTorch impl
) -> None:
    batch_size = input_ptr.shape[0]
    if batch_size == 0:
        return
    num_tokens = int(cu_num_tokens_ptr[-1].item())
    if num_tokens == 0:
        return

    # Apply replace_from → replace_to substitution.
    src = input_ptr.clone()
    src[src == replace_from] = replace_to

    # Build per-request token counts: counts[i] = cu[i] - cu[i-1].
    prev = torch.zeros(batch_size, dtype=cu_num_tokens_ptr.dtype,
                       device=cu_num_tokens_ptr.device)
    prev[1:] = cu_num_tokens_ptr[:-1]
    counts = (cu_num_tokens_ptr - prev).to(torch.int64)

    # Expand each scalar to its token segment via repeat_interleave.
    req_indices = torch.repeat_interleave(
        torch.arange(batch_size, device=input_ptr.device),
        counts,
    )  # [num_tokens]

    output_ptr.copy_(src[req_indices])


expand_kernel = _GridLaunchable(_expand_kernel_pyt)


# ---------------------------------------------------------------------------
# 2. rejection_greedy_sample_kernel
#
# Upstream call:
#   rejection_greedy_sample_kernel[(batch_size,)](
#       output_token_ids, cu_num_draft_tokens,
#       draft_token_ids, target_argmax, bonus_token_ids,
#       is_greedy, max_spec_len,
#       uniform_probs, synthetic_conditional_rates,
#       SYNTHETIC_MODE=synthetic_mode,
#   )
#
# For ngram/suffix: SYNTHETIC_MODE=False, is_greedy=None (all_greedy=True).
# The standard path stores target_argmax per position until first mismatch
# with draft token, then stops; appends bonus token if all accepted.
# ---------------------------------------------------------------------------

def _rejection_greedy_sample_kernel_pyt(
    output_token_ids_ptr: torch.Tensor,      # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr: torch.Tensor,   # [batch_size]
    draft_token_ids_ptr: torch.Tensor,       # [num_tokens]
    target_argmax_ptr: torch.Tensor,         # [num_tokens]
    bonus_token_ids_ptr: torch.Tensor,       # [batch_size]
    is_greedy_ptr,                           # [batch_size] tensor or None
    max_spec_len: int,
    uniform_probs_ptr,                       # [num_tokens] tensor or None
    synthetic_conditional_rates_ptr,         # [num_spec_tokens] tensor or None
    SYNTHETIC_MODE: bool = False,
) -> None:
    batch_size = int(cu_num_draft_tokens_ptr.shape[0])
    device = cu_num_draft_tokens_ptr.device

    prev = torch.zeros(batch_size, dtype=cu_num_draft_tokens_ptr.dtype,
                       device=device)
    prev[1:] = cu_num_draft_tokens_ptr[:-1]
    start_idxs = prev.to(torch.int64)
    end_idxs = cu_num_draft_tokens_ptr.to(torch.int64)

    for req_idx in range(batch_size):
        # Skip non-greedy requests — handled by the random-sample kernel.
        if is_greedy_ptr is not None and not bool(is_greedy_ptr[req_idx].item()):
            continue

        s = int(start_idxs[req_idx].item())
        e = int(end_idxs[req_idx].item())
        num_draft = e - s

        rejected = False
        for pos in range(num_draft):
            if rejected:
                break
            tok_idx = s + pos
            draft_tok = int(draft_token_ids_ptr[tok_idx].item())
            target_tok = int(target_argmax_ptr[tok_idx].item())

            if SYNTHETIC_MODE:
                u = float(uniform_probs_ptr[tok_idx].item())
                rate = float(synthetic_conditional_rates_ptr[pos].item())
                accepted = u < rate
                token_id = draft_tok if accepted else target_tok
                rejected = not accepted
            else:
                token_id = target_tok
                rejected = (draft_tok != target_tok)

            output_token_ids_ptr[req_idx, pos] = token_id

        if not rejected:
            output_token_ids_ptr[req_idx, num_draft] = int(
                bonus_token_ids_ptr[req_idx].item()
            )


rejection_greedy_sample_kernel = _GridLaunchable(
    _rejection_greedy_sample_kernel_pyt
)


# ---------------------------------------------------------------------------
# 3. rejection_random_sample_kernel
#
# Upstream call:
#   rejection_random_sample_kernel[(batch_size,)](
#       output_token_ids, cu_num_draft_tokens,
#       draft_token_ids, draft_probs, target_probs,
#       bonus_token_ids, recovered_token_ids, uniform_probs,
#       is_greedy, max_spec_len, vocab_size,
#       synthetic_conditional_rates,
#       NO_DRAFT_PROBS=draft_probs is None,
#       SYNTHETIC_MODE=synthetic_mode,
#   )
#
# For ngram/suffix: NO_DRAFT_PROBS=True (draft_prob treated as 1.0).
# Greedy requests are skipped (handled by the greedy kernel above).
# ---------------------------------------------------------------------------

def _rejection_random_sample_kernel_pyt(
    output_token_ids_ptr: torch.Tensor,      # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr: torch.Tensor,   # [batch_size]
    draft_token_ids_ptr: torch.Tensor,       # [num_tokens]
    draft_probs_ptr,                         # [num_tokens, vocab_size] or None
    target_probs_ptr: torch.Tensor,          # [num_tokens, vocab_size]
    bonus_token_ids_ptr: torch.Tensor,       # [batch_size]
    recovered_token_ids_ptr: torch.Tensor,   # [num_tokens]
    uniform_probs_ptr: torch.Tensor,         # [num_tokens]
    is_greedy_ptr: torch.Tensor,             # [batch_size]
    max_spec_len: int,
    vocab_size: int,
    synthetic_conditional_rates_ptr,         # [num_spec_tokens] tensor or None
    NO_DRAFT_PROBS: bool = True,
    SYNTHETIC_MODE: bool = False,
) -> None:
    batch_size = int(cu_num_draft_tokens_ptr.shape[0])
    device = cu_num_draft_tokens_ptr.device

    prev = torch.zeros(batch_size, dtype=cu_num_draft_tokens_ptr.dtype,
                       device=device)
    prev[1:] = cu_num_draft_tokens_ptr[:-1]
    start_idxs = prev.to(torch.int64)
    end_idxs = cu_num_draft_tokens_ptr.to(torch.int64)

    for req_idx in range(batch_size):
        # Greedy requests are handled by rejection_greedy_sample_kernel.
        if bool(is_greedy_ptr[req_idx].item()):
            continue

        s = int(start_idxs[req_idx].item())
        e = int(end_idxs[req_idx].item())
        num_draft = e - s

        rejected = False
        for pos in range(num_draft):
            if rejected:
                break
            tok_idx = s + pos
            draft_tok = int(draft_token_ids_ptr[tok_idx].item())
            u = float(uniform_probs_ptr[tok_idx].item())

            if SYNTHETIC_MODE:
                rate = float(synthetic_conditional_rates_ptr[pos].item())
                accepted = u < rate
            else:
                if NO_DRAFT_PROBS:
                    draft_prob = 1.0
                else:
                    draft_prob = float(
                        draft_probs_ptr[tok_idx, draft_tok].item()
                    )
                target_prob = float(
                    target_probs_ptr[tok_idx, draft_tok].item()
                )
                # Mirror Triton logic exactly: draft_prob==0 means reject.
                accepted = (
                    draft_prob > 0
                    and target_prob / draft_prob >= u
                )

            if accepted:
                token_id = draft_tok
            else:
                rejected = True
                token_id = int(recovered_token_ids_ptr[tok_idx].item())

            output_token_ids_ptr[req_idx, pos] = token_id

        if not rejected:
            output_token_ids_ptr[req_idx, num_draft] = int(
                bonus_token_ids_ptr[req_idx].item()
            )


rejection_random_sample_kernel = _GridLaunchable(
    _rejection_random_sample_kernel_pyt
)


# ---------------------------------------------------------------------------
# 4. sample_recovered_tokens_kernel
#
# Upstream call:
#   sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
#       recovered_token_ids, cu_num_draft_tokens,
#       draft_token_ids, draft_probs, target_probs,
#       inv_q, vocab_size, BLOCK_SIZE,
#       NO_DRAFT_PROBS=draft_probs is None,
#       USE_FP64_GUMBEL=use_fp64_gumbel,
#   )
#
# ``inv_q`` is pre-computed by the ``sample_recovered_tokens`` wrapper:
#   q.exponential_(); inv_q = q.reciprocal()
#
# For NO_DRAFT_PROBS=True (ngram/suffix):
#   prob[pos, v] = target_probs[tok_idx, v], masked to 0 at v==draft_tok.
#   score[pos, v] = prob[pos, v] * inv_q[req_idx, v]
#   recovered[pos] = argmax_v(score[pos, :])
#
# For NO_DRAFT_PROBS=False:
#   prob[pos, v] = max(target_probs[tok_idx, v] - draft_probs[tok_idx, v], 0)
#
# USE_FP64_GUMBEL is always False on QAIC (no float64 support); accepted
# but ignored.
# ---------------------------------------------------------------------------

def _sample_recovered_tokens_kernel_pyt(
    output_token_ids_ptr: torch.Tensor,      # [num_tokens]
    cu_num_draft_tokens_ptr: torch.Tensor,   # [batch_size]
    draft_token_ids_ptr: torch.Tensor,       # [num_tokens]
    draft_probs_ptr,                         # [num_tokens, vocab_size] or None
    target_probs_ptr: torch.Tensor,          # [num_tokens, vocab_size]
    inv_q_ptr: torch.Tensor,                 # [batch_size, vocab_size]
    vocab_size: int,
    BLOCK_SIZE: int = 8192,                  # constexpr, unused in PyTorch
    NO_DRAFT_PROBS: bool = True,
    USE_FP64_GUMBEL: bool = False,           # always False on QAIC
) -> None:
    batch_size = int(cu_num_draft_tokens_ptr.shape[0])
    device = cu_num_draft_tokens_ptr.device

    prev = torch.zeros(batch_size, dtype=cu_num_draft_tokens_ptr.dtype,
                       device=device)
    prev[1:] = cu_num_draft_tokens_ptr[:-1]
    start_idxs = prev.to(torch.int64)
    end_idxs = cu_num_draft_tokens_ptr.to(torch.int64)

    for req_idx in range(batch_size):
        s = int(start_idxs[req_idx].item())
        e = int(end_idxs[req_idx].item())
        num_draft = e - s
        if num_draft == 0:
            continue

        # Flat token indices for this request.
        tok_slice = slice(s, e)

        if NO_DRAFT_PROBS:
            # prob[pos, v] = target_probs[s+pos, v], 0 at v == draft_tok.
            prob = target_probs_ptr[tok_slice].clone()  # [num_draft, vocab]
            draft_toks = draft_token_ids_ptr[tok_slice].to(torch.int64)
            # Use scatter_ to zero out the draft-token column for each pos.
            # (Avoids 2-D advanced indexing which may not be supported on QAIC.)
            pos_idx = torch.arange(num_draft, device=device)
            prob.scatter_(1, draft_toks.view(-1, 1), 0.0)
        else:
            prob = torch.clamp(
                target_probs_ptr[tok_slice] - draft_probs_ptr[tok_slice],
                min=0.0,
            )  # [num_draft, vocab]

        # Gumbel-max trick: score[pos, v] = prob[pos, v] * inv_q[req, v].
        # inv_q[req_idx] is [vocab_size]; broadcast over num_draft positions.
        score = prob * inv_q_ptr[req_idx].unsqueeze(0)  # [num_draft, vocab]
        recovered = score.argmax(dim=-1).to(torch.int32)  # [num_draft]
        output_token_ids_ptr[s:e].copy_(recovered)


sample_recovered_tokens_kernel = _GridLaunchable(
    _sample_recovered_tokens_kernel_pyt
)


# ---------------------------------------------------------------------------
# 5. generate_uniform_probs  (float32 replacement for float64 upstream)
#
# Upstream uses torch.float64 to avoid sampling exact 0.0 (see PyTorch
# issue #16706).  QAIC does not support float64, so we use float32.
# The probability of an exact 0.0 in float32 is ~2^-24 ≈ 6e-8, acceptable.
# ---------------------------------------------------------------------------

def generate_uniform_probs(
    num_tokens: int,
    num_draft_tokens: list,
    generators: dict,
    device: torch.device,
) -> torch.Tensor:
    """float32 uniform samples; per-request generators are honoured."""
    uniform_probs = torch.rand(
        (num_tokens,),
        dtype=torch.float32,
        device=device,
    )
    start_idx = 0
    for req_idx, n in enumerate(num_draft_tokens):
        if n == 0:
            continue
        end_idx = start_idx + n
        gen = generators.get(req_idx)
        if gen is not None:
            uniform_probs[start_idx:end_idx].uniform_(generator=gen)
        start_idx = end_idx
    return uniform_probs


# ---------------------------------------------------------------------------
# install() — idempotent module-level monkey-patch
# ---------------------------------------------------------------------------

_shim_installed: bool = False


def _patch_sampler_temperature() -> None:
    """Avoid QAIC's unsupported tensor/scalar comparison in vLLM Sampler."""
    import vllm.v1.sample.sampler as sampler_module

    if getattr(sampler_module.Sampler, "_qaic_temperature_patched", False):
        return

    sampling_eps = sampler_module._SAMPLING_EPS

    def apply_temperature(
        logits: torch.Tensor,
        temp: torch.Tensor,
        all_random: bool,
    ) -> torch.Tensor:
        if not all_random:
            host_temp = temp.cpu()
            host_temp = torch.where(
                host_temp < sampling_eps,
                torch.ones_like(host_temp),
                host_temp,
            )
            temp = host_temp.to(device=temp.device)
        return logits.div_(temp.unsqueeze(dim=1))

    sampler_module.Sampler.apply_temperature = staticmethod(apply_temperature)
    sampler_module.Sampler._qaic_temperature_patched = True


def _patch_sampler_comparisons() -> None:
    """Move sampler comparison predicates to CPU for the QAIC backend."""
    import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_module
    import vllm.v1.sample.sampler as sampler_module

    if not getattr(topk_topp_module, "_qaic_comparisons_patched", False):
        original_apply_top_k_top_p_pytorch = (
            topk_topp_module.apply_top_k_top_p_pytorch
        )

        def apply_top_k_top_p_pytorch(
            logits: torch.Tensor,
            k: torch.Tensor | None,
            p: torch.Tensor | None,
            allow_cpu_sync: bool = False,
        ) -> torch.Tensor:
            if logits.device.type != "qaic":
                return original_apply_top_k_top_p_pytorch(
                    logits, k, p, allow_cpu_sync
                )
            if p is None and k is None:
                return logits

            logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
            if k is not None:
                top_k_mask = logits_sort.size(1) - k.to(torch.long)
                top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
                top_k_mask = torch.lt(
                    logits_sort.cpu(), top_k_mask.cpu()
                ).to(device=logits.device)
                logits_sort.masked_fill_(top_k_mask, -float("inf"))

            if p is not None:
                probs_sort = logits_sort.softmax(dim=-1)
                probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
                top_p_mask = torch.le(
                    probs_sum.cpu(),
                    (1 - p.unsqueeze(dim=1)).cpu(),
                ).to(device=logits.device)
                top_p_mask[:, -1] = False
                logits_sort.masked_fill_(top_p_mask, -float("inf"))

            return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)

        topk_topp_module.apply_top_k_top_p_pytorch = apply_top_k_top_p_pytorch
        topk_topp_module._qaic_comparisons_patched = True

    if getattr(sampler_module.Sampler, "_qaic_sample_patched", False):
        return

    def sample(self, logits, sampling_metadata, logprobs_mode_override=None):
        if logits.device.type != "qaic":
            return sampler_module.Sampler._qaic_original_sample(
                self, logits, sampling_metadata, logprobs_mode_override
            )

        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                processed_logprobs = None
                if (
                    sampling_metadata.max_num_logprobs is not None
                    or sampling_metadata.logprob_token_ids
                ):
                    if logprobs_mode == "processed_logits":
                        processed_logprobs = logits
                    elif logprobs_mode == "processed_logprobs":
                        processed_logprobs = self.compute_logprobs(logits)
                return greedy_sampled, processed_logprobs

        assert sampling_metadata.temperature is not None
        logits = self.apply_temperature(
            logits, sampling_metadata.temperature, sampling_metadata.all_random
        )
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )
        if greedy_sampled is None:
            return random_sampled, processed_logprobs

        greedy_mask = (
            sampling_metadata.temperature.cpu()
            < sampler_module._SAMPLING_EPS
        ).to(device=logits.device)
        sampled = torch.where(
            greedy_mask,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,
        )
        return sampled, processed_logprobs

    sampler_module.Sampler._qaic_original_sample = sampler_module.Sampler.sample
    sampler_module.Sampler.sample = sample
    sampler_module.Sampler._qaic_sample_patched = True


def install() -> None:
    """Replace Triton kernel objects in ``vllm.v1.sample.rejection_sampler``
    with PyTorch equivalents wrapped in ``_GridLaunchable``.

    Must be called before LLM is instantiated, i.e. from
    ``QaicPlatform.pre_register_and_update()``.

    The function resolves kernel names via the module's ``__dict__``
    (which is the same object as each function's ``__globals__``), so
    patching here automatically takes effect inside ``rejection_sample()``
    and ``sample_recovered_tokens()`` without touching their call sites.
    """
    global _shim_installed
    if _shim_installed:
        return

    import vllm.v1.sample.rejection_sampler as _rs

    _rs.expand_kernel = expand_kernel
    _rs.rejection_greedy_sample_kernel = rejection_greedy_sample_kernel
    _rs.rejection_random_sample_kernel = rejection_random_sample_kernel
    _rs.sample_recovered_tokens_kernel = sample_recovered_tokens_kernel
    _rs.generate_uniform_probs = generate_uniform_probs
    _patch_sampler_temperature()
    _patch_sampler_comparisons()

    _shim_installed = True
    logger.info(
        "vllm_qaic: Triton rejection-sampler kernels replaced with "
        "PyTorch equivalents (ngram/suffix SpD enabled in PYT mode)."
    )
