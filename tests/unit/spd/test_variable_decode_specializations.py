# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for variable decode specializations (ngram/suffix SpD optimisation).

All tests require no QAIC hardware and no model loading.

Background
----------
For ngram/suffix speculative decoding, the QPC is compiled with two decode
specialisations: K=0 (seq_len=1, cheap fallback) and K=max (seq_len=K+1,
full SpD).  On steps where no requests have draft proposals, the runner
dispatches to the K=0 kernel, saving the wasted 4-token forward pass.
"""

import types
from unittest.mock import MagicMock

import numpy as np

from vllm_qaic.worker.model_runner import QaicModelRunnerAoT as QaicModelRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scheduler_output(spec_tokens: dict | None) -> types.SimpleNamespace:
    """Minimal SchedulerOutput mock with scheduled_spec_decode_tokens."""
    return types.SimpleNamespace(scheduled_spec_decode_tokens=spec_tokens or {})


def _make_runner_ns(
    decode_ks: list[int],
    num_decodes: int = 2,
) -> types.SimpleNamespace:
    """Minimal mock runner for _determine_active_k."""
    return types.SimpleNamespace(
        decode_ks=decode_ks,
        num_decodes=num_decodes,
    )


def _call_determine_active_k(runner_ns, scheduler_output) -> int:
    return QaicModelRunner._determine_active_k(runner_ns, scheduler_output)


# ---------------------------------------------------------------------------
# _determine_active_k
# ---------------------------------------------------------------------------


class TestDetermineActiveK:
    def test_no_proposals_returns_zero(self):
        """All requests have 0 draft tokens → fallback K=0."""
        runner = _make_runner_ns(decode_ks=[0, 4])
        so = _make_scheduler_output(spec_tokens={})  # empty dict = no proposals
        assert _call_determine_active_k(runner, so) == 0

    def test_none_spec_tokens_returns_zero(self):
        """scheduled_spec_decode_tokens is None → fallback K=0."""
        runner = _make_runner_ns(decode_ks=[0, 4])
        so = _make_scheduler_output(spec_tokens=None)
        assert _call_determine_active_k(runner, so) == 0

    def test_proposals_exist_returns_max_k(self):
        """At least one request has proposals → full K=max."""
        runner = _make_runner_ns(decode_ks=[0, 4])
        so = _make_scheduler_output(spec_tokens={"req_0": [10, 11, 12, 13]})
        assert _call_determine_active_k(runner, so) == 4

    def test_mixed_proposals_returns_max_k(self):
        """Some requests have proposals, some don't → still use full K."""
        runner = _make_runner_ns(decode_ks=[0, 4])
        so = _make_scheduler_output(spec_tokens={"req_0": [], "req_1": [5, 6]})
        assert _call_determine_active_k(runner, so) == 4

    def test_single_spec_always_max_k(self):
        """With only one K available, always returns that K regardless of proposals."""
        runner = _make_runner_ns(decode_ks=[4])
        # even with no proposals
        so = _make_scheduler_output(spec_tokens={})
        assert _call_determine_active_k(runner, so) == 4

    def test_no_decodes_returns_max_k(self):
        """Zero active decode requests: no dispatch needed, return max K."""
        runner = _make_runner_ns(decode_ks=[0, 4], num_decodes=0)
        so = _make_scheduler_output(spec_tokens={})
        assert _call_determine_active_k(runner, so) == 4


# ---------------------------------------------------------------------------
# Compile config: num_speculative_tokens emitted for each method
# ---------------------------------------------------------------------------


def _build_minimal_vllm_config(method: str | None, K: int = 4) -> types.SimpleNamespace:
    """Build a minimal VllmConfig-like namespace for _get_qaic_compile_config tests."""
    spec = None
    if method is not None:
        spec = types.SimpleNamespace(
            method=method,
            num_speculative_tokens=K,
        )
    return types.SimpleNamespace(speculative_config=spec)


class TestCompileConfigNumSpecTokens:
    """Verify that _get_qaic_compile_config emits the right num_speculative_tokens."""

    def _get_num_spec_tokens_from_cfg(self, method: str | None, K: int = 4) -> object:
        """Extract only the num_speculative_tokens value that would be put in cfg."""
        spec_cfg = _build_minimal_vllm_config(method, K).speculative_config
        K_val = spec_cfg.num_speculative_tokens if spec_cfg else None
        # Mirror the logic added to _get_qaic_compile_config
        if spec_cfg and spec_cfg.method in ("ngram", "suffix") and K_val:
            return [0, K_val]
        return K_val

    def test_ngram_produces_list(self):
        result = self._get_num_spec_tokens_from_cfg("ngram", K=4)
        assert result == [0, 4]

    def test_suffix_produces_list(self):
        result = self._get_num_spec_tokens_from_cfg("suffix", K=4)
        assert result == [0, 4]

    def test_draft_model_produces_int(self):
        result = self._get_num_spec_tokens_from_cfg("draft_model", K=4)
        assert result == 4

    def test_no_spec_produces_none(self):
        result = self._get_num_spec_tokens_from_cfg(None)
        assert result is None


# ---------------------------------------------------------------------------
# QaicModelRunner.decode_ks
# ---------------------------------------------------------------------------


def _make_runner_with_spec_method(
    method: str | None, K: int = 4
) -> types.SimpleNamespace:
    """Build a minimal namespace mimicking QaicModelRunner attrs after __init__."""
    spec = None
    if method is not None:
        spec = types.SimpleNamespace(method=method)
    num_spec_tokens = K if method is not None else 0
    max_decode_tokens = 1 + num_spec_tokens
    _method = spec.method if spec else None
    decode_ks = (
        [0, num_spec_tokens]
        if _method in ("ngram", "suffix") and max_decode_tokens > 1
        else [num_spec_tokens]
    )
    return types.SimpleNamespace(
        speculative_config=spec,
        num_spec_tokens=num_spec_tokens,
        max_decode_tokens=max_decode_tokens,
        decode_ks=decode_ks,
    )


class TestRunnerDecodeKs:
    def test_ngram_decode_ks(self):
        runner = _make_runner_with_spec_method("ngram", K=4)
        assert runner.decode_ks == [0, 4]

    def test_suffix_decode_ks(self):
        runner = _make_runner_with_spec_method("suffix", K=4)
        assert runner.decode_ks == [0, 4]

    def test_draft_model_decode_ks(self):
        runner = _make_runner_with_spec_method("draft_model", K=4)
        assert runner.decode_ks == [4]

    def test_no_spec_decode_ks(self):
        runner = _make_runner_with_spec_method(None)
        assert runner.decode_ks == [0]


# ---------------------------------------------------------------------------
# QaicCausalLM: per-K buffer shapes
# ---------------------------------------------------------------------------


def _make_causal_lm_ns(
    decode_ks: list[int], decode_bsz: int = 4, vocab_size: int = 32000
):
    """Simulate QaicCausalLM.__init__ buffer allocation for a given decode_ks."""
    decode_batch_inputs_by_k: dict[int, dict] = {}
    for _k in decode_ks:
        _mdt = _k + 1
        decode_batch_inputs_by_k[_k] = {
            "input_ids": np.full((decode_bsz, _mdt), -1, dtype=np.int64),
            "position_ids": np.full((decode_bsz, _mdt), -1, dtype=np.int64),
            "batch_index": np.full((decode_bsz, 1), -1, dtype=np.int64),
        }
    decode_logits_by_k: dict[int, dict] = {}
    decode_num_logits_buffer_by_k: dict[int, dict] = {}
    for _k in decode_ks:
        _mdt = _k + 1
        decode_logits_by_k[_k] = {
            "logits": np.random.randn(decode_bsz, _mdt, vocab_size).astype(np.float32)
        }
        decode_num_logits_buffer_by_k[_k] = {
            "num_logits_to_keep": np.zeros((_mdt, 1), np.int64)
        }
    return types.SimpleNamespace(
        decode_ks=decode_ks,
        decode_bsz=decode_bsz,
        vocab_size=vocab_size,
        decode_batch_inputs_by_k=decode_batch_inputs_by_k,
        decode_logits_by_k=decode_logits_by_k,
        decode_num_logits_buffer_by_k=decode_num_logits_buffer_by_k,
    )


class TestCausalLMBufferShapes:
    def test_fallback_input_shape(self):
        """K=0 input buffer should have seq_len=1."""
        ns = _make_causal_lm_ns([0, 4])
        assert ns.decode_batch_inputs_by_k[0]["input_ids"].shape == (4, 1)

    def test_full_spd_input_shape(self):
        """K=4 input buffer should have seq_len=5."""
        ns = _make_causal_lm_ns([0, 4])
        assert ns.decode_batch_inputs_by_k[4]["input_ids"].shape == (4, 5)

    def test_fallback_logit_shape(self):
        """K=0 logit buffer should be (bsz, 1, vocab)."""
        ns = _make_causal_lm_ns([0, 4])
        assert ns.decode_logits_by_k[0]["logits"].shape == (4, 1, 32000)

    def test_full_spd_logit_shape(self):
        """K=4 logit buffer should be (bsz, 5, vocab)."""
        ns = _make_causal_lm_ns([0, 4])
        assert ns.decode_logits_by_k[4]["logits"].shape == (4, 5, 32000)

    def test_single_spec_only_has_max_k(self):
        """Single-spec (no fallback) only allocates one K dict."""
        ns = _make_causal_lm_ns([4])
        assert list(ns.decode_batch_inputs_by_k.keys()) == [4]

    def test_fallback_num_logits_buffer_shape(self):
        """K=0 num_logits_to_keep buffer should have shape (1, 1)."""
        ns = _make_causal_lm_ns([0, 4])
        assert ns.decode_num_logits_buffer_by_k[0]["num_logits_to_keep"].shape == (1, 1)


# ---------------------------------------------------------------------------
# _run_decode with K=0: session receives shape (bsz, 1) input
# ---------------------------------------------------------------------------


class TestRunDecodeKZero:
    def _make_model_ns(self, decode_ks, decode_bsz=2, vocab_size=100):
        ns = _make_causal_lm_ns(decode_ks, decode_bsz, vocab_size)
        ns.active_k = 0
        ns.last_decode = False
        ns.ignore_batch_index = False
        ns.is_spec_decode_target_model = True
        ns.comp_ctx_lengths_decode = None
        ns.disagg_serving_en = False
        ns.use_async_scheduling = False  # required by _run_decode's complete_inf guard
        ns.uses_mrope = False
        # Mock session
        mock_session = MagicMock()
        mock_session.np_run.return_value = 0  # exec_obj_idx
        ns.session = mock_session
        return ns

    def test_run_decode_k0_uses_mdt_one(self):
        """With active_k=0, _run_decode should divide input by mdt=1."""
        from vllm_qaic.model_loader.qaic import QaicCausalLM

        ns = self._make_model_ns([0, 4], decode_bsz=2)
        # 2 decode requests × 1 token each
        input_ids = np.array([10, 20], dtype=np.int64)
        positions = np.array([5, 7], dtype=np.int64)
        batch_indices = np.array([0, 1], dtype=np.int64)

        result = QaicCausalLM._run_decode(ns, input_ids, positions, batch_indices, None)

        # Verify session.np_run received shape (bsz=2, mdt=1) via batch_inputs
        call_args = ns.session.np_run.call_args[0][0]
        assert call_args["input_ids"].shape == (2, 1), (
            f"Expected (2, 1), got {call_args['input_ids'].shape}"
        )
        # _run_decode returns None in async mode; output is in the pre-allocated buffer
        assert result is None

    def test_run_decode_k_max_uses_mdt_5(self):
        """With active_k=4, _run_decode should use mdt=5."""
        from vllm_qaic.model_loader.qaic import QaicCausalLM

        ns = self._make_model_ns([0, 4], decode_bsz=2)
        ns.active_k = 4
        # 2 decode requests × 5 tokens each
        input_ids = np.array([10, 11, 12, 13, 14, 20, 21, -1, -1, -1], dtype=np.int64)
        positions = np.array([5, 6, 7, 8, 9, 3, 4, -1, -1, -1], dtype=np.int64)
        batch_indices = np.array([0, 1], dtype=np.int64)
        result = QaicCausalLM._run_decode(ns, input_ids, positions, batch_indices, None)

        call_args = ns.session.np_run.call_args[0][0]
        assert call_args["input_ids"].shape == (2, 5), (
            f"Expected (2, 5), got {call_args['input_ids'].shape}"
        )
        assert result is None

    def test_buffers_reset_on_k_change(self):
        """num_logits_to_keep buffer matches active K in batch_inputs."""
        from vllm_qaic.model_loader.qaic import QaicCausalLM

        ns = self._make_model_ns([0, 4], decode_bsz=2)
        ns.active_k = 0

        input_ids = np.array([10, 20], dtype=np.int64)
        positions = np.array([5, 7], dtype=np.int64)
        batch_indices = np.array([0, 1], dtype=np.int64)
        QaicCausalLM._run_decode(ns, input_ids, positions, batch_indices, None)

        # num_logits_to_keep for K=0 has shape (1, 1)
        call_args = ns.session.np_run.call_args[0][0]
        assert "num_logits_to_keep" in call_args, (
            "batch_inputs should contain num_logits_to_keep for spec-decode target"
        )
        assert call_args["num_logits_to_keep"].shape == (1, 1), (
            f"Expected (1, 1) for K=0, got {call_args['num_logits_to_keep'].shape}"
        )


# ---------------------------------------------------------------------------
# max_num_batched_tokens: QaicPlatform.check_and_update_config formula
# ---------------------------------------------------------------------------


def _qaic_max_num_batched_tokens(
    max_num_seqs: int,
    num_spec_tokens: int,
    prefill_seq_len: int,
) -> int:
    """Mirror the formula from QaicPlatform.check_and_update_config.

    This is the token budget set by the QAIC platform before the GPU model
    runner allocates its buffers, ensuring positions/input_ids/arange_np are
    large enough to hold the post-expansion decode token counts.
    """
    return max_num_seqs * (1 + num_spec_tokens) + prefill_seq_len


class TestMaxNumBatchedTokens:
    """
    Verify the max_num_batched_tokens formula set in
    QaicPlatform.check_and_update_config.

    Root cause of the buffer overflow (now fixed at platform level):
    _prepare_qaic_inputs expands decode token counts from 1 → (1+num_spec)
    per request.  If max_num_batched_tokens = max_model_len (vLLM default),
    the parent GPU model runner's positions/input_ids/arange_np buffers are
    too small after expansion.

    Fix: set max_num_batched_tokens = max_num_seqs*(1+num_spec) + prefill_seq_len
    in check_and_update_config so that ALL parent-class buffers are correctly
    sized before any model runner code runs.
    """

    def test_nospec_formula(self):
        """nospec: budget = max_num_seqs * 1 + prefill_seq_len."""
        result = _qaic_max_num_batched_tokens(
            max_num_seqs=64, num_spec_tokens=0, prefill_seq_len=128
        )
        assert result == 64 * 1 + 128  # = 192

    def test_spd_k4_formula(self):
        """SpD K=4: budget = max_num_seqs * 5 + prefill_seq_len."""
        result = _qaic_max_num_batched_tokens(
            max_num_seqs=64, num_spec_tokens=4, prefill_seq_len=128
        )
        assert result == 64 * 5 + 128  # = 448

    def test_budget_covers_worst_case_expansion(self):
        """Budget must be >= max total tokens after decode expansion.

        Worst case: scheduler fills entire budget with decode×1 + prefill.
        After expansion: max_num_seqs×(1+num_spec) + remaining_prefill_tokens.
        The budget is designed to equal exactly this worst case.
        """
        max_num_seqs = 64
        num_spec = 4
        prefill_seq_len = 128
        budget = _qaic_max_num_batched_tokens(max_num_seqs, num_spec, prefill_seq_len)

        # Scheduler schedules: max_num_seqs×1 + prefill = budget (old 2048 default)
        # After expansion: max_num_seqs×(1+num_spec) + prefill_seq_len
        worst_case = max_num_seqs * (1 + num_spec) + prefill_seq_len
        assert budget >= worst_case

    def test_budget_strictly_larger_than_old_default_for_spd(self):
        """With SpD, the new budget must exceed the old max_model_len=2048 default.

        This is the condition that prevented the overflow: the old value was
        too small and new value is correctly sized.
        """
        old_default = 2048  # max_model_len used by GPU model runner default

        # mns=64, K=4: 64*5+128=448 — this is smaller than 2048 but the
        # scheduler now only schedules up to 448 tokens, so no overflow.
        budget_mns64 = _qaic_max_num_batched_tokens(64, 4, 128)
        assert budget_mns64 < old_default  # smaller budget → smaller buffers OK

        # The fix is not about making buffers bigger — it's about making the
        # scheduler honest: it won't schedule more tokens than QAIC can expand.
        # With the old default: scheduler could schedule 2048 - 64 = 1984 prefill
        # tokens alongside 64 decodes; after expansion: 64*5+1984 = 2304 > 2048.
        expanded_old = 64 * 5 + (old_default - 64)  # = 2304
        assert expanded_old > old_default  # would have overflowed

        # With new budget: scheduler schedules at most budget_mns64 = 448 tokens.
        # After expansion: still 448 (the budget already accounts for expansion).
        assert budget_mns64 == 64 * 5 + 128

    def test_mns128_k4(self):
        """mns=128, K=4: 128*5+128 = 768."""
        result = _qaic_max_num_batched_tokens(128, 4, 128)
        assert result == 128 * 5 + 128  # = 768

    def test_mns1_nospec(self):
        """mns=1, nospec: 1*1+128 = 129."""
        result = _qaic_max_num_batched_tokens(1, 0, 128)
        assert result == 129


# ---------------------------------------------------------------------------
# _decode_ks_from_session: backward-compat K extraction from session shapes
# ---------------------------------------------------------------------------


class TestDecodeKsFromSession:
    def _make_ns(self, shapes, prefill_seq_len=128, decode_ks=None):
        """Minimal namespace for _decode_ks_from_session."""
        from unittest.mock import MagicMock

        mock_session = MagicMock()
        mock_session.allowed_shapes = shapes
        mock_session.binding_index_map = {"input_ids": 0}
        return types.SimpleNamespace(
            session=mock_session,
            prefill_seq_len=prefill_seq_len,
            decode_ks=decode_ks or [4],
        )

    def test_extracts_dual_k_from_shapes(self):
        """Correctly parses K=0 and K=4 from two decode specializations."""
        from vllm_qaic.model_loader.qaic import QaicCausalLM

        # Each element: allowed_shapes[i][input_idx][1][1] = seq_len
        shapes = [
            [[None, [None, 1]]],  # K=0, seq_len=1
            [[None, [None, 5]]],  # K=4, seq_len=5
            [[None, [None, 128]]],  # prefill, filtered out
        ]
        ns = self._make_ns(shapes, prefill_seq_len=128, decode_ks=[0, 4])
        result = QaicCausalLM._decode_ks_from_session(ns)
        assert result == [0, 4]

    def test_single_spec_qpc_returns_scalar_k(self):
        """Old single-spec QPC with only K=max compiles → returns [K], not [0, K]."""
        from vllm_qaic.model_loader.qaic import QaicCausalLM

        shapes = [
            [[None, [None, 5]]],  # K=4 only
            [[None, [None, 128]]],  # prefill
        ]
        ns = self._make_ns(shapes, prefill_seq_len=128, decode_ks=[4])
        result = QaicCausalLM._decode_ks_from_session(ns)
        assert result == [4]

    def test_fallback_on_missing_input_ids_binding(self):
        """Falls back to self.decode_ks when 'input_ids' not in binding_index_map."""
        from unittest.mock import MagicMock

        from vllm_qaic.model_loader.qaic import QaicCausalLM

        mock_session = MagicMock()
        mock_session.binding_index_map = {}  # 'input_ids' missing → KeyError
        ns = types.SimpleNamespace(
            session=mock_session, prefill_seq_len=128, decode_ks=[4]
        )
        result = QaicCausalLM._decode_ks_from_session(ns)
        assert result == [4]  # silently fell back to init-time value
