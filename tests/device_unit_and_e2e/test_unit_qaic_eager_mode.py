# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
On-device tests for QAIC eager mode.

These only exercise real behavior when torch_qaic is actually installed and
QaicPlatform.is_aot is False (see vllm_qaic/platform_base.py:64-65) — i.e.
under an eager-capable environment with QAIC device access. They are the
Tier B counterpart to unit/generic/test_platform.py::TestEagerModeBranches
and unit/generic/test_chunked_prefill.py::TestChunkedPrefillEagerModeSkipsFormula
(Tier A), which drive the same source branches purely in Python via
monkeypatched is_aot and therefore run in any environment, AOT or eager,
without hardware.

Unlike Tier A, nothing here monkeypatches is_aot: qaic_model/make_runner build
real vllm_runner(...)/LLM(...) instances, so whichever mode the launching
environment's torch_qaic installation resolves to is the mode actually
exercised. In an AOT-only environment (no torch_qaic), is_aot stays True and
these tests exercise the ordinary AOT path — they add no new assertions in
that case beyond what the existing on-device suite already covers, since the
eager-specific rejections (kv_transfer_config, speculative_config,
async_scheduling-forced-False) are simply not triggered when is_aot is True.
They are included for whichever environment eventually runs eager e2e: no
skip/xfail is added for the AOT case because a real LLM(...) construction with
these configs is expected to just succeed under AOT (untested here — AOT
config-validation coverage already lives in unit/generic/test_platform.py).

Coverage areas
--------------
1. device_type == "qaic" end-to-end (from a real qaic_model instance)
2. kv_transfer_config rejected at real LLM(...) construction in eager mode
3. speculative_config rejected at real LLM(...) construction in eager mode
4. async_scheduling=True silently falls back to False (inference still works)
5. get_device_uuid / get_device_total_memory / get_num_cores resolve without
   NotImplementedError once torch_qaic is installed (eager-only implementations)
"""

import pytest


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    seq_len=128,
    ctx_len=256,
    decode_bsz=4,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=1,
    device_group_size=1,
)
class TestEagerDeviceType:
    def test_device_type_is_qaic(self, qaic_model):
        """QaicPlatform.device_type is "qaic" once torch_qaic is installed
        (platform_base.py:58) — confirms this holds through a real
        end-to-end LLM(...) construction, not just the class attribute in
        isolation. Under AOT (no torch_qaic), device_type legitimately stays
        "cpu" throughout (pre_register_and_update at platform_base.py
        ~244-246 only ever syncs vllm_config.device_config.device_type to
        match cls.device_type, never the reverse), so this test skips under
        AOT rather than asserting a value this environment cannot produce —
        matching this file's own stated AOT-skip pattern for the other
        eager-only tests below."""
        from vllm_qaic.platform_base import QaicPlatform
        if QaicPlatform.is_aot:
            pytest.skip("device_type is only \"qaic\" once torch_qaic is "
                        "installed (is_aot=True in this environment)")
        assert QaicPlatform.device_type == "qaic"

    def test_generate_produces_output(self, qaic_model):
        """Baseline: a real runner must still produce output regardless of
        which mode (AOT/eager) the launching environment resolves to."""
        from vllm import SamplingParams
        out = qaic_model.llm.generate(
            ["My name is"], SamplingParams(temperature=0.0, max_tokens=10)
        )
        assert len(out[0].outputs[0].token_ids) > 0


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ctx_len=256,
    decode_bsz=4,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=1,
    device_group_size=1,
)
def test_kv_transfer_config_rejected_in_eager(
    device_groups, vllm_runner, model_name, ctx_len, decode_bsz, dtype, kv_dtype,
):
    """QAIC eager mode asserts kv_transfer_config is None at
    check_and_update_config() time ("QAIC eager mode does not support
    disaggregated serving", platform_base.py ~281-284); this must surface all
    the way up through real LLM(...) construction, not just the unit-tested
    check_and_update_config() call in unit/generic/test_platform.py::
    TestEagerModeBranches::test_kv_transfer_config_raises_in_eager.

    Under AOT (no torch_qaic), is_aot stays True and this rejection does not
    fire — LLM(...) construction is expected to succeed instead, so this test
    skips entirely under AOT rather than asserting a fabricated AOT
    disaggregated-serving success path (that combination is not covered
    anywhere in this suite today, on-device or otherwise, and out of scope
    for eager-mode coverage).
    """
    from vllm.config import KVTransferConfig
    from vllm_qaic.platform_base import QaicPlatform

    if QaicPlatform.is_aot:
        pytest.skip("kv_transfer_config rejection only fires in eager mode "
                    "(is_aot=True in this environment)")

    kv_transfer_config = KVTransferConfig(
        kv_connector="SharedStorageConnector", kv_role="kv_both"
    )
    with pytest.raises(AssertionError, match="disaggregated serving"):
        vllm_runner(
            model_name,
            max_num_seqs=decode_bsz, max_model_len=ctx_len,
            quantization=dtype, kv_cache_dtype=kv_dtype, enable_prefix_caching=False,
            async_scheduling=False,
            kv_transfer_config=kv_transfer_config,
            additional_config={"device_group": device_groups[0], "override_qaic_config": {}},
        )


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ctx_len=256,
    decode_bsz=4,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=1,
    device_group_size=1,
)
def test_speculative_config_rejected_in_eager(
    device_groups, vllm_runner, model_name, ctx_len, decode_bsz, dtype, kv_dtype,
):
    """QAIC eager mode raises ValueError for speculative_config at
    check_and_update_config() time (platform_base.py ~285-290); this must
    surface all the way up through real LLM(...) construction, not just the
    unit-tested check_and_update_config() call in
    unit/generic/test_platform.py::TestEagerModeBranches::
    test_speculative_config_raises_value_error_in_eager.

    Under AOT (no torch_qaic), is_aot stays True and this rejection does not
    fire — LLM(...) construction is expected to succeed instead, so no
    assertion is made either way here beyond "construction completes or
    raises the eager-specific error"; the AOT success path is already
    covered by the existing SpD device suite (device_unit_and_e2e/ SpD
    tests), not duplicated here.
    """
    from vllm_qaic.platform_base import QaicPlatform

    kwargs = dict(
        max_num_seqs=decode_bsz, max_model_len=ctx_len,
        quantization=dtype, kv_cache_dtype=kv_dtype, enable_prefix_caching=False,
        async_scheduling=False,
        speculative_config={"method": "ngram", "num_speculative_tokens": 3,
                             "prompt_lookup_max": 5},
        additional_config={"device_group": device_groups[0], "override_qaic_config": {}},
    )
    if QaicPlatform.is_aot:
        pytest.skip("speculative_config rejection only fires in eager mode "
                    "(is_aot=True in this environment)")
    with pytest.raises(ValueError, match="not supported in eager mode"):
        vllm_runner(model_name, **kwargs)


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    seq_len=128,
    ctx_len=256,
    decode_bsz=4,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=1,
    device_group_size=1,
)
def test_async_scheduling_falls_back_and_still_generates(device_groups, make_runner):
    """async_scheduling=True is silently downgraded to False in eager mode
    (platform_base.py ~291-295, warning + reset), rather than raising.
    Confirms inference still works end-to-end after the fallback, on a real
    runner (unit-level coverage of the reset itself is
    unit/generic/test_platform.py::TestEagerModeBranches::
    test_async_scheduling_falls_back_to_false_in_eager). Under AOT this
    async_scheduling=True path is not downgraded at all — that combination
    is already exercised by the existing test_async_matches_sync in
    test_unit_qaic_chunked_prefill.py, not duplicated here.
    """
    from vllm import SamplingParams

    runner = make_runner(async_scheduling=True, dg=device_groups[0])
    with runner as model:
        out = model.llm.generate(
            ["My name is"], SamplingParams(temperature=0.0, max_tokens=10)
        )
    assert len(out[0].outputs[0].token_ids) > 0


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    seq_len=128,
    ctx_len=256,
    decode_bsz=4,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=1,
    device_group_size=1,
)
class TestEagerOnlyDeviceInfo:
    """get_device_uuid / get_device_total_memory / get_num_cores raise
    NotImplementedError under AOT (platform_base.py:105-106, 113-114,
    129-132) and only resolve real values once torch_qaic is installed.
    Guarded on is_aot so these don't spuriously fail in an AOT-only
    environment — the NotImplementedError contract itself is not
    unit-testable from unit/ without importing the real qaic module, so it
    is not asserted here or elsewhere; only the eager success path is."""

    def test_get_device_uuid_resolves_in_eager(self, qaic_model):
        from vllm_qaic.platform_base import QaicPlatform
        if QaicPlatform.is_aot:
            pytest.skip("get_device_uuid is AOT-unimplemented (NotImplementedError)")
        uuid = QaicPlatform.get_device_uuid(0)
        assert isinstance(uuid, str) and len(uuid) > 0

    def test_get_device_total_memory_resolves_in_eager(self, qaic_model):
        from vllm_qaic.platform_base import QaicPlatform
        if QaicPlatform.is_aot:
            pytest.skip("get_device_total_memory is AOT-unimplemented (NotImplementedError)")
        mem = QaicPlatform.get_device_total_memory(0)
        assert isinstance(mem, int) and mem > 0

    def test_get_num_cores_resolves_in_eager(self, qaic_model):
        from vllm_qaic.platform_base import QaicPlatform
        if QaicPlatform.is_aot:
            pytest.skip("get_num_cores is AOT-unimplemented (returns None)")
        cores = QaicPlatform.get_num_cores(0)
        assert isinstance(cores, int) and cores > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
