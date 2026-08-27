# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import gc
from concurrent.futures import ThreadPoolExecutor

import pytest


# ============================================================================
# Expect behavior:
# [main thread] init & warmup model (with 1 prompt generate)
# [executor] start to run & call model to generate 1 prompt
# [main thread] delete model object
# [executor] processed prompts are handled by another process, it'll run to
# end, regardless of deletion happens in the middle
# [main thread] check that the main text and the executor's text are same.
# ============================================================================


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ctx_len=256,
    dtype="mxfp6",
    kv_dtype="mxint8",
)
def test_offline_stability(
    vllm_runner,
    model_name,
    seq_len,
    ctx_len,
    decode_bsz,
    dtype,
    kv_dtype,
    device_group,
    override_qaic_config,
    sampling_params,
):
    prompts = ["My name is"]

    # initialize and warm up run
    model = vllm_runner(
        model_name,
        max_num_seqs=decode_bsz,
        max_model_len=ctx_len,
        long_prefill_token_threshold=seq_len,
        quantization=dtype,
        kv_cache_dtype=kv_dtype,
        enable_prefix_caching=False,
        async_scheduling=False,
        additional_config={
            "device_group": device_group,
            "override_qaic_config": override_qaic_config,
        },
    )

    output = model.generate(prompts, sampling_params)
    main_op = [texts[0] for _, texts in output]

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(model.generate, prompts, sampling_params)

        del model
        gc.collect()

        output = future.result()
        thread_op = [texts[0] for _, texts in output]

    assert main_op == thread_op, (
        "The outputs are not stable before and after the thread creation!!"
    )
