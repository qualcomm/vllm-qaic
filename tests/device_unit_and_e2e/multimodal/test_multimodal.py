# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import random

import pytest
import regex as re
from transformers import AutoProcessor, AutoTokenizer
from QEfficient import QEFFAutoModelForSpeechSeq2Seq

from .conftest import (
    MAX_END_TOKENS,
    INTERNVL_NUM_PATCHES,
    QWEN_PROCESSOR_KWARGS,
    NUM_CORES,
    build_model_input,
    create_qeff_model_and_processor,
    encode_if_mm,
    is_granite,
    is_internvl,
    is_qwenvl,
    prepare_qeff_inputs,
    update_qaic_config,
)

from vllm import SamplingParams


def _tokenizer_for(model_name: str):
    if is_granite(model_name):
        return AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _sampling_params_for(model_name: str, tokenizer) -> SamplingParams:
    stop_token_ids = None
    if is_internvl(model_name):
        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
        stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in stop_tokens]
    return SamplingParams(
        temperature=0.0, max_tokens=MAX_END_TOKENS, stop_token_ids=stop_token_ids
    )


def _mm_processor_kwargs(model_name: str) -> dict:
    if is_qwenvl(model_name):
        return {"mm_processor_kwargs": QWEN_PROCESSOR_KWARGS}
    if is_internvl(model_name):
        return {"mm_processor_kwargs": {"max_dynamic_patch": INTERNVL_NUM_PATCHES - 1}}
    return {}


class _DualQpcTestBase:
    """Model-agnostic "dual-QPC" tests: a pooling/encoder LLM produces image
    embeddings, fed into a separate generation LLM. Each subclass pins one
    vision-language model via its own `qaic_test_config` marker."""

    def test_dual_qpc_single_image(
        self,
        mm_input,
        model_name,
        device_groups,
        make_runner,
        decode_bsz,
        ctx_len,
        seq_len,
        dtype,
        kv_dtype,
    ):
        """Compare vLLM's output with QEfficient's output and ensure they match."""
        tokenizer = _tokenizer_for(model_name)
        sampling_params = _sampling_params_for(model_name, tokenizer)
        updated_input = build_model_input(model_name, mm_input, tokenizer)
        override_cfg = update_qaic_config(model_name, None)
        mm_kwargs = _mm_processor_kwargs(model_name)

        with (
            make_runner(
                async_scheduling=False,
                dg=device_groups[0],
                max_num_seqs=1,
                runner="pooling",
                quantization=None,
                kv_cache_dtype="auto",
                override_qaic_config=override_cfg,
                trust_remote_code=is_internvl(model_name),
                enable_mm_embeds=True,
                limit_mm_per_prompt={"image": 1},
                **mm_kwargs,
            ) as qllm_embed,
            make_runner(
                async_scheduling=False,
                dg=device_groups[1],
                max_num_seqs=decode_bsz,
                override_qaic_config=override_cfg,
                trust_remote_code=is_internvl(model_name),
                enable_mm_embeds=True,
                limit_mm_per_prompt={"image": 1},
                **mm_kwargs,
            ) as qllm_gen,
        ):
            inputs = [
                {"prompt": p, "multi_modal_data": {"image": img}}
                for img, p in updated_input
            ]
            gen_inputs = encode_if_mm(qllm_embed, inputs, model_name)
            outputs = qllm_gen.llm.generate(gen_inputs, sampling_params=sampling_params)
            qllm_output = [op.outputs[0].text for op in outputs]

        qeff_model, processor = create_qeff_model_and_processor(
            model_name,
            device_groups[1],
            seq_len,
            ctx_len,
            dtype,
            kv_dtype,
            kv_offload=True,
            override_cfg=override_cfg,
            tokenizer_obj=tokenizer,
        )

        qeff_output = []
        for img, p in updated_input:
            inputs_proc = prepare_qeff_inputs(model_name, processor, img, p, tokenizer)
            if is_qwenvl(model_name):
                inputs_proc = qeff_model.model.prepare_inputs_for_generation(
                    inputs=inputs_proc,
                    prefill_seq_len=seq_len,
                    batch_size=decode_bsz,
                )
            output = qeff_model.generate(
                inputs=inputs_proc,
                device_ids=device_groups[1],
                generation_len=MAX_END_TOKENS,
            )
            text = processor.tokenizer.batch_decode(
                output.generated_ids, skip_special_tokens=True
            )[0]
            qeff_output.append(text)
        del qeff_model

        assert len(qllm_output) == len(qeff_output) == len(mm_input)
        for o1, o2 in zip(qllm_output, qeff_output, strict=False):
            assert len(o1) > 1
            o1 = o1.strip()
            o2 = o2.strip()[: len(o1)]  # qeff may add additional tokens at the end
            print(o1)
            print(o2)
            assert o1 == o2

    def test_dual_qpc_single_image_cb(
        self,
        mm_input,
        model_name,
        device_groups,
        make_runner,
        decode_bsz,
    ):
        """Compare vLLM's single batch outputs with continuous batching outputs."""
        tokenizer = _tokenizer_for(model_name)
        sampling_params = _sampling_params_for(model_name, tokenizer)
        updated_input = build_model_input(model_name, mm_input, tokenizer)
        override_cfg = update_qaic_config(model_name, None)
        mm_kwargs = _mm_processor_kwargs(model_name)

        inputs = [
            {"prompt": p, "multi_modal_data": {"image": img}}
            for img, p in updated_input
        ]
        repeated_inputs = inputs * 3
        random.shuffle(repeated_inputs)

        with (
            make_runner(
                async_scheduling=False,
                dg=device_groups[0],
                max_num_seqs=1,
                runner="pooling",
                quantization=None,
                kv_cache_dtype="auto",
                override_qaic_config=override_cfg,
                trust_remote_code=is_internvl(model_name),
                enable_mm_embeds=True,
                limit_mm_per_prompt={"image": 1},
                **mm_kwargs,
            ) as qllm_embed,
            make_runner(
                async_scheduling=False,
                dg=device_groups[1],
                max_num_seqs=1,
                override_qaic_config=override_cfg,
                trust_remote_code=is_internvl(model_name),
                enable_mm_embeds=True,
                limit_mm_per_prompt={"image": 1},
                **mm_kwargs,
            ) as qllm_gen,
        ):
            gen_inputs = encode_if_mm(qllm_embed, repeated_inputs, model_name)
            outputs = qllm_gen.llm.generate(gen_inputs, sampling_params=sampling_params)

        with make_runner(
            async_scheduling=False,
            dg=device_groups[1],
            max_num_seqs=decode_bsz,
            override_qaic_config=override_cfg,
            trust_remote_code=is_internvl(model_name),
            enable_mm_embeds=True,
            limit_mm_per_prompt={"image": 1},
            **mm_kwargs,
        ) as qllm_gen_cb:
            outputs_cb = qllm_gen_cb.llm.generate(
                gen_inputs, sampling_params=sampling_params
            )

        assert len(outputs) == len(outputs_cb) == len(repeated_inputs)
        for o1, o2 in zip(outputs, outputs_cb, strict=False):
            assert len(o2.outputs[0].text) > 1
            print(o1.outputs[0].text)
            print(o2.outputs[0].text)
            assert o1.outputs[0].text == o2.outputs[0].text

    def test_dual_qpc_text_only(
        self,
        text_input,
        model_name,
        device_groups,
        make_runner,
        decode_bsz,
    ):
        """Ensure that outputs for text only prompts are consistent."""
        tokenizer = _tokenizer_for(model_name)
        sampling_params = _sampling_params_for(model_name, tokenizer)
        override_cfg = update_qaic_config(model_name, None)
        mm_kwargs = _mm_processor_kwargs(model_name)

        inputs = [
            {"prompt": p}
            for _, p in build_model_input(
                model_name, [(None, t) for t in text_input], tokenizer, num_images=0
            )
        ]
        repeated_inputs = inputs * 3

        with make_runner(
            async_scheduling=False,
            dg=device_groups[1],
            max_num_seqs=decode_bsz,
            override_qaic_config=override_cfg,
            trust_remote_code=is_internvl(model_name),
            enable_mm_embeds=True,
            limit_mm_per_prompt={"image": 1},
            **mm_kwargs,
        ) as qllm_gen:
            outputs = qllm_gen.llm.generate(
                repeated_inputs, sampling_params=sampling_params
            )

        grouped_outputs: dict[str, set[str]] = {}
        for op, ip in zip(outputs, repeated_inputs, strict=False):
            text = op.outputs[0].text
            prompt = ip["prompt"]
            grouped_outputs.setdefault(prompt, set()).add(text)

        assert len(grouped_outputs) == len(inputs)
        for o in grouped_outputs.values():
            assert len(o) == 1
            val = next(iter(o))
            print(val)
            assert len(val) > 1

    def test_dual_qpc_mixed_modality(
        self,
        mm_input,
        text_input,
        model_name,
        device_groups,
        make_runner,
        decode_bsz,
    ):
        """Ensure outputs are consistent when some prompts are text-only and
        others include multimodal data."""
        tokenizer = _tokenizer_for(model_name)
        sampling_params = _sampling_params_for(model_name, tokenizer)
        updated_input = build_model_input(model_name, mm_input, tokenizer)
        override_cfg = update_qaic_config(model_name, None)
        mm_kwargs = _mm_processor_kwargs(model_name)

        inputs = [
            {"prompt": p, "multi_modal_data": {"image": img}}
            for img, p in updated_input
        ]
        inputs += [
            {"prompt": p}
            for _, p in build_model_input(
                model_name, [(None, t) for t in text_input], tokenizer, num_images=0
            )
        ]
        repeated_inputs = inputs * 2
        random.shuffle(repeated_inputs)

        with (
            make_runner(
                async_scheduling=False,
                dg=device_groups[0],
                max_num_seqs=1,
                runner="pooling",
                quantization=None,
                kv_cache_dtype="auto",
                override_qaic_config=override_cfg,
                trust_remote_code=is_internvl(model_name),
                enable_mm_embeds=True,
                limit_mm_per_prompt={"image": 1},
                **mm_kwargs,
            ) as qllm_embed,
            make_runner(
                async_scheduling=False,
                dg=device_groups[1],
                max_num_seqs=decode_bsz,
                override_qaic_config=override_cfg,
                trust_remote_code=is_internvl(model_name),
                enable_mm_embeds=True,
                limit_mm_per_prompt={"image": 1},
                **mm_kwargs,
            ) as qllm_gen,
        ):
            gen_inputs = encode_if_mm(qllm_embed, repeated_inputs, model_name)
            outputs = qllm_gen.llm.generate(gen_inputs, sampling_params=sampling_params)

        grouped_outputs: dict[str, set[str]] = {}
        for op, ip in zip(outputs, repeated_inputs, strict=False):
            text = op.outputs[0].text
            prompt = ip["prompt"]
            grouped_outputs.setdefault(prompt, set()).add(text)

        for o in grouped_outputs.values():
            # A text-only prompt should produce a single unique output; a
            # prompt with images should produce three (one per image).
            assert len(o) == 1 or len(o) == 3
            for val in o:
                print(val)
                assert len(val) > 1


def _multi_image_test(
    self,
    mm_input,
    model_name,
    device_groups,
    make_runner,
    decode_bsz,
):
    """Two images in a single request. Runs twice to verify output consistency."""
    tokenizer = _tokenizer_for(model_name)
    sampling_params = _sampling_params_for(model_name, tokenizer)
    override_cfg = update_qaic_config(model_name, None)
    mm_kwargs = _mm_processor_kwargs(model_name)

    multi_image_input = [([img, img], "Compare the two images.") for img, _ in mm_input]
    built_input = build_model_input(
        model_name, multi_image_input, tokenizer, num_images=2
    )
    inputs = [
        {"prompt": prompt, "multi_modal_data": {"image": imgs}}
        for imgs, prompt in built_input
    ]

    with (
        make_runner(
            async_scheduling=False,
            dg=device_groups[0],
            max_num_seqs=1,
            runner="pooling",
            quantization=None,
            kv_cache_dtype="auto",
            override_qaic_config=override_cfg,
            trust_remote_code=is_internvl(model_name),
            enable_mm_embeds=True,
            limit_mm_per_prompt={"image": 2},
            **mm_kwargs,
        ) as qllm_embed,
        make_runner(
            async_scheduling=False,
            dg=device_groups[1],
            max_num_seqs=decode_bsz,
            override_qaic_config=override_cfg,
            trust_remote_code=is_internvl(model_name),
            enable_mm_embeds=True,
            limit_mm_per_prompt={"image": 2},
            **mm_kwargs,
        ) as qllm_gen,
    ):
        gen_inputs = encode_if_mm(qllm_embed, inputs, model_name)
        outputs1 = qllm_gen.llm.generate(gen_inputs, sampling_params=sampling_params)
        outputs2 = qllm_gen.llm.generate(gen_inputs, sampling_params=sampling_params)

    assert len(outputs1) == len(outputs2) == len(multi_image_input)
    for o1, o2 in zip(outputs1, outputs2, strict=False):
        assert len(o1.outputs[0].text) > 1
        print(o1.outputs[0].text)
        print(o2.outputs[0].text)
        assert o1.outputs[0].text == o2.outputs[0].text


def _qwenvl_multi_resolution_test(
    self,
    mm_input,
    model_name,
    device_groups,
    make_runner,
    decode_bsz,
):
    """Test Qwen2.5VL/Qwen3VL compiled with multiple resolution specializations.
    Images are resized to random sizes. Runs twice to verify output consistency."""
    from .conftest import QWEN_IMAGE_HEIGHT, QWEN_IMAGE_WIDTH

    tokenizer = _tokenizer_for(model_name)
    sampling_params = _sampling_params_for(model_name, tokenizer)

    # Three resolution specializations: base, 2x, and 3x
    heights = [QWEN_IMAGE_HEIGHT, QWEN_IMAGE_HEIGHT * 2, QWEN_IMAGE_HEIGHT * 3]
    widths = [QWEN_IMAGE_WIDTH, QWEN_IMAGE_WIDTH * 2, QWEN_IMAGE_WIDTH * 3]
    override_cfg = {"height": heights, "width": widths}
    mm_kwargs = _mm_processor_kwargs(model_name)

    # Resize each image to a random size. Fixed seed ensures determinism.
    rng = random.Random(42)
    random_sizes = [(rng.randint(100, 2000), rng.randint(100, 2000)) for _ in mm_input]
    updated_input = build_model_input(
        model_name,
        [
            (img.resize((w, h)), question)
            for (img, question), (h, w) in zip(mm_input, random_sizes, strict=False)
        ],
        tokenizer,
    )
    inputs = [
        {"prompt": p, "multi_modal_data": {"image": img}} for img, p in updated_input
    ]

    with (
        make_runner(
            async_scheduling=False,
            dg=device_groups[0],
            max_num_seqs=1,
            runner="pooling",
            quantization=None,
            kv_cache_dtype="auto",
            override_qaic_config=override_cfg,
            trust_remote_code=is_internvl(model_name),
            enable_mm_embeds=True,
            limit_mm_per_prompt={"image": 1},
            **mm_kwargs,
        ) as qllm_embed,
        make_runner(
            async_scheduling=False,
            dg=device_groups[1],
            max_num_seqs=decode_bsz,
            override_qaic_config=override_cfg,
            trust_remote_code=is_internvl(model_name),
            enable_mm_embeds=True,
            limit_mm_per_prompt={"image": 1},
            **mm_kwargs,
        ) as qllm_gen,
    ):
        gen_inputs = encode_if_mm(qllm_embed, inputs, model_name)
        outputs1 = qllm_gen.llm.generate(gen_inputs, sampling_params=sampling_params)
        outputs2 = qllm_gen.llm.generate(gen_inputs, sampling_params=sampling_params)

    assert len(outputs1) == len(outputs2) == len(mm_input)
    for o1, o2 in zip(outputs1, outputs2, strict=False):
        assert len(o1.outputs[0].text) > 1
        print(o1.outputs[0].text)
        print(o2.outputs[0].text)
        assert o1.outputs[0].text == o2.outputs[0].text


@pytest.mark.qaic_test_config(
    model_name="Qwen/Qwen3-VL-2B-Instruct",
    ctx_len=4096,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=2,
    device_group_size=1,
)
class TestQwen3VL(_DualQpcTestBase):
    test_multi_image = _multi_image_test
    test_qwenvl_multi_resolution = _qwenvl_multi_resolution_test


@pytest.mark.qaic_test_config(
    model_name="Qwen/Qwen2.5-VL-2B-Instruct",
    ctx_len=4096,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=2,
    device_group_size=1,
)
class TestQwen25VL(_DualQpcTestBase):
    test_multi_image = _multi_image_test
    test_qwenvl_multi_resolution = _qwenvl_multi_resolution_test


@pytest.mark.qaic_test_config(
    model_name="OpenGVLab/InternVL2_5-1B",
    ctx_len=4096,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=2,
    device_group_size=1,
)
class TestInternVL(_DualQpcTestBase):
    test_multi_image = _multi_image_test


@pytest.mark.qaic_test_config(
    model_name="llava-hf/llava-interleave-qwen-0.5b-hf",
    ctx_len=4096,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=2,
    device_group_size=1,
)
class TestLlava(_DualQpcTestBase):
    pass


@pytest.mark.qaic_test_config(
    model_name="google/gemma-3-4b-it",
    ctx_len=4096,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=2,
    device_group_size=1,
)
class TestGemma3(_DualQpcTestBase):
    pass


@pytest.mark.qaic_test_config(
    model_name="ibm-granite/granite-vision-3.2-2b",
    ctx_len=6144,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=2,
    device_group_size=1,
)
class TestGranite(_DualQpcTestBase):
    pass


@pytest.mark.qaic_test_config(
    model_name="openai/whisper-tiny.en",
    ctx_len=448,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=1,
    device_group_size=1,
)
def test_whisper(
    audio_data, model_name, device_group, make_runner, decode_bsz, ctx_len
):
    updated_input = [(data, "<|startoftranscript|>") for data in audio_data]
    encoder_ctx_len = 1500
    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_END_TOKENS)

    with make_runner(
        async_scheduling=False,
        dg=device_group,
        max_num_seqs=decode_bsz,
        max_num_batched_tokens=encoder_ctx_len,
        hf_overrides={"max_source_positions": encoder_ctx_len},
    ) as qllm:
        inputs = [
            {"prompt": p, "multi_modal_data": {"audio": data}}
            for data, p in updated_input
        ]
        outputs = qllm.llm.generate(inputs, sampling_params=sampling_params)
        qllm_output = [op.outputs[0].text for op in outputs]

    model = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    model.compile(
        num_devices=len(device_group),
        num_cores=NUM_CORES,
        encoder_ctx_len=encoder_ctx_len,
        ctx_len=ctx_len,
        mxfp6_matmul=True,
        aic_enable_depth_first=True,
    )
    processor = AutoProcessor.from_pretrained(model_name)

    qeff_output = []
    for data, prompt in updated_input:
        output = model.generate(
            inputs=processor(data[0], sampling_rate=data[1], return_tensors="pt"),
            device_ids=device_group,
            generation_len=MAX_END_TOKENS,
        )
        qeff_output.append(
            processor.tokenizer.batch_decode(
                output.generated_ids, skip_special_tokens=True
            )[0]
        )
    del model

    assert len(qllm_output) == len(qeff_output) == len(audio_data)
    for o1, o2 in zip(qllm_output, qeff_output, strict=False):
        o1 = re.sub("<[^>]+>", "", o1).strip()  # remove timestamps
        o2 = o2.strip()
        print(o1)
        print(o2)
        assert o1 == o2
