# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""CPU protocol tests with real tiny Qwen3.5 models, not recorded logits.

Only the QAIC session is replaced. The VLM concatenation/commit code, model
forward and rejection sampler run normally. Controlled drafts exercise zero,
partial and full acceptance; pretrained DLM/device parity belongs in the e2e suite.
"""

import copy
from itertools import cycle
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from QEfficient import QEFFAutoModelForImageTextToText
from transformers import (
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeConfig,
    Qwen3_5MoeForConditionalGeneration,
)

from vllm_qaic.model_loader.qaic_multimodal import QaicMultiModal

from .test_patch_rejection_sampler import (
    _make_rejection_sampler,
    _make_sampling_metadata,
    _make_spec_metadata,
    rs_kernels,  # noqa: F401
)

BLOCK = 16
PREFILL = 32
CTX = 512
VOCAB = 500


def _tiny_vlm(moe):
    config_cls = Qwen3_5MoeConfig if moe else Qwen3_5Config
    model_cls = (
        Qwen3_5MoeForConditionalGeneration if moe else Qwen3_5ForConditionalGeneration
    )
    config = config_cls(
        text_config={
            "vocab_size": VOCAB,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "layer_types": ["linear_attention", "full_attention"] * 2,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "max_position_embeddings": CTX,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 1.0,
                "mrope_section": [3, 3, 2],
            },
            **(
                {
                    "num_experts": 4,
                    "num_experts_per_tok": 2,
                    "moe_intermediate_size": 32,
                    "shared_expert_intermediate_size": 32,
                }
                if moe
                else {}
            ),
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
    )
    config._attn_implementation = "eager"
    return model_cls(config).eval()


@pytest.mark.usefixtures("rs_kernels")
@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
@pytest.mark.parametrize("with_image", [False, True], ids=["text", "image"])
@pytest.mark.parametrize("generation_len", [32, 256])
@pytest.mark.parametrize("seed", [7, 23])
@torch.no_grad()
def test_qwen35_long_greedy_matches_vanilla(moe, with_image, generation_len, seed):
    torch.manual_seed(seed)
    hf = _tiny_vlm(moe)
    reference = copy.deepcopy(hf)
    qaic_config = {"target_layer_ids": [1, 3], "speculative_model_type": "target"}
    qeff = QEFFAutoModelForImageTextToText(hf, kv_offload=True, qaic_config=qaic_config)
    if moe:
        qeff.transform(
            ctx_len=CTX,
            seq_len=PREFILL,
            bs=1,
            num_devices=1,
            aic_num_cores=4,
            qaic_config=qaic_config,
            prefill_only=True,
            prefill_seq_len=PREFILL,
        )
    wrapper = qeff.lang_model.model
    wrapper.language_model.dflash_chunk_size = BLOCK
    inputs = qeff.model.get_dummy_inputs(kv_offload=True, ctx_len=CTX)["lang"]
    cache = inputs["past_key_values"]
    image_idx = torch.zeros((1, 1), dtype=torch.long)
    prompt = torch.randint(10, VOCAB, (1, PREFILL + 3))
    prompt_positions = torch.arange(prompt.shape[1]).view(1, 1, -1).repeat(4, 1, 1)
    vision = torch.zeros((1, 2, 32))
    if with_image:
        prompt[:, 1:3] = hf.config.image_token_id
        # The two image tokens share time/height but have distinct widths.
        prompt_positions[1:3, :, 2] = 1
        # Real vision encoder output is shared by vanilla and DFlash.
        vision = (
            reference.model.get_image_features(
                torch.randn(8, 3 * 2 * 16 * 16), torch.tensor([[1, 2, 4]])
            )
            .pooler_output[0]
            .unsqueeze(0)
        )
    embeddings = reference.get_input_embeddings()(prompt)
    if with_image:
        embeddings[:, 1:3] = vision

    ref = reference.model.language_model(
        inputs_embeds=embeddings, position_ids=prompt_positions, use_cache=True
    )
    expected = [reference.lm_head(ref.last_hidden_state[:, -1]).argmax(-1).item()]
    for pos in range(prompt.shape[1], prompt.shape[1] + generation_len - 1):
        ref = reference.model.language_model(
            input_ids=torch.tensor([[expected[-1]]]),
            past_key_values=ref.past_key_values,
            cache_position=torch.tensor([pos]),
            use_cache=True,
        )
        expected.append(
            reference.lm_head(ref.last_hidden_state[:, -1]).argmax(-1).item()
        )

    for start in range(0, prompt.shape[1], PREFILL):
        count = min(PREFILL, prompt.shape[1] - start)
        ids = torch.zeros((1, PREFILL), dtype=torch.long)
        ids[:, :count] = prompt[:, start : start + count]
        positions = torch.full((4, 1, PREFILL), -1, dtype=torch.long)
        positions[:, :, :count] = prompt_positions[:, :, start : start + count]
        prefill = wrapper(
            input_ids=ids,
            vision_embeds=vision,
            image_idx=image_idx,
            position_ids=positions,
            past_key_values=cache,
            cache_commit=torch.ones(1, dtype=torch.long),
            num_logits_to_keep=torch.zeros((1, 1), dtype=torch.long),
        )
        image_idx, cache = prefill[2], prefill[4]

    model = QaicMultiModal.__new__(QaicMultiModal)
    model.dflash_hybrid = True
    model.num_spec_tokens = model.active_k = BLOCK - 1
    model.decode_bsz = 1
    model.uses_mrope = True
    model.ignore_batch_index = False
    model.is_spec_decode_target_model = True
    model.comp_ctx_lengths_decode = None
    model.use_async_scheduling = False
    model.tlm_hidden_states_name = "tlm_hidden_states"
    model.decode_batch_inputs_by_k = {
        BLOCK - 1: {
            "input_ids": np.zeros((1, 2 * BLOCK), dtype=np.int64),
            "position_ids": np.full((4, 1, 2 * BLOCK), -1, dtype=np.int64),
            "batch_index": np.zeros((1, 1), dtype=np.int64),
            "cache_commit": np.zeros(1, dtype=np.int64),
        }
    }
    model.decode_num_logits_buffer_by_k = {
        BLOCK - 1: {"num_logits_to_keep": np.zeros((BLOCK, 1), dtype=np.int64)}
    }
    model._dflash_prefix_ids = np.zeros((1, BLOCK), dtype=np.int64)
    model._dflash_prefix_positions = np.full((4, 1, BLOCK), -1, dtype=np.int64)
    model._dflash_decode_hidden = np.zeros((1, 2 * BLOCK, 32), dtype=np.float32)
    calls = 0

    def cpu_session(batch_inputs, is_prefill):
        nonlocal cache, calls
        assert not is_prefill
        calls += 1
        before = copy.deepcopy(cache)
        out = wrapper(
            input_ids=torch.from_numpy(batch_inputs["input_ids"]),
            position_ids=torch.from_numpy(batch_inputs["position_ids"]),
            vision_embeds=vision,
            image_idx=image_idx,
            past_key_values=cache,
            cache_commit=torch.from_numpy(batch_inputs["cache_commit"]),
            num_logits_to_keep=torch.from_numpy(batch_inputs["num_logits_to_keep"]),
        )
        cache = out[4]
        # Verification may read its suffix, but retained KV may only commit
        # the previous accepted prefix. Check both full-attention layers.
        committed = batch_inputs["position_ids"][0, 0, :BLOCK]
        untouched = ~np.isin(np.arange(CTX), committed[committed >= 0])
        for layer in (1, 3):
            for new, old in zip(cache[layer], before[layer], strict=True):
                torch.testing.assert_close(
                    new[:, :, untouched], old[:, :, untouched], rtol=0, atol=0
                )
        if np.all(committed < 0):
            for layer in (0, 2):
                torch.testing.assert_close(cache[layer], before[layer], rtol=0, atol=0)
        batch_inputs["logits"][:] = out[0].numpy()
        batch_inputs["tlm_hidden_states"][:] = out[3].numpy()
        return calls

    model.session = SimpleNamespace(np_run=cpu_session, complete_inf=Mock())
    sampler = _make_rejection_sampler()
    actual = [prefill[0].argmax(-1).item()]
    assert actual == expected[:1]
    counts = []
    for accepted in cycle([0, 1, 7, BLOCK - 1]):
        if len(actual) >= generation_len:
            break
        valid = min(BLOCK, generation_len - len(actual))
        drafts = expected[len(actual) : len(actual) + valid - 1].copy()
        if accepted < len(drafts):
            drafts[accepted:] = [(token + 1) % VOCAB for token in drafts[accepted:]]
        ids = np.zeros(BLOCK, dtype=np.int64)
        ids[:valid] = [actual[-1], *drafts]
        pos = prompt.shape[1] + len(actual) - 1
        positions = np.tile(np.arange(pos, pos + BLOCK), (4, 1))
        positions[:, valid:] = -1
        logits = np.empty((1, BLOCK, VOCAB), dtype=np.float32)
        model._run_decode(ids, positions, np.array([0]), logits)
        output = (
            sampler(
                _make_spec_metadata([drafts]),
                None,
                torch.from_numpy(logits[0, :valid].copy()),
                _make_sampling_metadata([0.0], seed=0),
            )
            .sampled_token_ids[0]
            .tolist()
        )
        output = [token for token in output if token >= 0]
        assert len(output) == min(accepted + 1, valid), (
            f"offset={len(actual)}, seed_token={actual[-1]}, drafts={drafts}, "
            f"expected={expected[len(actual) : len(actual) + valid]}, "
            f"target={logits[0, :valid].argmax(-1).tolist()}, sampled={output}"
        )
        counts.append(len(output))
        model.queue_dflash_prefix([output])
        actual.extend(output)
        assert actual == expected[: len(actual)]
    assert len(actual) == generation_len
    assert actual == expected
    assert calls == len(counts)
    assert model.session.complete_inf.call_count == calls
    if generation_len == 256:
        assert {1, 2, 8, 16} <= set(counts)
