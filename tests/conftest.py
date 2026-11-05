# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/tests/conftest.py
import math
from contextlib import nullcontext, suppress
from typing import Any, TypeVar, cast
from collections.abc import Callable

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from vllm import LLM, SamplingParams
from vllm.config.model import ConvertOption, RunnerOption
from vllm.logprobs import Logprob, PromptLogprobs, SampleLogprobs
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams
from vllm.utils.torch_utils import set_default_torch_num_threads

_M = TypeVar("_M")
_R = TypeVar("_R")

_PromptMultiModalInput = list[_M] | list[list[_M]]

PromptImageInput = _PromptMultiModalInput[Image.Image]
PromptAudioInput = _PromptMultiModalInput[tuple[np.ndarray, int]]
PromptVideoInput = _PromptMultiModalInput[np.ndarray]

# Representation of generated sequence as a tuple of
# * Token ID list
# * String
# * List of top sample logprobs for each sampled token
#
# Assumes prompt logprobs were not requested.
TokensTextLogprobs = tuple[
    list[int], str, list[dict[int, float]] | SampleLogprobs | None
]

# Representation of generated sequence as a tuple of
# * Token ID list
# * String
# * Optional list of top sample logprobs for each sampled token
# * Optional list of top prompt logprobs for each prompt token
#
# Allows prompt logprobs to be requested.
TokensTextLogprobsPromptLogprobs = tuple[
    list[int],
    str,
    list[dict[int, float]] | SampleLogprobs | None,
    list[dict[int, float] | None] | PromptLogprobs | None,
]


class VllmRunner:
    """
    The default value of some arguments have been modified from
    {class}`~vllm.LLM` as follows:

    - `trust_remote_code`: Set to `True` instead of `False` for convenience.
    - `seed`: Set to `0` instead of `None` for test reproducibility.
    - `max_model_len`: Set to `1024` instead of `None` to reduce memory usage.
    - `block_size`: To reduce memory usage, set default to `64` if on XPU
        devices, otherwise default to `16`.
    - `enable_chunked_prefill`: Set to `False` instead of `None` for
      test reproducibility.
    - `enforce_eager`: Set to `False` to test CUDA graph.
    """

    def __init__(
        self,
        model_name: str,
        runner: RunnerOption = "auto",
        convert: ConvertOption = "auto",
        tokenizer_name: str | None = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = True,
        seed: int = 0,
        max_model_len: int | None = 1024,
        dtype: str = "auto",
        disable_log_stats: bool = True,
        tensor_parallel_size: int = 1,
        block_size: int = 16 if not torch.xpu.is_available() else 64,
        enable_chunked_prefill: bool | None = False,
        enforce_eager: bool | None = False,
        # Set this to avoid hanging issue
        default_torch_num_threads: int | None = None,
        **kwargs,
    ) -> None:
        init_ctx = (
            nullcontext()
            if default_torch_num_threads is None
            else set_default_torch_num_threads(default_torch_num_threads)
        )

        if not kwargs.get("compilation_config"):
            # Note(@tdoublep): This is set to 4 because some tests (e.g., hybrid
            # model tests) may set max_num_seqs=4. If min cudagraph_capture_size is
            # set to larger than max_num_seqs, then it will lead to *no* graphs
            # being captured which can trigger edge cases that we don't handle yet.
            kwargs["compilation_config"] = {"cudagraph_capture_sizes": [4]}

            # Make sure we have atleast one cudagraph large enough for a single decode.
            if (speculative_config := kwargs.get("speculative_config")) and (
                num_speculative_tokens := speculative_config["num_speculative_tokens"]
            ):
                kwargs["compilation_config"]["cudagraph_capture_sizes"].append(
                    num_speculative_tokens + 1
                )

        with init_ctx:
            self.llm = LLM(
                model=model_name,
                runner=runner,
                convert=convert,
                tokenizer=tokenizer_name,
                tokenizer_mode=tokenizer_mode,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                seed=seed,
                enforce_eager=enforce_eager,
                disable_log_stats=disable_log_stats,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                block_size=block_size,
                enable_chunked_prefill=enable_chunked_prefill,
                **kwargs,
            )

    def get_inputs(
        self,
        prompts: list[str]
        | list[torch.Tensor]
        | list[list[int]]
        | list[dict[str, Any]],
        images: PromptImageInput | None = None,
        videos: PromptVideoInput | None = None,
        audios: PromptAudioInput | None = None,
    ) -> list[dict[str, Any]]:
        if any(
            x is not None and len(x) != len(prompts) for x in [images, videos, audios]
        ):
            raise ValueError(
                "All non-None multimodal inputs must have the same length as prompts"
            )

        inputs = list[dict[str, Any]]()
        for i, prompt in enumerate(prompts):
            # If we're passing an encoder/decoder prompt, we assume it
            # already contains the multimodal data in the prompt
            if isinstance(prompt, dict):
                assert images is None and audios is None and videos is None
                inputs.append(prompt.copy())
            else:
                prompt_dict = dict[str, Any]()
                if isinstance(prompt, str):
                    prompt_dict["prompt"] = prompt
                elif isinstance(prompt, list):
                    prompt_dict["prompt_token_ids"] = prompt
                else:
                    prompt_dict["prompt_embeds"] = prompt

                multi_modal_data = dict[str, Any]()
                if images is not None and (image := images[i]) is not None:
                    multi_modal_data["image"] = image
                if videos is not None and (video := videos[i]) is not None:
                    multi_modal_data["video"] = video
                if audios is not None and (audio := audios[i]) is not None:
                    multi_modal_data["audio"] = audio

                if multi_modal_data:
                    prompt_dict["multi_modal_data"] = multi_modal_data

                inputs.append(prompt_dict)

        return inputs

    def generate(
        self,
        prompts: list[str] | list[torch.Tensor] | list[list[int]],
        sampling_params: SamplingParams,
        images: PromptImageInput | None = None,
        videos: PromptVideoInput | None = None,
        audios: PromptAudioInput | None = None,
        return_logprobs: bool = False,
        **kwargs: Any,
    ) -> list[tuple[list[list[int]], list[str]]] | tuple[list, list]:
        inputs = self.get_inputs(prompts, images=images, videos=videos, audios=audios)

        req_outputs = self.llm.generate(
            inputs, sampling_params=sampling_params, **kwargs
        )

        outputs: list[tuple[list[list[int]], list[str]]] = []
        logprobs = []
        for req_output in req_outputs:
            prompt_str = req_output.prompt
            prompt_ids = req_output.prompt_token_ids
            req_sample_output_ids: list[list[int]] = []
            req_sample_output_strs: list[str] = []
            req_logprobs = []
            if req_output.prompt_logprobs:
                req_logprobs.extend(req_output.prompt_logprobs)
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = list(sample.token_ids)
                req_sample_output_ids.append(prompt_ids + output_ids)
                req_sample_output_strs.append((prompt_str or "") + output_str)
                if sample.logprobs:
                    req_logprobs.extend(sample.logprobs)
            outputs.append((req_sample_output_ids, req_sample_output_strs))
            logprobs.append(req_logprobs)
        return outputs if not return_logprobs else (outputs, logprobs)

    @staticmethod
    def _final_steps_generate_w_logprobs(
        req_outputs: list[RequestOutput],
        include_prompt_token_ids: bool = False,
    ) -> list[TokensTextLogprobsPromptLogprobs]:
        outputs: list[TokensTextLogprobsPromptLogprobs] = []
        for req_output in req_outputs:
            assert len(req_output.outputs) > 0
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = list(sample.token_ids)
                output_logprobs = sample.logprobs
            if include_prompt_token_ids:
                outputs.append(
                    (  # type: ignore[arg-type]
                        output_ids,
                        output_str,
                        output_logprobs,
                        req_output.prompt_token_ids,
                        req_output.prompt_logprobs,
                    )
                )
            else:
                outputs.append(
                    (
                        output_ids,
                        output_str,
                        output_logprobs,
                        req_output.prompt_logprobs,
                    )
                )

        return outputs

    def generate_w_logprobs(
        self,
        prompts: list[str],
        sampling_params: SamplingParams,
        images: PromptImageInput | None = None,
        audios: PromptAudioInput | None = None,
        videos: PromptVideoInput | None = None,
        include_prompt_token_ids: bool = False,
        **kwargs: Any,
    ) -> list[TokensTextLogprobs] | list[TokensTextLogprobsPromptLogprobs]:
        inputs = self.get_inputs(prompts, images=images, videos=videos, audios=audios)

        req_outputs = self.llm.generate(
            inputs, sampling_params=sampling_params, **kwargs
        )

        toks_str_logsprobs_prompt_logprobs = self._final_steps_generate_w_logprobs(
            req_outputs, include_prompt_token_ids
        )
        # Omit prompt logprobs if not required by sampling params
        return (
            [x[0:-1] for x in toks_str_logsprobs_prompt_logprobs]
            if sampling_params.prompt_logprobs is None
            else toks_str_logsprobs_prompt_logprobs
        )

    def generate_greedy(
        self,
        prompts: list[str] | list[torch.Tensor] | list[list[int]],
        max_tokens: int,
        images: PromptImageInput | None = None,
        videos: PromptVideoInput | None = None,
        audios: PromptAudioInput | None = None,
        **kwargs: Any,
    ) -> list[tuple[list[int], str]]:
        greedy_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outputs = self.generate(
            prompts,
            greedy_params,
            images=images,
            videos=videos,
            audios=audios,
            **kwargs,
        )
        return [(output_ids[0], output_str[0]) for output_ids, output_str in outputs]

    def generate_greedy_logprobs(
        self,
        prompts: list[str],
        max_tokens: int,
        num_logprobs: int | None,
        num_prompt_logprobs: int | None = None,
        images: PromptImageInput | None = None,
        audios: PromptAudioInput | None = None,
        videos: PromptVideoInput | None = None,
        stop_token_ids: list[int] | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> list[TokensTextLogprobs] | list[TokensTextLogprobsPromptLogprobs]:
        greedy_logprobs_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=num_logprobs,
            prompt_logprobs=num_prompt_logprobs,
            stop_token_ids=stop_token_ids,
            stop=stop,
        )

        return self.generate_w_logprobs(
            prompts,
            greedy_logprobs_params,
            images=images,
            audios=audios,
            videos=videos,
            **kwargs,
        )

    def generate_prompt_perplexity(
        self, prompts: list[str], mask: list[str] | None = None
    ) -> list[float]:
        """
        Return the perplexity score associated with generating the prompts

        :param prompts: list of prompts to score
        :return: perplexity score of each prompt
        """
        outputs = self.generate_greedy_logprobs(
            prompts, max_tokens=1, num_logprobs=None, num_prompt_logprobs=0
        )

        mask_prefix_lens = (
            [len(self.llm.get_tokenizer()(prefix)["input_ids"]) for prefix in mask]
            if mask is not None
            else [0 for _ in range(len(prompts))]
        )

        perplexities = []
        for output, mask_prefix_len in zip(outputs, mask_prefix_lens, strict=False):
            output = cast(TokensTextLogprobsPromptLogprobs, output)
            token_data = cast(list[dict[int, Logprob] | None], output[3])
            assert token_data[0] is None

            token_log_probs = []
            for token_logprob_entry in token_data[mask_prefix_len + 1 :]:
                assert token_logprob_entry is not None
                assert len(token_logprob_entry) == 1
                token_log_prob = list(token_logprob_entry.values())[0].logprob
                token_log_probs.append(token_log_prob)

            perplexity = math.exp(-sum(token_log_probs) / len(token_log_probs))
            perplexities.append(perplexity)

        return perplexities

    def generate_beam_search(
        self,
        prompts: list[str],
        beam_width: int,
        max_tokens: int,
        images: PromptImageInput | None = None,
        videos: PromptVideoInput | None = None,
        audios: PromptAudioInput | None = None,
        concurrency_limit: int | None = None,
    ) -> list[tuple[list[list[int]], list[str]]]:
        inputs = self.get_inputs(prompts, images=images, videos=videos, audios=audios)

        outputs = self.llm.beam_search(
            inputs,
            BeamSearchParams(beam_width=beam_width, max_tokens=max_tokens),
            concurrency_limit=concurrency_limit,
        )
        returned_outputs = []
        for output in outputs:
            token_ids = [x.tokens for x in output.sequences]
            texts = [x.text for x in output.sequences]
            returned_outputs.append((token_ids, texts))
        return returned_outputs

    def classify(self, prompts: list[str]) -> list[list[float]]:
        req_outputs = self.llm.classify(prompts)
        return [req_output.outputs.probs for req_output in req_outputs]

    def embed(
        self,
        prompts: list[str],
        images: PromptImageInput | None = None,
        videos: PromptVideoInput | None = None,
        audios: PromptAudioInput | None = None,
        *args,
        **kwargs,
    ) -> list[list[float]]:
        inputs = self.get_inputs(prompts, images=images, videos=videos, audios=audios)

        req_outputs = self.llm.embed(inputs, *args, **kwargs)
        return [req_output.outputs.embedding for req_output in req_outputs]

    def token_embed(self, prompts: list[str]) -> list[list[float]]:
        req_outputs = self.llm.encode(prompts, pooling_task="token_embed")
        return [req_output.outputs.data for req_output in req_outputs]

    def token_classify(self, prompts: list[str]) -> list[list[float]]:
        req_outputs = self.llm.encode(prompts, pooling_task="token_classify")
        return [req_output.outputs.data for req_output in req_outputs]

    def reward(self, prompts: list[str]) -> list[list[float]]:
        req_outputs = self.llm.encode(prompts, pooling_task="token_classify")
        return [req_output.outputs.data for req_output in req_outputs]

    def score(
        self,
        text_1: list[str] | str,
        text_2: list[str] | str,
        *args,
        **kwargs,
    ) -> list[float]:
        req_outputs = self.llm.score(text_1, text_2, *args, **kwargs)
        return [req_output.outputs.score for req_output in req_outputs]

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        return self.llm.apply_model(func)

    def get_llm(self) -> LLM:
        return self.llm

    def collective_rpc(self, *args, **kwargs):
        return self.llm.collective_rpc(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Explicitly shutdown the engine core to release GPU resources
        # This is needed because when executing consecutive tests, the GC
        # might not be fast enough in shutting down the llm engine. This can
        # lead to OOMs because when the next test starts some GPU memory is
        # still in use.
        with suppress(Exception):
            self.llm.llm_engine.engine_core.shutdown()
        del self.llm
        # Import lazily (rather than at module load time) so this picks up the
        # qaic plugin's patched cleanup_dist_env_and_memory, which is applied
        # during platform registration, after this module is first collected.
        from vllm.distributed import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()


@pytest.fixture(scope="session")
def vllm_runner():
    return VllmRunner
