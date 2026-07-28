# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import argparse
import inspect
import os
import random
import model_configs_llm
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "My name is"
    # Add more prompts here
]

random.shuffle(prompts)

# define qpc parameters
ctx_len = 256
seq_len = 128
decode_bsz = 1

# define sampling parameters
sampling_temp = 0.0

# set QAIC specific environment variables
os.environ["VLLM_QAIC_PREFILL_SEQ_LEN"] = str(seq_len)


def test_llm_vllm(model_name: str, tp_size: int, gen_len: int):
    model_name = model_name.replace("--", "/")
    cfg = model_configs_llm.get_config(model_name)
    tp_size = cfg.get("tp_size", tp_size)
    dtype = cfg.get("dtype", "float16")
    tokenizer_mode = cfg.get("tokenizer_mode", "auto")
    trust_remote_code = cfg.get("trust_remote_code", True)
    # Create a LLM.
    llm = LLM(
        model=model_name,
        trust_remote_code=trust_remote_code,
        max_num_seqs=decode_bsz,  # determines decode batch size
        max_model_len=ctx_len,  # ctx_len (does not account for padding,
        # but does account for prompt and generated tokens)
        # quantization="bf16", # Preferred quantization
        # kv_cache_dtype="bfloat16",  # Preferred option to same KV cache
        # and increase performance
        disable_log_stats=False,
        # enable_prefix_caching=False,
        # gpu_memory_utilization=1.0,
        enable_prefix_caching=False,
        enforce_eager=True,
        async_scheduling=False,
        long_prefill_token_threshold=seq_len,
        dtype=dtype,
        tokenizer_mode=tokenizer_mode,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.95),
        # override_qaic_config={"prefill_max_seq_len": 128}
    )

    # Patch tokenizers whose _pad() doesn't accept the padding_side kwarg
    # added in newer transformers versions (e.g. ChatGLMTokenizer).
    tok = llm.get_tokenizer()
    if tok is not None and hasattr(tok, "_pad"):
        sig = inspect.signature(tok.__class__._pad)
        params = sig.parameters
        if "padding_side" not in params and not any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        ):
            _orig_pad = tok.__class__._pad
            tok.__class__._pad = lambda self, *a, padding_side=None, **kw: _orig_pad(
                self, *a, **kw
            )

    samplingParam = SamplingParams(
        temperature=sampling_temp,
        # min_tokens=gen_len,
        max_tokens=gen_len,
    )
    results = llm.generate(
        prompts,
        sampling_params=samplingParam,
    )
    del llm

    for r in results:
        print(f"Prompt Token Id Shape: {len(r.prompt_token_ids)}")
        print(r.outputs[0].text)
        print(f"Output Token shape {len(r.outputs[0].token_ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--gen-len", type=int, default=20)

    args = parser.parse_args()
    test_llm_vllm(**(args.__dict__))
