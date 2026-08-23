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


def delete_hf_checkpoint(model_name: str) -> None:
    """Delete the cached HF checkpoint for model_name to free disk between runs.

    Uses the cache configured by HF_HOME / HF_HUB_CACHE. Never raises, since it
    runs during teardown and must not mask a failure from the run itself.

    Skips a model_name that is a local path. Otherwise deletes unconditionally --
    the flag is opt-in, so passing it is the authorisation to evict, and the sweep
    has to bound disk use whether or not the model ran successfully. Ownership is
    deliberately not consulted: a shared cache is group-writable, so the owner of
    a repo dir is just whoever downloaded it first and says nothing about who
    needs it.
    """
    if os.path.isdir(model_name):
        print(f"[cleanup] '{model_name}' is a local path - not deleting")
        return

    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        revisions = []
        for repo in cache_info.repos:
            if repo.repo_type != "model" or repo.repo_id != model_name:
                continue
            revisions.extend(rev.commit_hash for rev in repo.revisions)

        if not revisions:
            print(f"[cleanup] no deletable cache entry for '{model_name}'")
            return

        strategy = cache_info.delete_revisions(*revisions)
        freed = strategy.expected_freed_size_str
        strategy.execute()
        print(f"[cleanup] deleted checkpoint '{model_name}' (freed {freed})")

    except Exception as exc:  # noqa: BLE001 - teardown must not mask the run
        print(f"[cleanup] could not delete '{model_name}': {exc}")


def cleanup(model_name=None, delete_checkpoint=False):
    if delete_checkpoint and model_name:
        delete_hf_checkpoint(model_name)


def test_llm_vllm(
    model_name: str, tp_size: int, gen_len: int, delete_hf_checkpoint=False
):
    model_name = model_name.replace("--", "/")
    cfg = model_configs_llm.get_config(model_name)
    # Resolve through the config module so the TP used here always matches the
    # tp<N> suffix ci_fallback_ops.sh puts on this run's log file.
    tp_size = model_configs_llm.get_tp_size(model_name, tp_size)
    dtype = cfg.get("dtype", "float16")
    tokenizer_mode = cfg.get("tokenizer_mode", "auto")
    trust_remote_code = cfg.get("trust_remote_code", True)
    print(f"Model:{model_name}, TP_SIZE:{tp_size}, DTYPE:{dtype}")
    # Everything below runs under try/finally so the checkpoint is still deleted
    # when the model fails to load -- most failures happen inside LLM(), and
    # without this each failed model would leave its full checkpoint on disk.
    llm = None
    results = []
    try:
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
                tok.__class__._pad = (
                    lambda self, *a, padding_side=None, **kw: _orig_pad(self, *a, **kw)
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
    finally:
        del llm
        cleanup(model_name=model_name, delete_checkpoint=delete_hf_checkpoint)

    for r in results:
        print(f"Prompt Token Id Shape: {len(r.prompt_token_ids)}")
        print(r.outputs[0].text)
        print(f"Output Token shape {len(r.outputs[0].token_ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--gen-len", type=int, default=20)
    parser.add_argument(
        "--delete-hf-checkpoint",
        action="store_true",
        help="Delete the model's cached HF checkpoint once the run is done",
    )

    args = parser.parse_args()
    test_llm_vllm(**(args.__dict__))
