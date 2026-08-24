# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import os
import time

import evaluate
import nltk
import numpy as np
import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.benchmarks.datasets import MLPerfDataset, is_valid_sequence
from vllm.outputs import RequestOutput

MLPERF_DATASET_PATH = "mgoin/mlperf-inference-llama2-data"


def load_mlperf_dataset(
    num_seqs: int, tokenizer: PreTrainedTokenizerBase, ctx_len: int
) -> tuple[list[str], list[str]]:
    """Streams the MLPerf Llama2 accuracy dataset
    (https://huggingface.co/datasets/mgoin/mlperf-inference-llama2-data)
    from the HF Hub via vLLM's own `MLPerfDataset` loader, returning the
    first `num_seqs` chat-formatted prompts (and their reference answers)
    that fit within `ctx_len` tokens."""
    dataset = MLPerfDataset(
        dataset_path=MLPERF_DATASET_PATH,
        dataset_split="train",
        disable_shuffle=True,
    )

    prompts: list[str] = []
    references: list[str] = []
    for item in dataset.data:
        if len(prompts) >= num_seqs:
            break

        messages = [
            {"role": "system", "content": item["system_prompt"]},
            {"role": "user", "content": item["question"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompt_len = len(tokenizer(prompt).input_ids)
        if not is_valid_sequence(
            prompt_len,
            output_len=0,
            max_prompt_len=ctx_len,
            max_total_len=ctx_len,
            skip_min_output_len_check=True,
        ):
            continue

        prompts.append(prompt)
        references.append(item["output"])
    return prompts, references


def postprocess_text(preds, targets):
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

    return preds, targets


def compute_mlperf_rouge(
    vllm_preds: list[RequestOutput],
    targets: list[str],
    checkpoint_path: str,
    nltk_data: str | None = None,
) -> dict:
    if nltk_data is not None:
        nltk.data.path.append(nltk_data)
    nltk.download("punkt", download_dir=nltk_data)
    nltk.download("punkt_tab", download_dir=nltk_data)
    metric = evaluate.load("rouge")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        model_max_length=2048,
        padding_side="left",
        use_fast=False,
    )

    # LLM.generate() always returns outputs re-sorted into input-submission
    # order, so predictions align positionally with the ground truths built
    # from the same dataset slice.
    targets = targets[: len(vllm_preds)]

    target_required = []
    preds_token_ids = []

    gen_tok_len = 0
    for pred, target in zip(vllm_preds, targets, strict=False):
        target_required.append(target)
        pred_token_ids = pred.outputs[0].token_ids
        gen_tok_len += len(pred_token_ids)
        preds_token_ids.append(pred_token_ids)

    preds_decoded_text = tokenizer.batch_decode(
        preds_token_ids, skip_special_tokens=True
    )

    preds, targets = postprocess_text(preds_decoded_text, target_required)

    result = metric.compute(
        predictions=preds, references=targets, use_stemmer=True, use_aggregator=False
    )
    result = {k: round(np.mean(v) * 100, 4) for k, v in result.items()}
    prediction_lens = [len(pred) for pred in preds]
    gen_num = len(preds)

    result = {
        **result,
        "gen_len": np.sum(prediction_lens),
        "gen_num": gen_num,
        "gen_tok_len": gen_tok_len,
        "tokens_per_sample": round(gen_tok_len / gen_num, 1),
    }
    return result


def calculate_throughput_perf_metrics(
    elapsed_time: float,
    outputs: list[RequestOutput],
) -> dict:
    prompt_num_tokens = 0
    gen_num_tokens = 0
    for output in outputs:
        prompt_num_tokens += len(output.prompt_token_ids)
        gen_num_tokens += len(output.outputs[0].token_ids)
    total_num_tokens = prompt_num_tokens + gen_num_tokens
    num_requests = len(outputs)
    reqps = num_requests / elapsed_time
    gen_tokps = gen_num_tokens / elapsed_time
    tokps = total_num_tokens / elapsed_time

    metrics = dict(
        elapsed_time=elapsed_time,
        prompt_num_tokens=prompt_num_tokens,
        gen_num_tokens=gen_num_tokens,
        total_num_tokens=total_num_tokens,
        num_requests=num_requests,
        reqps=reqps,
        gen_tokps=gen_tokps,
        tokps=tokps,
    )
    return metrics


def calculate_metrics(
    elapsed_time: float,
    outputs: list[RequestOutput],
    mlperf_targets: list[str] | None = None,
    model: str | None = None,
    nltk_data: str | None = None,
) -> dict:
    metrics: dict = calculate_throughput_perf_metrics(elapsed_time, outputs)

    if mlperf_targets is not None:
        rouge_metrics: dict = compute_mlperf_rouge(
            vllm_preds=outputs,
            targets=mlperf_targets,
            checkpoint_path=model,
            nltk_data=nltk_data,
        )
        metrics.update(rouge_metrics)

    return metrics


def log_perf_metrics(id, metrics, device_group):
    pid = os.getpid()
    msg = (
        f"PERFORMANCE METRICS FOR ID {id}, DEVICES: {device_group}, PID: {pid}\n"
        f"\t\tElapsed time: {metrics['elapsed_time']}\n"
        f"\t\tPrompt number of tokens: {metrics['prompt_num_tokens']}\n"
        f"\t\tGenerated number of tokens: {metrics['gen_num_tokens']}\n"
        f"\t\tTotal number of tokens: {metrics['total_num_tokens']}\n"
        f"\t\tTotal number of requests: {metrics['num_requests']}\n"
        f"\t\tThroughput: {metrics['reqps']} requests/s,"
        f" {metrics['gen_tokps']} generated tokens/s,"
        f" {metrics['tokps']} E2E tokens/s\n"
    )
    assert "rouge1" in metrics, "[ERROR] rouge metrics missing from computed metrics"
    rouge_msg = (
        f"\t\trouge1: {metrics['rouge1']}\n"
        f"\t\trouge2: {metrics['rouge2']}\n"
        f"\t\trougeL: {metrics['rougeL']}\n"
        f"\t\trougeLsum: {metrics['rougeLsum']}\n"
        f"\t\tgen_len: {metrics['gen_len']}\n"
        f"\t\tgen_num: {metrics['gen_num']}\n"
        f"\t\tgen_tok_len: {metrics['gen_tok_len']}\n"
        f"\t\ttokens_per_sample: {metrics['tokens_per_sample']}\n"
    )
    msg += rouge_msg
    rouge_1 = metrics["rouge1"]
    rouge_2 = metrics["rouge2"]
    rouge_L = metrics["rougeL"]
    print(msg)
    return rouge_1, rouge_2, rouge_L


class TestAccuracy:
    @pytest.mark.qaic_test_config(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ctx_len=256,
        dtype="mxfp6",
        kv_dtype="mxint8",
    )
    def test_accuracy(
        self,
        qaic_model,
        sampling_params,
        model_name,
        decode_bsz,
        ctx_len,
        device_pool_ids,
    ):
        llm = qaic_model.get_llm()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        inputs, targets = load_mlperf_dataset(
            num_seqs=decode_bsz, tokenizer=tokenizer, ctx_len=ctx_len
        )

        start: float = time.perf_counter()
        outputs = llm.generate(inputs, sampling_params)
        elapsed_time: float = time.perf_counter() - start

        metrics: dict = calculate_metrics(
            elapsed_time,
            outputs,
            mlperf_targets=targets,
            model=model_name,
            nltk_data="./",
        )

        r1, r2, rL = log_perf_metrics(device_pool_ids, metrics, device_pool_ids)
        assert r1 is not None, "[ERROR] Rouge1 score is None"
        assert r2 is not None, "[ERROR] Rouge2 score is None"
        assert rL is not None, "[ERROR] RougeL score is None"
