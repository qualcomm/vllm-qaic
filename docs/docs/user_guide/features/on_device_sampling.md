# On-Device Sampling

On-device sampling (ODS) moves next-token selection onto the Cloud AI accelerator itself, instead of transferring full next-token probability information back to the host for CPU-side sampling. Because the accelerator performs the selection directly, the amount of data transferred from device to host per generation step — and the host-side compute needed to process it — is reduced.

!!! tip "Quick recommendation"
    ODS is a deployment-level opt-in for the ahead-of-time (AOT/precompiled) execution path. It is not a per-request switch — once enabled for a deployment, every request served by that deployment uses device-side token selection.

## Enabling On-Device Sampling

ODS reuses the existing `override_qaic_config` deployment-configuration surface — no new configuration mechanism is introduced. Enable it with `aic_include_sampler`:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_num_seqs=8,
    max_model_len=2048,
    long_prefill_token_threshold=128,
    quantization="mxfp6",
    kv_cache_dtype="mxint8",
    additional_config={
        "override_qaic_config": {
            "aic_include_sampler": 1,
        },
    },
)
```

**Docker example:**

```bash
docker run --rm -it --network host \
  --device /dev/accel/ \
  --shm-size=4gb \
  ghcr.io/quic/cloud_ai_inference_vllm:1.21.2.0 \
  --host 127.0.0.1 --port 8000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --max-model-len 2048 \
  --max-num-seq 8 \
  --max-seq-len-to-capture 128 \
  --quantization mxfp6 \
  --kv-cache-dtype mxint8 \
  --additional-config '{"override_qaic_config":{"aic_include_sampler":1}}'
```

!!! note "AOT mode only"
    `aic_include_sampler` is only supported on the ahead-of-time (AOT/precompiled) execution path. Enabling it on the just-in-time/eager execution path is rejected with a clear error at configuration/start-up time.

For runnable end-to-end usage, use the maintained examples:

- `examples/qaic_ods_offline.py`
- `examples/qaic_ods_online_client.py`

## Top-K Candidate Limit

The `max_top_k_ids` config option caps the top-k candidate count the compiled ODS binding supports:

| Option | Default | Effect |
|--------|---------|--------|
| `max_top_k_ids` | `512` | Maximum `top_k` value accepted for a request. Requests specifying a `top_k` above this limit are **rejected** with a clear, actionable error — the value is never silently clamped. |

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    additional_config={
        "override_qaic_config": {
            "aic_include_sampler": 1,
            "max_top_k_ids": 256,
        },
    },
)
```

## Supported Sampling Controls

From the caller's perspective, sampling behaves the same as without ODS — the only difference is that token selection now happens on the accelerator instead of the host:

| Control | Supported under ODS |
|---------|----------------------|
| Greedy (deterministic) decoding | :white_check_mark: |
| Temperature | :white_check_mark: |
| Top-k | :white_check_mark: (subject to `max_top_k_ids`) |
| Top-p | :white_check_mark: |
| Min-p | :white_check_mark: |
| Repetition penalty | :white_check_mark: |
| Presence penalty | :white_check_mark: |
| Reproducible seed | :white_check_mark: (see [seed limitation](#reproducible-seed-limitation) below) |

Repetition-penalty and presence-penalty state is fully reset whenever a continuous-batching slot is reused by a new, unrelated request, so no prior request's generation history influences a new occupant's output.

### Frequency-penalty limitation

Frequency-penalty is **not** enforced under ODS. This is not a silently-ignored option:

!!! note "Frequency-penalty is warned, not blocked"
    A request specifying a non-default `frequency_penalty` while ODS is active still completes successfully — it is not rejected. The caller receives exactly one warning per request stating that `frequency_penalty` was not applied, and generated output does not reflect the requested frequency-penalty adjustment.

### Reproducible-seed limitation

!!! note "Seed reproducibility is device-side only"
    Reproducible-seed requests under ODS produce device-side-reproducible output for a given seed, but that output is **not guaranteed to match** what the same seed would produce under host-side (non-accelerated) sampling. The two paths run in different execution environments, so bit-for-bit parity between them is not guaranteed. This is a known, documented limitation — not a defect.

## What's Rejected

ODS rejects the following combinations up front, with a clear, actionable error — never a partial failure or silent degradation:

| Combination | Rejected because |
|-------------|-------------------|
| Just-in-time/eager execution mode | ODS requires the ahead-of-time/precompiled path. |
| [Speculative decoding](speculative_decoding.md) (any method) | ODS and speculative decoding are not yet compatible. |
| [Disaggregated serving](disaggregated_serving.md) (split prefill/decode) | ODS and disaggregated serving are not yet compatible. |
| Guided/structured output constraints | Structured/grammar-constrained decoding is not yet supported alongside ODS. |
| Generated-token confidence scores (`logprobs`) | ODS does not materialize full-vocabulary logits on the host, so per-token confidence scores are unavailable. |
| Prompt-token confidence scores (`prompt_logprobs`) | Same reason as above. |

Requests for logprobs or prompt logprobs are rejected at request-validation time, before any generation work begins.

## Debug/Evaluation Sub-Mode

An optional `aic_return_pdfs` config key enables a debug/evaluation sub-mode that returns full probability distributions from the device each step, instead of having the device make the token selection itself. It requires `aic_include_sampler` to also be enabled:

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    additional_config={
        "override_qaic_config": {
            "aic_include_sampler": 1,
            "aic_return_pdfs": 1,
        },
    },
)
```

!!! warning "Not suitable for production"
    `aic_return_pdfs` is a debug/evaluation-only sub-mode intended for offline accuracy comparison against the non-accelerated sampling path. It forfeits the entire point of on-device sampling: instead of reducing device-to-host data transfer and host-side compute, it transfers full probability distributions every step, incurring the same (or greater) transfer/compute cost that ODS is designed to eliminate. Never enable `aic_return_pdfs` in a production deployment.
