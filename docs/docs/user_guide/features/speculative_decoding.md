# Speculative Decoding

Speculative decoding (SpD) accelerates token generation by using a fast proposer to draft candidate tokens that the target model verifies in a single forward pass. When acceptance rates are high, effective tokens-per-second during decode can increase significantly (actual improvement is workload-dependent).

!!! tip "Quick recommendation"
    Start with **ngram** SpD — it requires no separate model binary or dedicated core allocation and works well for summarization and conversational workloads. Switch to **draft_model** if you need higher acceptance rates on diverse generation tasks.

!!! note "Triton-CPU backend required for AOT SpD"
    For AOT mode, the rejection sampler uses Triton kernels running on CPU.
    Install the triton-cpu backend during setup:
    ```bash
    TRITON_CPU=1 ./scripts/install.sh aot
    # Or standalone after installation:
    ./scripts/install_triton_cpu.sh
    ```

## Methods

Cloud AI supports three speculative decoding methods:

| Method | Proposer | Separate Model Binary | Dedicated Core Allocation | Best For |
|--------|----------|-------------|-------------|----------|
| `ngram` | N-gram pattern matching | No | No | Summarization, repetitive output |
| `suffix` | Suffix array matching | No | No | Code completion, long-context echo |
| `draft_model` | Lightweight LLM | Yes | Yes | General text generation |

## N-gram Speculative Decoding

Requires no separate model binary and no dedicated core allocation — runs within the target model's existing resources:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_num_seqs=8,
    max_model_len=2048,
    long_prefill_token_threshold=128,
    quantization="mxfp6",
    kv_cache_dtype="mxint8",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
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
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --max-model-len 256 \
  --max-num-seq 16 \
  --max-seq-len-to-capture 128 \
  --quantization mxfp6 \
  --kv-cache-dtype mxint8 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":5}'
```

## Draft-Model Speculative Decoding

Uses a lightweight model (e.g., Llama-3.2-1B) to propose tokens for a larger target model (e.g., Llama-3.1-8B):

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_num_seqs=4,
    max_model_len=2048,
    long_prefill_token_threshold=128,
    quantization="mxfp6",
    kv_cache_dtype="mxint8",
    additional_config={
        "override_qaic_config": {
            "device_group": [0, 1, 2, 3],
            "num_cores": 10,       # 10 cores for target model
        },
        "draft_override_qaic_config": {
            "device_group": [0, 1, 2, 3],   # Same device
            "num_cores": 6,        # 6 cores for draft model
        },
    },
    speculative_config={
        "method": "draft_model",
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "num_speculative_tokens": 3,
    },
)
```

**Docker example:**

```bash
docker run --rm -it --network host \
  --device /dev/accel/ \
  --shm-size=4gb \
  -e HF_TOKEN=<your_hf_token> \
  ghcr.io/quic/cloud_ai_inference_vllm:1.21.2.0 \
  --host 127.0.0.1 --port 8000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --max-model-len 2048 \
  --max-num-seq 4 \
  --max-seq-len-to-capture 128 \
  --quantization mxfp6 \
  --kv-cache-dtype mxint8 \
  --speculative-config '{"method":"draft_model","model":"meta-llama/Llama-3.2-1B-Instruct","num_speculative_tokens":3}' \
  --additional-config '{"override_qaic_config":{"device_group":[0,1,2,3],"num_cores":10},"draft_override_qaic_config":{"device_group":[0,1,2,3],"num_cores":6}}'
```

!!! info "QEfficient Reference"
    Draft model compilation requires separate QPC generation. See the
    [QEfficient SpD Guide](https://quic.github.io/efficient-transformers/speculative_decoding.html)
    for compilation details.

## Core Allocation

On a Cloud AI 100 Ultra card (4 QIDs, 16 cores per QID), cores are split between the target model (TLM) and draft model (DLM) on each QID:

```text
Per QID (16 NSP cores):
+--------------------+----------------+
| Target Model (10)  | Draft Model (6)|
+--------------------+----------------+

Applied across QID 0-3 -- both models share the same device group.
```

No additional hardware is required — both models run on the same device.

!!! note "Default allocation"
    When no explicit `num_cores` configuration is provided, cores are split **8/8** (equal between target and draft). The 10/6 split shown above is a recommended allocation for an 8B target + 1B draft.

## Performance Tuning

??? tip "Tuning parameters"
    | Parameter | Effect | Recommendation |
    |-----------|--------|----------------|
    | `num_speculative_tokens` | Tokens proposed per step | 3-5 (higher = more aggressive) |
    | `override_qaic_config.num_cores` | Compute for verification (target) | 10-12 for 8B models |
    | `draft_override_qaic_config.num_cores` | Compute for proposing (draft) | 4-6 for 1B models |
    | `max_num_seqs` | Batch size | Lower batch = higher acceptance rate |

## SpD with Disaggregated Serving

SpD can be combined with disaggregated serving — proposals are generated and verified entirely on the decode node. See [Disaggregated Serving](disaggregated_serving.md#speculative-decoding-with-disaggregated-serving).
