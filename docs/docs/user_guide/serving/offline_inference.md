# Offline Inference

Use the `vllm.LLM` class for batch inference without running a server.

## Basic Usage

```bash
export QAIC_VISIBLE_DEVICES=0
```

=== "AOT Mode"

    ```python
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_num_seqs=4,
        max_model_len=2048,
        long_prefill_token_threshold=128,
        quantization="mxfp6",
        kv_cache_dtype="mxint8",
        async_scheduling=False,
    )

    prompts = [
        "The history of artificial intelligence dates back",
        "Large language models are trained on",
        "The key difference between supervised and unsupervised learning is",
    ]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
        print("-" * 50)
    ```

=== "PYT Mode"

    ```python
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_num_seqs=4,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        async_scheduling=False,
    )

    prompts = [
        "The history of artificial intelligence dates back",
        "Large language models are trained on",
        "The key difference between supervised and unsupervised learning is",
    ]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
        print("-" * 50)
    ```

## Key Parameters

| Parameter | Purpose | QAIC Recommendation |
|-----------|---------|-------------------|
| `max_num_seqs` | Decode batch size | Start with 4-8 |
| `max_model_len` | Max context length | Match QPC compilation ctx_len |
| `long_prefill_token_threshold` | Prefill sequence length | Match QPC seq_len (e.g., 128) |
| `quantization` | Compute precision | `"mxfp6"` for AOT |
| `kv_cache_dtype` | KV cache precision | `"mxint8"` for AOT |
| `gpu_memory_utilization` | Device memory fraction | 0.9-1.0 |

## Sampling Constraints

QAIC supports:

- **Greedy sampling**: `temperature < 1e-5` (or `temperature=0.0`)
- **Random sampling**: `temperature > 0` with `best_of=1`

Not supported:

- `best_of > 1`

## Memory Management

When running multiple inference sessions in a script, explicitly clean up:

```python
import gc

outputs = llm.generate(prompts, sampling_params)
# ... process outputs ...

del llm
gc.collect()

# Now safe to create a new LLM instance
```
