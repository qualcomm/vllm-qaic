# Compute Context Length (CCL)

Compute Context Length (CCL) improves performance for large-context inference by dynamically switching between multiple compute context lengths within a single compiled model.

Traditional AOT compilation specializes a model for one fixed context length. As context length increases, throughput drops and TTFT increases — even when actual prompt and generation lengths are small. CCL addresses this by:

- Compiling one model once with multiple context-length specializations
- Dynamically selecting the appropriate compute context length at runtime
- Avoiding unnecessary attention computation over the full maximum context

## Key Properties

| Property | Detail |
|---|---|
| One compilation → one QPC | No recompilation for different prompt sizes |
| QPC size | Unchanged compared to non-CCL models |
| Runtime switching | Automatic and transparent |
| Enable flag | `ccl_enabled=True` |

## Supported Contexts

CCL works across:

- Standard vLLM serving
- SpD with `draft_model` method
- Disaggregated Serving (decode-only)

!!! warning "CCL + SpD limitation"
    CCL with speculative decoding is **only supported using the `draft_model` method**. The `suffix` and `ngram` methods are not compatible with CCL because they require multiple decode specializations, which QEfficient does not currently support with CCL-enabled models.

Performance benefits increase significantly for **8K+ context lengths**.

## How It Works

During compilation the model is specialized for a list of compute context lengths (CCL lists). At runtime, the system selects the appropriate CCL value based on position IDs — KV-cache reads, attention masks, and memory accesses are restricted to the active CCL.

## Usage

### Online Serving

```bash
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 256 \
  --max-num-seqs 16 \
  --long-prefill-token-threshold 128 \
  --quantization mxfp6 \
  --kv-cache-dtype mxint8 \
  --additional-config '{"override_qaic_config": {"ccl_enabled": true}}'
```

No additional flags are required — the CCL list is auto-generated.

### With Speculative Decoding

Enable CCL the same way as standard vLLM. Automatic CCL list generation is supported and recommended — optional prefill and decode lists may be provided but are not required.

### With Disaggregated Serving

!!! warning "Decode only"
    CCL is supported **only during decoding** in disaggregated mode. Prefilling with CCL is not supported.

Provide `comp_ctx_lengths_decode` explicitly:

```bash
--additional-config '{"decode_override_qaic_config": {"ccl_enabled": true, "comp_ctx_lengths_decode": [4096, 8192, 16384, 32768]}}'
```

## Benchmarking

Use standard vLLM benchmarking scripts — no extra flags needed:

```bash
python3 benchmarks/benchmark_serving.py \
  --backend openai \
  --base-url http://127.0.0.1:8000 \
  --dataset-name sharegpt \
  --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --seed 12345
```

## Best Practices

- Compile with a large maximum context (e.g., 128K) if future needs are unknown
- Avoid over-populating CCL lists — auto-generation is recommended
- For disaggregated serving, always provide `comp_ctx_lengths_decode` explicitly
