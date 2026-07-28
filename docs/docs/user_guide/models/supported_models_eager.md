# Supported Models — Eager Mode

Eager (PYT) mode currently supports a focused set of vision-language models via `torch_qaic`, an out-of-tree PyTorch backend for Qualcomm Cloud AI devices.

## Validated Models

| Model | Architecture | Input Types | Devices | Status |
|-------|-------------|-------------|---------|--------|
| Qwen/Qwen2.5-VL-3B-Instruct | Qwen2VLForConditionalGeneration | Text + Image + Video | 1 QID | :test_tube: |
| Qwen/Qwen2.5-VL-7B-Instruct | Qwen2VLForConditionalGeneration | Text + Image + Video | 1 QID | :test_tube: |
| Qwen/Qwen2.5-VL-32B-Instruct | Qwen2VLForConditionalGeneration | Text + Image + Video | 4 QIDs (TP=4) | :test_tube: |
| Qwen/Qwen3-VL-32B-Instruct | Qwen3VLForConditionalGeneration | Text + Image + Video | 4 QIDs (TP=4) | :test_tube: |

??? example "Quick start — Qwen2.5-VL-7B in Eager mode"
    ```python
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        max_num_seqs=4,
        max_model_len=4096,
        enforce_eager=True,
        async_scheduling=False,
    )
    ```

## Status Legend

| Symbol | Meaning |
|--------|---------|
| :white_check_mark: | Validated and passing accuracy tests |
| :test_tube: | Experimental (runs but not fully validated) |

## Key Differences from AOT

| Aspect | AOT | Eager |
|--------|-----|-------|
| Model onboarding | Requires QEfficient compilation support | Any PyTorch model with torch_qaic ops |
| Quantization | mxfp6, mxint8 KV cache | Not supported |
| First-token latency | Depends on QPC compile time (cached) | Immediate startup |
| Text-only LLMs | Broad support | Not validated |
| Speculative decoding | Full support (ngram, suffix, draft) | Not supported |
| Multimodal (VLM) | Supported | Experimental |
