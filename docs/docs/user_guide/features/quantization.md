# Quantization

QAIC supports two quantization formats that are hardware-native on Qualcomm Cloud AI 100.

## Supported Methods

| Method | Format | Use Case | CLI Flag |
|--------|--------|----------|----------|
| **mxfp6** | Microscaling FP6 | Weight and activation computation | `--quantization mxfp6` |
| **mxint8** | Microscaling INT8 | KV cache compression | `--kv-cache-dtype mxint8` |

## Recommended Configuration

For most workloads, use mxfp6 compute with mxint8 KV cache:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --quantization mxfp6 \
  --kv-cache-dtype mxint8 \
  --max-num-seqs 8 \
  --max-model-len 2048
```

## MX Quantization Formats

Qualcomm's Microscaling (MX) formats are hardware-native on Cloud AI 100:

| Format | Bits | Typical Use |
|--------|------|-------------|
| mxfp6 | 6-bit float | Weight and activation computation |
| mxint8 | 8-bit int | KV cache storage (saves memory bandwidth) |

MX formats provide better accuracy-per-bit than standard INT quantization because they use per-block scaling factors. See the [Microscaling Formats paper (arXiv:2310.10537)](https://arxiv.org/abs/2310.10537) for technical details.

## Usage Examples

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    quantization="mxfp6",
    kv_cache_dtype="mxint8",
    max_num_seqs=8,
    max_model_len=2048,
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Hello, world!"], sampling_params)
```

!!! info "QEfficient Reference"
    Quantization is applied during model compilation. See the
    [QEfficient documentation](https://quic.github.io/efficient-transformers/source/quick_start.html)
    for how `mxfp6_matmul` and other quantization options are set at compile time.
