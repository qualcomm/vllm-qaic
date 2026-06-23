# Feature Support Matrix

Status of vLLM features on Qualcomm Cloud AI hardware.

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Supported and validated |
| 🧪 | Experimental |
| 📝 | Planned |
| ❌ | Not supported |

## Feature Matrix

| Feature | AOT Mode | Eager Mode | Notes |
|---------|----------|------------|-------|
| **Inference** | | | |
| Text generation | ✅ | ✅ | Core serving capability |
| Continuous batching | ✅ | ✅ | |
| **Quantization** | | | |
| mxfp6 | ✅ | ❌ | Hardware-native compute quantization |
| mxint8 KV cache | ✅ | ❌ | `--kv-cache-dtype mxint8` |
| **Speculative Decoding** | | | |
| N-gram | ✅ | ❌ | |
| Suffix | ✅ | ❌ | |
| Draft model | ✅ | ❌ | Separate DLM on same device |
| **Advanced Features** | | | |
| LoRA adapters | ✅ | ❌ | Hot-swap adapters |
| Disaggregated serving | ✅ | ❌ | xEyPzD prefill/decode split |
| Multimodal (VLM) | ✅ | 🧪 | kv_offload architecture |
| Embedding models | ✅ | ❌ | Pooling tasks (Score, embed, classify, rerank) |
| Encoder-decoder | ✅ | ❌ | Whisper |
| Tensor parallelism | ✅ | ✅ | Across QIDs |
| Pipeline parallelism | ✅ | ❌ | Across QIDs |

## Known Constraints

| Constraint | Scope | Notes |
|------------|-------|-------|
| Prefix caching | Both modes | 📝 Planned |
| MLA attention | Both modes | Multi-head Latent Attention — 📝 Planned |
| Async output | Both modes | `supports_async_output` is False — 📝 Planned |

## Feature Combinations

Some features cannot be used together. Here is the compatibility matrix for common combinations:

| Combination | AOT | Eager | Notes |
|-------------|-----|-------|-------|
| SpD + Disaggregated | ✅ | ❌ | Supported on decode node |
| LoRA + Disaggregated | ❌ | ❌ | Not supported |
| Multimodal + CCL (Compiled Context Lengths) | ✅ | 🧪 | Supported (vision encoder on separate device) |
| Tensor Parallel + Disaggregated | ✅ | ❌ | Supported (TP within each node) |
| Quantization + LoRA | ✅ | ❌ | Supported (quantized base + LoRA adapters) |