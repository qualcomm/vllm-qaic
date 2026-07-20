# Feature Support Matrix

Status of vLLM features on Qualcomm Cloud AI hardware. Features marked :white_check_mark: are production-ready and validated; :test_tube: features are functional but may change without notice.

## Status Legend

| Symbol | Meaning |
|--------|---------|
| :white_check_mark: | Supported and validated |
| :test_tube: | Experimental |
| :clipboard: | Planned |
| :no_entry: | Not supported |

## Feature Matrix

| Feature | AOT Mode | Eager Mode | Notes |
|---------|----------|------------|-------|
| **Inference** | | | |
| Text generation | :white_check_mark: | :white_check_mark: | Core serving capability |
| Continuous batching | :white_check_mark: | :white_check_mark: | |
| **Quantization** | | | |
| mxfp6 | :white_check_mark: | :no_entry: | Hardware-native compute quantization |
| mxint8 KV cache | :white_check_mark: | :no_entry: | `--kv-cache-dtype mxint8` |
| **Speculative Decoding** | | | |
| N-gram | :white_check_mark: | :no_entry: | |
| Suffix | :white_check_mark: | :no_entry: | |
| Draft model | :white_check_mark: | :no_entry: | Separate DLM on same device |
| **Sampling** | | | |
| On-device sampling | :white_check_mark: | :no_entry: | Debug sub-mode (`aic_return_pdfs`) not for production |
| **Advanced Features** | | | |
| LoRA adapters | :white_check_mark: | :no_entry: | Hot-swap adapters |
| Disaggregated serving | :white_check_mark: | :no_entry: | xEyPzD prefill/decode split |
| Multimodal (VLM) | :white_check_mark: | :test_tube: | kv_offload architecture |
| Embedding models | :white_check_mark: | :no_entry: | Pooling tasks (Score, embed, classify, rerank) |
| Encoder-decoder | :white_check_mark: | :no_entry: | Whisper |
| Tensor parallelism | :white_check_mark: | :white_check_mark: | Across QIDs |
| Pipeline parallelism | :white_check_mark: | :no_entry: | Across QIDs |

## Known Limitations

| Limitation | Scope | Notes |
|------------|-------|-------|
| Prefix caching | Both modes | :clipboard: Planned |
| MLA attention | Both modes | Multi-head Latent Attention not supported |
| Async output | Both modes | `supports_async_output` is False |

