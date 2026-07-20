# Engine Arguments

QAIC-specific arguments passed to the vLLM engine via CLI flags or the Python API.

## Common Flags (Quick Reference)

| Flag | Value | Purpose |
|------|-------|---------|
| `--quantization` | `mxfp6` | Hardware-native compute quantization |
| `--kv-cache-dtype` | `mxint8` | Compressed KV cache for memory efficiency |
| `--max-num-seqs` | 4-16 | Decode batch size (start low, scale up) |
| `--max-model-len` | varies | Max context length (prompt + completion) |
| `--additional-config` | JSON | QAIC-specific config (device group, cores) |

---

## `additional_config`

The `additional_config` dictionary holds QAIC-specific configuration.

**Speculative decoding example:**

```python
LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    additional_config={
        "override_qaic_config": {
            "device_group": [0, 1, 2, 3],  # QID(s) for the target model
            "num_cores": 10,               # NSP cores per device for target
        },
        "draft_override_qaic_config": {    # For speculative decoding
            "device_group": [0, 1, 2, 3],  # QID(s) for the draft model
            "num_cores": 6,                # Cores allocated to draft
        },
    },
)
```

**Multimodal (VLM) example:**

```python
LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    additional_config={
        "override_qaic_config": {
            "device_group": [0, 1, 2, 3],
            "num_cores": 16,
        },
        "height": [364, 512],              # VLM: compiled image heights
        "width": [532, 910],               # VLM: compiled image widths
    },
)
```

!!! warning "SpD and Multimodal are mutually exclusive"
    Speculative decoding cannot be combined with multimodal models. Do not include
    `draft_override_qaic_config` alongside `height`/`width` in the same configuration.

### `override_qaic_config` Fields

All `qaic-compile` arguments can be passed as input arguments. The table below lists the key supported options:

| # | Field | Default | Description |
|---|-----------|---------|-------------|
| 1 | `num_cores`, `aic_num_cores` | `16` (or `8` for SpD draft on same device group) | Number of NSP cores |
| 2 | `dfs`, `aic_enable_depth_first` | `True` | Depth-first scheduling, To disable use dfs = False |
| 3 | `mos` | `-1` | Degree of weight splitting across cores to reduce on-chip memory |
| 4 | `num_devices` | — | Number of devices for auto-device mode. Provide either `num_devices` or explicit QiDs using `device_group` |
| 5 | `mdts_mos` | — | Degree of weight splitting across multi-device tensor slices to improve memory and compute efficiency |
| 6 | `mxint8`, `mxint8_en`, `mxint8_kv_cache` | — | MXINT8 compression of MDP IO traffic. Prefer `--kv-cache-dtype mxint8` vLLM argument |
| 7 | `mxfp6`, `mxfp6_matmul`, `mxfp6_en` | — | Compress MatMul weights to MXFP6 E2M3. Prefer `--quantization mxfp6` vLLM argument |
| 8 | `device_group` | — | List of device IDs |
| 9 | `embed_seq_len` | `None` | List of model lengths; compiler generates one QPC for multiple lengths, vLLM switches based on prompt for higher performance |
| 10 | `comp_ctx_lengths_prefill` | — | List of prefill-stage context lengths for CCL; compiler generates a single binary with multiple program codes, enabling dynamic context length switching for prefill|
| 12 | `comp_ctx_lengths_decode` | — | List of decode-stage context lengths for CCL; compiler generates a single binary with multiple program codes, enabling dynamic context length switching for decode|
| 13 | `ccl_enabled` | `False` | Auto-generate optimized CCL lists for prefill/decode when `comp_ctx_lengths_prefill`/`comp_ctx_lengths_decode` are not provided |
| 14 | `num_patches` | — | Number of patches for VLM compilation |
| 15 | `height` | — | List of Image height for vision+language binary compilation |
| 16 | `width` | — | List of Image width for vision+language binary compilation |
| 20 | `kv_offload` | `False` | Enable KV cache offload |
| 23 | `pooling_device` | — | Device for pooler execution: `"qaic"` or `"cpu"`. Required to get pooled outputs |
| 24 | `pooling_method` | — | Pooling method for `qaic` pooling: `"mean"`, `"avg"`, `"cls"`, `"max"`, or custom |
| 25 | `normalize` | — | Set `True` to normalize pooled outputs (`qaic` pooling only) |
| 26 | `softmax` | — | Set `True` to apply softmax to pooled outputs (`qaic` pooling only) |
| 27 | `prefill_only` | `None` | Disaggregated serving mode: `True` = compile prefill QPC only, `False` = decode QPC only, `None` = single QPC for both |
| 28 | `aic_include_sampler` | `False` | Enable on-device sampling (ODS) — sampling runs on the accelerator instead of the host. Deployment-level (compile-time) toggle only; there is no per-request enable/disable. |
| 29 | `max_top_k_ids` | `512` | Maximum top-k candidate count supported by the compiled ODS binding. Per-request `top_k` values above this limit are rejected. Only meaningful when `aic_include_sampler` is enabled. |
| 30 | `aic_return_pdfs` | `False` | Debug/evaluation-only ODS sub-mode: returns full probability distributions from the device each step for offline accuracy comparison. High transfer cost — never enable for production serving. Only meaningful when `aic_include_sampler` is enabled. |

### `draft_override_qaic_config` Fields

Same fields as `override_qaic_config`, applied to the draft model in speculative decoding. Typically uses fewer cores (e.g., 6) since the draft model is smaller.

!!! info "Compile-time mapping"
    Fields in `override_qaic_config` are passed as keyword arguments to `QEfficient.compile(**dict)`. See the
    [QEfficient compile API](https://quic.github.io/efficient-transformers/) for the full list of accepted parameters.

## Standard vLLM Arguments (QAIC-Relevant)

| Argument | QAIC Notes |
|----------|-----------|
| `--max-num-seqs` | Decode batch size. Directly affects device memory and throughput. |
| `--max-model-len` | Maximum context length (prompt + generated tokens). |
| `--long-prefill-token-threshold` | Sequence length threshold for static shape padding. |
| `--quantization` | Use `mxfp6` for optimal QAIC performance. |
| `--kv-cache-dtype` | Use `mxint8` for KV cache compression. |
| `--gpu-memory-utilization` | Fraction of device memory for KV cache (default: 0.9). **PYT mode only** — AOT mode allocates based on QPC memory requirements. |
| `--tensor-parallel-size` | Number of QIDs for tensor parallelism. |
| `--enforce-eager` | Required for PYT mode (`True`). No effect in AOT. |
| `--async-scheduling` | Set to `False` for PYT mode. AOT supports `True`. |
| `--speculative-config` | JSON for SpD method. See [Speculative Decoding](../features/speculative_decoding.md). |
| `--enable-mm-embeds` | Enable multimodal embedding input (for kv_offload VLM mode). |

## CLI Example

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-num-seqs 4 \
  --max-model-len 2048 \
  --long-prefill-token-threshold 128 \
  --quantization mxfp6 \
  --kv-cache-dtype mxint8 \
  --additional-config '{"override_qaic_config":{"device_group":[0],"num_cores":16}}'
```

!!! info "QEfficient Reference"
    Compilation parameters (batch_size, ctx_len, num_cores, mxfp6_matmul) are set during
    QPC compilation, not at serving time. See the
    [QEfficient Features Guide](https://quic.github.io/efficient-transformers/source/features_enablement.html).
