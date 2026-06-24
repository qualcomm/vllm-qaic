# Architecture

The vLLM QAIC plugin integrates with vLLM's hardware plugin system to run inference on Qualcomm Cloud AI 100 accelerators. Unlike in-tree backends (GPU, CPU), this plugin is distributed as a separate pip package that registers itself at import time — vLLM discovers it automatically when Cloud AI hardware is present.

---

## Plugin Registration

The plugin registers via Python entry points in `pyproject.toml`:

```toml
[project.entry-points."vllm.platform_plugins"]
qaic = "vllm_qaic:QaicPlugin"
```

At startup, vLLM discovers and loads the QAIC platform plugin, which registers:

- The `QaicPlatform` class (device management, feature checks)
- QAIC-specific runtime patches
- Attention backend selection
- Worker and model runner classes

## Component Hierarchy

```
vllm.v1.engine
  +-- QaicWorker
        +-- Device initialization
        +-- Session management
        +-- Distributed comm setup
        +-- QaicModelRunner
              +-- Static-shape input prep
              +-- Prefill/decode scheduling
              +-- SpD integration
              +-- Profile instrumentation
              +-- QPC Model Loader
              |     +-- QPC path resolution
              |     +-- Session configuration
              |     +-- Multi-modal loading
              +-- Attention Backend
                    +-- KV cache management
                    +-- Attention computation
```

## Platform Class

`QaicPlatform` (`vllm/platforms/qaic_base.py`) provides:

| Responsibility | Implementation |
|---------------|----------------|
| Device detection | `QAIC_VISIBLE_DEVICES` parsing |
| Quantization support | mxfp6, mxint8 KV cache |
| Attention backend | Linear SDPA batched (eager mode) |
| Model config validation | Context length, batch size, feature compatibility checks |
| Feature guards | Async output (False), MLA (blocked) |

## Execution Modes

The vLLM QAIC plugin supports two execution modes, enabling flexibility between optimized static compilation and dynamic PyTorch-based inference:

| | AoT (Ahead-of-Time) | PYT (PyTorch Eager / JIT) |
|---|---|---|
| **Model Source** | HuggingFace models | vLLM models (selected models) |
| **Model Transformation** | QEfficient (export + quantize) | vLLM / vLLM-QAIC |
| **Compilation** | QAIC Glow Compiler → QPC binary | Torch.compile via Triton or eager ATen ops |
| **Fused Kernels** | Hardware-native via QPC | vLLM fused kernels (RMSNorm, TopK Softmax, etc.) |
| **Runtime** | AoT runtime | JIT runtime |

### AOT Mode (Primary)

```
HuggingFace Model -> QEfficient (export + compile) -> QPC -> qaic_model_runner (AOT) -> Output Tokens
                     --------------------------------        ---------------------
                       one-time compilation step               QAIC Device Session
```

### PYT Mode (Eager)

```
vLLM Models -> qaic_model_runner (PYT) -> PyTorch execution on torch_qaic backend -> Output Tokens
               --------------------        ----------------------------------------
                 Model execution             Kernel dispatch to QAIC hardware
```

## Device Topology

```
Cloud AI 100 Ultra Card (64 NSP cores total)
+-- QID 0:  16 NSP Cores
+-- QID 1:  16 NSP Cores
+-- QID 2:  16 NSP Cores
+-- QID 3:  16 NSP Cores
```

## File Map

| Component | Location |
|-----------|----------|
| Platform class | `vllm/platforms/qaic_base.py` |
| Worker | `vllm/v1/worker/qaic_worker.py` |
| Model runner | `vllm/v1/worker/qaic_model_runner.py` |
| QPC loader | `vllm/model_executor/model_loader/qaic.py` |
| Attention backends | `vllm/v1/attention/backends/qaic_attn*.py` |
| SpD draft model | `vllm/v1/spec_decode/qaic_draft_model.py` |
| KV connector | `vllm/distributed/kv_transfer/kv_connector/v1/qaic_connector.py` |
| Plugin entry | `vllm-qaic/vllm_qaic/__init__.py` |
| Environment vars | `vllm-qaic/vllm_qaic/envs.py` |
