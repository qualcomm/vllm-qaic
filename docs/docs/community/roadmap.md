# Roadmap

Upcoming features and improvements for the vLLM QAIC plugin.

*Last updated: June 2026*

## AOT Mode

| Feature | Status | Description |
|---------|--------|-------------|
| LMCache integration | :white_check_mark: Ongoing | KV cache sharing across fleet via LMCache |
| llm-d orchestration | :white_check_mark: Ongoing | Kubernetes-native distributed inference |
| Additional model architectures | :white_check_mark: Ongoing | Expanding validated model list |

## PYT (Eager / JIT) Mode

| Feature | Status | Description |
|---------|--------|-------------|
| Fused HexNN kernels | :white_check_mark: Ongoing | PagedAttention, MLA, GatedLinearAttention |
| `torch.compile` support | :clipboard: Planned | Wider model coverage via compiled graphs |
| Triton kernel support | :clipboard: Planned | Execution of Triton Kernels |
| Text-only LLM support | :white_check_mark: Ongoing | Extending beyond VLMs |
| Speculative decoding | :clipboard: Planned | SpD support in eager mode |

## Status Legend

| Symbol | Meaning |
|--------|---------|
| :white_check_mark: Ongoing | Active development |
| :clipboard: Planned | On the roadmap, not yet started |
| :test_tube: Experimental | Available but not production-ready |
