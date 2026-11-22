# Developer Guide

Technical documentation for contributors and advanced users working on the vLLM QAIC plugin.

`vllm-qaic` is a pip-installable plugin that integrates with vLLM through Python entry points. No modification to vLLM source code is required — the plugin registers its platform, worker, and model runner implementations at import time, and vLLM dispatches to them when Cloud AI hardware is detected.

## Sections

| Section | Description |
|---------|-------------|
| [Architecture](architecture.md) | Plugin registration, component hierarchy, execution modes, device topology |
| [Profiling](profiling.md) | Device-level tracing via `VLLM_TORCH_QAIC_PROFILER_DIR`; host profiling via `--profiler-config` |
| [Contributing](contributing.md) | Contribution guidelines, licensing, and process |
