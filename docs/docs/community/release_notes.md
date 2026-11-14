# Release Notes

## v1.23.4 (vLLM 0.23.0)

Maintenance and feature release on top of v1.23.0 (33 changes). Targets Apps SDK
1.23 and QEfficient `release/v1.23.4`. It includes everything in v0.23.0 below.

### New Features

- Encoder Cache (EC) support, with documentation and updated examples (#69, #229, #236)
- Speculative decoding with disaggregated serving (#75)
- SpD rejection sampler now runs on Numba; `triton-cpu` is optional, and SPD
  dependencies are available as an optional extra (#171, #172, #238)
- KV cache transfer for rank 4 conv and recurrent states in GDN models (#120)
- Support for versioned Hexagon tools SDK layouts (#156, #196)
- Python 3.13 support; Python 3.10 dropped (#168, #197)
- Setup guide for installing vllm-qaic from Apps SDK wheels (#166)

### Bug Fixes

- Fix gptoss models generating junk or null output (#204)
- Prevent padded decode rows from aliasing active batch slots (#206)
- Fix async disaggregated prefill CCL slot cleanup (#223)
- Fix prefill to wait for the oldest exec object when none are free (#231)
- Patch `VllmConfig.max_concurrent_batches` for pipeline prefill (#218)
- Key disaggregated KV hand-off by request id to prevent collisions (#165)
- Stop defaulting `replicate_kv_heads` for disaggregated compile (#241)
- Fix QEfficient compile config forwarding (#233)
- Fix CCL prefill multimodal input shapes (#118)
- Skip chunked-prefill assert for vision encoder (#214)
- Pin encoder cache/compute budget to `max_num_batched_tokens` for vision encoder (#178)
- Increase `dataPathTimeoutMs` (#200)
- Remove `long_prefill_token_threshold` (#74)

### PyTorch (JIT) Mode

- Update torch to 2.13 (#132)
- Hard-code `enable_prefix_caching` to False (#131)
- Fix `QAIC_VISIBLE_DEVICES` usage in examples (#158)

### Build and Dependencies

- Update efficient-transformers to 1.23.4 (#226)
- Pin `mistral_common` to v1.11.3 (#228)
- SDK version set to 1.23 (#135); fix installer/Dockerfile default dependency
  versions (#142); use hyphenated `vllm-qaic` in default SDK path (#177)
- Fix pytest fixture registration for disaggregated tests (#227)

### Version Compatibility

| vllm-qaic | vLLM | Apps SDK | QEfficient | torch (AOT) | torch (PYT) |
|-----------|------|----------|------------|-------------|-------------|
| v1.23.4 | v0.23.0 | 1.23 | release/v1.23.4 | 2.7.0+cpu | 2.13.0+cpu |

Changes since v1.23.0:

- Apps SDK: 1.22 -> 1.23
- QEfficient: `main` -> `release/v1.23.4`
- torch (PYT): 2.11.0+cpu -> 2.13.0+cpu; torchvision (PYT): 0.26.0+cpu -> 0.28.0+cpu
- `mistral_common`: `>= 1.11.3` -> `== 1.11.3`
- Python: 3.10-3.12 -> 3.11-3.13
- `arctic-inference` is no longer in the default AOT requirements; install it
  with `pip install vllm-qaic[spd]` (`arctic-inference==0.1.1`)
- Default SDK install path changed from `/opt/qti-aic/integrations/vllm_qaic`
  to `/opt/qti-aic/integrations/vllm-qaic`
- Unchanged: vLLM v0.23.0, torch (AOT) 2.7.0+cpu, `transformers` 5.5.4

### Pull Requests

- Fix CCL prefill multimodal input shapes by @quic-vargupt in https://github.com/qualcomm/vllm-qaic/pull/118
- [PyT] update torch version to 2.13 by @quic-agokhale in https://github.com/qualcomm/vllm-qaic/pull/132
- Change SDK version to 1.23 by @quic-sanising in https://github.com/qualcomm/vllm-qaic/pull/135
- Fix installer/dockerfile default dependency versions by @quic-agokhale in https://github.com/qualcomm/vllm-qaic/pull/142
- [PyT] hard-code enable_prefix_caching to False by @quic-agokhale in https://github.com/qualcomm/vllm-qaic/pull/131
- docs: add setup guide for installing vllm-qaic from Apps SDK wheels by @quic-agokhale in https://github.com/qualcomm/vllm-qaic/pull/166
- [PyT] Fix QAIC_VISIBLE_DEVICES usage in examples and remove no-op code by @quic-agokhale in https://github.com/qualcomm/vllm-qaic/pull/158
- scripts: install triton-cpu build deps in install_triton_cpu.sh by @eplatero97 in https://github.com/qualcomm/vllm-qaic/pull/172
- [AOT][Spec Decode] Run the SpD rejection sampler on Numba; triton-cpu becomes optional by @eplatero97 in https://github.com/qualcomm/vllm-qaic/pull/171
- Key disagg KV hand-off by request id to prevent collisions by @quic-sanising in https://github.com/qualcomm/vllm-qaic/pull/165
- Use hyphenated vllm-qaic in default SDK path by @quic-agokhale in https://github.com/qualcomm/vllm-qaic/pull/177
- Pin encoder cache/compute budget to max_num_batched_tokens for vision encoder by @quic-sanising in https://github.com/qualcomm/vllm-qaic/pull/178
- Drop Python 3.10, add Python 3.13 support by @quic-sanising in https://github.com/qualcomm/vllm-qaic/pull/197
- Support versioned Hexagon tools SDK layouts by @quic-sanising in https://github.com/qualcomm/vllm-qaic/pull/196
- Increase dataPathTimeoutMs(branch: main) by @quic-mohmeh in https://github.com/qualcomm/vllm-qaic/pull/200
- Remove long_prefill_token_threshold by @quic-sanising in https://github.com/qualcomm/vllm-qaic/pull/74
- Skip chunked-prefill assert for vision encoder by @quic-sanising in https://github.com/qualcomm/vllm-qaic/pull/214
- Add support for Encoder Cache (EC) by @quic-sanising in https://github.com/qualcomm/vllm-qaic/pull/69
- Fix for gptoss models generating junk or null output by @quic-vargupt in https://github.com/qualcomm/vllm-qaic/pull/204
- [Spec Decode] Enable speculative decoding with disaggregated serving by @eplatero97 in https://github.com/qualcomm/vllm-qaic/pull/75
- Fix async disaggregated prefill CCL slot cleanup by @vjanfaza in https://github.com/qualcomm/vllm-qaic/pull/223
- fix: prevent padded decode rows from aliasing active batch slots by @bharathmedak in https://github.com/qualcomm/vllm-qaic/pull/206
- Patch VllmConfig max_concurrent_batches for pipeline prefill by @chaitanyasachdeva in https://github.com/qualcomm/vllm-qaic/pull/218
- Update efficient-transformers to 1.23.4 by @quic-sanising in https://github.com/qualcomm/vllm-qaic/pull/226
- Fix mistral_common dependency to v1.11.3 by @quic-sanising in https://github.com/qualcomm/vllm-qaic/pull/228
- Fix pytest fixture registration for disaggregated tests by @eplatero97 in https://github.com/qualcomm/vllm-qaic/pull/227
- Adding code changes for handling rank 4 conv & recurrent states for model with GDN during KV$ transfer by @quic-vargupt in https://github.com/qualcomm/vllm-qaic/pull/120
- Add documentation for Encoder Cache by @quic-sanising in https://github.com/qualcomm/vllm-qaic/pull/229
- Fix QEfficient compile config forwarding by @eplatero97 in https://github.com/qualcomm/vllm-qaic/pull/233
- Update examples for Encoder Cache by @quic-sanising in https://github.com/qualcomm/vllm-qaic/pull/236
- Fix prefill to wait for the oldest exec object when none are free by @chaitanyasachdeva in https://github.com/qualcomm/vllm-qaic/pull/231
- [bug] Stop defaulting replicate_kv_heads for disaggregated compile by @eplatero97 in https://github.com/qualcomm/vllm-qaic/pull/241
- [spec decode] Add optional SPD dependency extra by @eplatero97 in https://github.com/qualcomm/vllm-qaic/pull/238

**Full Changelog**: https://github.com/qualcomm/vllm-qaic/compare/v1.23.0...release/r1.23.4

### Contributors

Erick Platero, Sanidhya Singal, Apoorva Gokhale, Chaitanya Sachdeva,
quic-vargupt, BharathTeja, Vahid Janfaza, quic-mohmeh

## v0.23.0 (vLLM 0.23.0)

Upgrades the vLLM-QAIC plugin to track vLLM v0.23.0, with expanded model
support, a new speculative decoding proposer, scheduler correctness fixes,
and simplified packaging for AOT/PyTorch builds.

### AOT Execution Mode

**Model Support**

- Fixed Gemma3 multimodal processing for vLLM 0.23
- Fixed `max_pixels` resolution handling for Qwen-VL
- Fixed `batch_size` handling for Qwen3-235B
- Simplified the eager/AOT vision-language example to use dict-based
  config instead of engine args

**Speculative Decoding**

- Added DFlash: speculative decoding with a Diffusion LM proposer

**Scheduling & Execution**

- Fixed pending exec-object tracking for async prefill scheduling
- Fixed decode cleanup callback

### PyTorch (JIT) Execution Mode

- Added Rotary Embedding support
- Added MoE support for vLLM 0.23.0
- Aligned `QaicWorkerPyt` profiler with upstream profiler configuration
  changes

### Build & Packaging

- Upgraded base vLLM dependency to v0.23.0
- Made RDMA support optional when building UCX
- Added Rust frontend build support to Docker AOT/PyTorch images, with a
  configurable `--rust-image` flag and output wheel filename argument
- Changed the default triton-cpu install path to a local package
  directory, and made triton-cpu build opt-in (`TRITON_CPU=1`) for AOT
  wheels
- Fixed the bf16 kernel build for AI200

### CI & Developer Tooling

- Added a pre-commit validation workflow for PRs
- Ran pre-commit across the full codebase

### Pull Requests

- [v0.15.0] Update vllm dependency installation by @quic-agokhale in https://github.com/qualcomm/vllm-qaic/pull/49
- MoE support v0.23.0 by @Shrinivas-kulkarni9 in https://github.com/qualcomm/vllm-qaic/pull/54
- V0.23.0 by @ankitqti in https://github.com/qualcomm/vllm-qaic/pull/78
- Fix decode cleanup callback by @chaitanyasachdeva in https://github.com/qualcomm/vllm-qaic/pull/81
- Add pre-commit validation workflow for PRs by @quic-sanising in https://github.com/qualcomm/vllm-qaic/pull/79
- fixed build for bf16 kernel for AI200 by @sahil139 in https://github.com/qualcomm/vllm-qaic/pull/87
- Align QaicWorkerPyt profiler with upstream changes to profiler configuration by @quic-agokhale in https://github.com/qualcomm/vllm-qaic/pull/86
- Change default triton-cpu installation path to local directory in package folder by @quic-agokhale in https://github.com/qualcomm/vllm-qaic/pull/100
- Add Rust frontend build support to Docker AOT/PYT images by @ankitqti in https://github.com/qualcomm/vllm-qaic/pull/103
- Add Rotary Embedding support by @quic-agokhale in https://github.com/qualcomm/vllm-qaic/pull/84
- build: make RDMA support optional when building UCX by @quic-vargupt in https://github.com/qualcomm/vllm-qaic/pull/107
- Fix whlbuild triton-cpu default behavior by @quic-agokhale in https://github.com/qualcomm/vllm-qaic/pull/119
- Run pre-commit on all files by @quic-sanising in https://github.com/qualcomm/vllm-qaic/pull/93
- [v0.23.0] Fix pending exec-object tracking for async prefill scheduling by @animku in https://github.com/qualcomm/vllm-qaic/pull/114
- vllm-qaic: fix batch_size for Qwen3-235B by @chaitanyasachdeva in https://github.com/qualcomm/vllm-qaic/pull/117
- [DFlash] Add speculative decoding with Diffusion LM proposer by @vjanfaza in https://github.com/qualcomm/vllm-qaic/pull/71
- Fix Gemma3 multimodal processing for vLLM 0.23 by @quic-vargupt in https://github.com/qualcomm/vllm-qaic/pull/97
- Remove engine args to enable dict-based support for eager/aot mode vision language example by @Neeharika-Gupta in https://github.com/qualcomm/vllm-qaic/pull/111
- Fix max_pixels resolution handling for Qwen-VL by @chaitanyasachdeva in https://github.com/qualcomm/vllm-qaic/pull/125

**Full Changelog**: https://github.com/qualcomm/vllm-qaic/compare/release/v0.15.0...release/v0.23.0

## v0.15.0.dev3 (Initial Public Release)

Initial vLLM-QAIC plugin for Qualcomm Cloud AI Accelerators. Introduces V1
architecture support with both Ahead-of-Time (AOT) compiled and PyTorch JIT
execution modes.

### AOT Execution Mode

- Generic model execution (LLM, MoE, Embedding, Reranker, Score)
- QPC execution flow and QEFF model compilation
- CCL support for pipeline prefill
- Speculative decoding (Ngram, Suffix, Draft)
- LoRaX adapter serving (Dynamic & Fixed)
- Disaggregated Serving:
    - Pipeline prefill execution with Tensor Splitting
    - MDP Partitioner support
    - QAIC shared memory connector
    - LMCache connector
    - Disaggregated orchestration package integration
    - Grafana/Prometheus monitoring dashboard
    - LLM-D support
    - Auto head replication
    - KV cache transfer for heterogeneous attention layers
- Vision-Language Models:
    - 2QPC and 3QPC VLM execution
    - Qwen3VL, Qwen3VLMoE, InternVL, Qwen2.5VL
- Scheduling & Performance:
    - Scheduler optimization with async scheduling
    - Chunked prefill scheduling (reduces P99 TPOT)

### PyTorch (JIT) Execution Mode

- PyTorch eager mode execution
- Fused Kernels (RMSNorm, TopK Softmax, Grouped TopK, TopK Sigmoid)

### Version Compatibility

| vllm-qaic | vLLM | Apps SDK | QEfficient | torch (AOT) | torch (PYT) |
|-----------|------|----------|------------|-------------|-------------|
| v0.15.0.dev0 | v0.15.0 | >= 1.22.0 | main | 2.7.0+cpu | 2.10.0+cpu |

### Known Issues

- Prefix caching is not yet supported (planned)
- PYT mode does not support SpD, LoRA, or disaggregated serving
- `best_of > 1` is not supported

### Contributors

Ankit Arora, Sanidhya Singal, Xiyue Shi, Shivani Athavale, Erick Platero,
Mahesh Balasubramanian, Animesh Kumar, Apoorva Gokhale,
Gokkulnath Thirukonda Srinivasan, Jou-An Chen, Varun Gupta,
Abdul Rasheed Sahni, Mohit Mehta, Neeharika Gupta, Mrunal Kshirsagar,
Aditya Dhananjay Jadhav, Mohit Gupta, Sahil Deshmukh, Vahid Janfaza,
Chaitanya Sachdeva, Shriniwas Kulkarni
