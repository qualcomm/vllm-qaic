# vllm-qaic Feature → Unit Test Map

This document lists every feature in the vllm-qaic plugin, whether a unit test
already exists (and where), and whether it needs to be created.

> **Last updated:** Aug 2026 — this was originally a planning document written
> before the test files existed ("Action" column tracked what still needed to
> be created/copied). It has since drifted from reality in three ways, all
> corrected below: (1) Section 6 ("Tool Calling / Serving Protocol") planned an
> `unit_tests/online/` folder that was never actually created — no such folder,
> `test_openai_tool_parser.py`, or `test_serving_chat.py` exist anywhere in
> this suite; (2) Section 8's "~333 unit tests" total was a rough estimate at
> planning time and does not match actual collected test counts (parametrized
> files collect far more cases than one line per file suggests); (3) `unit/`
> has since been split into a pure-Python-only
> tree and a hardware-gated tree: every on-device/model-gated test mentioned
> below (e.g. `test_lora_integration.py`, `test_embedding_integration.py`,
> `test_consistency_integration.py`, `test_spd_integration.py`,
> `test_inference_correctness.py`, `test_model_loading.py`,
> `test_sampling_params.py`, and the on-device classes/methods once inside
> `test_chunked_prefill.py`/`test_error_handling.py`/
> `test_on_device_sampling.py`/`test_prefix_caching.py`/`test_samplers.py`/
> `test_spec_decode_unit.py`) has been extracted and moved to
> `vllm-qaic/tests/device_unit_and_e2e/`, rewritten against the
> `device/` fixture system (`qaic_model`, `device_group(s)`, `make_runner`);
> the leftover files that depended on a nonexistent `llm`/`embed_llm` fixture
> have since been deleted from `unit/` entirely (they were dead code, not
> live tests). References to those files/classes below describe the state at
> planning time — see `README.md`'s Quick Reference and Folder Details
> sections for the current, `pytest --collect-only`-verified layout (655
> tests collected in `unit/`; a further 161+ in `device_unit_and_e2e/`, plus
> a handful of files with pre-existing, unrelated collection errors in both
> trees that require native QAIC libs not present in every environment).

**Legend:**
- ✅ EXISTS — test file already exists, can be copied into `unit_tests/`
- 🆕 CREATE — no unit test exists, needs to be written
- ⚠️ PARTIAL — some unit tests exist but coverage is incomplete
- ❌ SKIP — feature requires hardware/model/network; not unit-testable without mocking

---

## 1. Platform & Configuration

| # | Feature | Source Module | Unit-testable logic | Existing test | Action |
|---|---------|--------------|---------------------|---------------|--------|
| 1.1 | QAIC env vars (VLLM_QAIC_*, VLLM_TORCH_QAIC_*) | `envs.py` | Default values, bool/int conversion, `__dir__`, `__getattr__` | None | 🆕 `generic/test_envs.py` |
| 1.2 | `_clean_config()` — override_qaic_config normalisation | `utils/qaic_utils.py` | Key normalisation, device_group parsing, mxfp6/mxint8/dfs/mos/num_cores mapping, ignore-list, bool strings, comp_ctx_lengths, embed_seq_len | None | 🆕 `generic/test_qaic_utils.py` ✅ already created |
| 1.3 | QAIC constants (QAIC_QUANTIZATION_LIST, QAIC_KV_CACHE_DTYPE) | `utils/__init__.py` | List membership, string values | None | 🆕 `generic/test_qaic_utils.py` (add class) |
| 1.4 | `QaicCacheConfig` — mxint8 cache dtype | `patch/patch_config.py` | Accepts "mxint8", "auto", "fp8_*"; monkey-patch applied | None | 🆕 `generic/test_patch_config.py` |
| 1.5 | `QaicDeviceConfig` — device=cpu for qaic | `patch/patch_config.py` | device_type="qaic" → torch.device("cpu"); monkey-patch applied | None | 🆕 `generic/test_patch_config.py` |
| 1.6 | `STR_DTYPE_TO_TORCH_DTYPE["mxint8"]` registration | `patch/patch_utils.py` | Key present in dict, maps to torch.int8 | None | 🆕 `generic/test_patch_utils.py` |
| 1.7 | `QaicPlatform` — AOT/eager detection, device name, dtype support | `platform_base.py` | `is_aot_inference()`, `get_device_name()`, `is_async_output_supported()`, `is_pin_memory_available()`, `check_if_supports_dtype()` | None | 🆕 `generic/test_platform.py` |
| 1.8 | `QaicPlatform.check_and_update_config()` — additional_config JSON/dict parsing | `platform_base.py` | JSON string → dict, Python literal string → dict, invalid string → ValueError | None | 🆕 `generic/test_platform.py` |
| 1.9 | `QaicPlatform._apply_dynamic_resolution_config()` — Qwen2.5VL/Qwen3VL pixel bounds | `platform_base.py` | min_pixels/max_pixels defaults, height/width list validation, mismatched lengths → AssertionError | None | 🆕 `generic/test_platform.py` |
| 1.10 | `QaicQuantConfig` — mxfp6 quantization config | `quantization/quant_config.py` | `get_name()`, `get_supported_act_dtypes()`, `get_config_filenames()`, `from_config()`, `override_quantization_method()` | None | 🆕 `generic/test_quant_config.py` |

---

## 2. Custom Ops / Kernels

| # | Feature | Source Module | Unit-testable logic | Existing test | Action |
|---|---------|--------------|---------------------|---------------|--------|
| 2.1 | RMS Norm reference math | `ops/layernorm.py` | FP32 reference: residual add + RMS norm; identity weight; zero input; output shape | `custom_ops/test_rms_norm_kernel.py` → `TestRmsNormReference` | ✅ COPY to `unit_tests/custom_ops/` |
| 2.2 | RMS Norm kernel numerics (CPU simulation) | `ops/layernorm.py` | 2D/ND inputs, weight scaling, large values, epsilon effect, dtype dispatch | `custom_ops/test_rms_norm_kernel.py` → `TestRmsNormKernelNumerics`, `TestDtypeDispatch` | ✅ COPY to `unit_tests/custom_ops/` |
| 2.3 | RMS Norm alignment constraint (N % 64 == 0) | `ops/layernorm.py` | Valid N values pass, invalid N values documented | `custom_ops/test_rms_norm_kernel.py` → `TestRmsNormAlignmentConstraint` | ✅ COPY to `unit_tests/custom_ops/` |
| 2.4 | RMS Norm multi-core striping | `ops/layernorm.py` | Row independence, ceil(M/numCores) coverage | `custom_ops/test_rms_norm_kernel.py` → `TestMultiCoreStriping` | ✅ COPY to `unit_tests/custom_ops/` |
| 2.5 | RMS Norm Hexagon kernel (hardware) | `ops/layernorm.py` | Actual kernel call on QAIC device | `custom_ops/test_rms_norm_kernel.py` → `TestHexagonKernelHW` | ✅ COPY (hardware tests skip on CPU) |
| 2.6 | Grouped TopK router — valid expert grouping check | `ops/grouped_topk_router.py` | `_has_valid_expert_grouping()`: valid/invalid combinations | `custom_ops/test_ops_topk_router.py` | ✅ now covered pure-Python (superseded `custom_ops/test_grouped_topk_qaic.py`, a hardware-only duplicate moved to `tests/_deprecated/`) |
| 2.7 | Grouped TopK router — QAIC support check | `ops/grouped_topk_router.py` | `_supports_qaic_grouped_topk()`: device type, dtype, shape constraints | `custom_ops/test_ops_topk_router.py` | ✅ now covered pure-Python (superseded `custom_ops/test_grouped_topk_qaic.py`, a hardware-only duplicate moved to `tests/_deprecated/`) |
| 2.8 | Regular TopK router — QAIC support check | `ops/topk_router.py` | `_supports_qaic_regular_topk()`: device type, dtype, topk limit | None | 🆕 `custom_ops/test_ops_topk_router.py` |

---

## 3. Speculative Decoding (SpD / PLD / DLM)

| # | Feature | Source Module | Unit-testable logic | Existing test | Action |
|---|---------|--------------|---------------------|---------------|--------|
| 3.1 | SpD metadata calculation (`_calc_spec_decode_metadata`) | `worker/model_runner.py` | draft_token_ids extraction, target_logits_indices, bonus_logits_indices; uniform/mixed/zero-draft/max-padding cases | `spec_decode/unit/test_calc_spec_decode_metadata.py` | ✅ COPY to `unit_tests/spd/` |
| 3.2 | SpD max_num_batched_tokens calculation | `worker/model_runner.py` | Budget formula with spec tokens | `spec_decode/unit/test_max_num_batched_tokens.py` | ✅ COPY to `unit_tests/spd/` |
| 3.3 | SpD variable-K decode specializations | `worker/model_runner.py` | Variable decode token counts | `spec_decode/unit/test_variable_decode_specializations.py` | ✅ COPY to `unit_tests/spd/` |
| 3.4 | Rejection sampler greedy-path optimizations (skip-clone, skip-softmax) | `patch/patch_rejection_sampler.py` | `_qaic_forward()` with all-greedy metadata: no clone, no softmax | None | 🆕 `samplers/test_rejection_sampler.py` |
| 3.5 | Draft model config extraction (draft_override_qaic_config fallback) | `spec_decode/qaic_draft_model.py` | Config priority: draft_override > override; async_scheduling assertion | None | 🆕 `spd/test_spec_decode_unit.py` |

---

## 4. LoRA

| # | Feature | Source Module | Unit-testable logic | Existing test | Action |
|---|---------|--------------|---------------------|---------------|--------|
| 4.1 | `search_adapters_in_cache()` — scan HF_HOME for adapters | `model_loader/qaic.py` | Empty cache → []; matching adapter found; different base model not returned | `lora/test_qaic_lora.py::test_qaic_search_adapters_in_cache` (needs HF download) | ⚠️ 🆕 `lora/test_lora_unit.py` (pure version, no download) ✅ already created |
| 4.2 | `verify_adaptername_to_id_consistency()` — JSON mapping validation | `model_loader/qaic.py` | Matching → True; extra key → False; missing key → False; empty → True | `lora/test_qaic_lora.py::test_qaic_get_qaic_model_dump_adaptername_to_id` (needs model load) | ⚠️ 🆕 `lora/test_lora_unit.py` (pure version) ✅ already created |
| 4.3 | adaptername_to_id JSON round-trip | `lora/test_qaic_lora.py` | Write/read preserves content; wrong path → FileNotFoundError; inconsistent → ValueError | `lora/test_qaic_lora.py` (inline) | ⚠️ 🆕 `lora/test_lora_unit.py` ✅ already created |

---

## 5. Disaggregated Serving

| # | Feature | Source Module | Unit-testable logic | Existing test | Action |
|---|---------|--------------|---------------------|---------------|--------|
| 5.1 | `RoundRobinSchedulingPolicy` | `qaic_disagg/proxy/server.py` | Basic round-robin cycling | `disaggregated_serving/test_policy.py` | ✅ COPY to `unit_tests/disaggregated/` |
| 5.2 | `LeastOutstandingSchedulingPolicy` | `qaic_disagg/proxy/server.py` | Initial distribution, outstanding count tracking, thread safety | `disaggregated_serving/test_policy.py` | ✅ COPY to `unit_tests/disaggregated/` |
| 5.3 | Prefill streaming latency stats helpers | `test_prefill_streaming.py` | `_compute_stats()` mean/median/p99; `_throughput()` tok/s formula | None | 🆕 `disaggregated/test_prefill_streaming_utils.py` ✅ created |

---

## 6. Tool Calling / Serving Protocol

> **Not implemented.** No QAIC-specific logic was identified here (tool-call
> parsing and chat-serving protocol are generic vLLM behaviour, not
> QAIC-specific), so no `unit_tests/online/` folder or test files were
> ultimately created. This section is kept for historical planning context
> only — do not look for `online/test_openai_tool_parser.py` or
> `online/test_serving_chat.py`; they do not exist.

| # | Feature | Source Module | Unit-testable logic | Existing test | Action |
|---|---------|--------------|---------------------|---------------|--------|
| 6.1 | OpenAI tool-call parser | `vllm.entrypoints.openai.tool_parsers` | Parse tool call JSON, function call extraction | `gptoss/test_openai_tool_parser.py` | ❌ NOT CREATED — generic vLLM, no QAIC-specific logic found |
| 6.2 | Serving chat protocol | `vllm.entrypoints.openai` | Chat message formatting, system/user/assistant roles | `gptoss/test_serving_chat.py` | ❌ NOT CREATED — generic vLLM, no QAIC-specific logic found |

---

## 7. Features with integration/E2E tests only — extractable unit tests

Even though the full E2E tests require hardware, each feature has **pure Python
helper logic** that can and should be unit-tested without any hardware.

| # | Feature | CI Script | Pure Python logic to unit-test | New test file |
|---|---------|-----------|-------------------------------|---------------|
| 7.1 | Output consistency | `run_consistency_test_qaic.sh` | QAIC-specific: `_clean_config()` normalisation already tested; no additional QAIC-specific logic | ❌ SKIP (generic vLLM) |
| 7.2 | Samplers | `run_samplers_test_qaic.sh` | QAIC-specific: `aic_include_sampler` config parsing already in `generic/test_qaic_utils.py`; ODS+SPD exclusion in `generic/test_platform.py` | ❌ SKIP (generic vLLM) |
| 7.3 | Online serving | `run_online_test_qaic.sh` | No QAIC-specific logic (generic vLLM API) | ❌ SKIP (generic vLLM) |
| 7.4 | Accuracy | `run_accuracy_test_qaic.sh` | No QAIC-specific logic (generic metric computation) | ❌ SKIP (generic) |
| 7.5 | LoRA consistency/loading | `run_lora_test_qaic.sh` | Already covered in `lora/test_lora_unit.py` | ✅ done |
| 7.6 | SpD E2E | `run_spd_test_qaic.sh` | Already covered in `spd/` | ✅ done |
| 7.7 | Disaggregated serving | `run_disaggregated_test_qaic.sh` | Already covered in `disaggregated/test_policy.py` + `test_prefill_streaming_utils.py` | ✅ done |
| 7.8 | Embedding models | `run_embed_test_qaic.sh` | No QAIC-specific logic (cosine similarity is generic) | ❌ SKIP (generic) |
| 7.9 | Multimodal (VLM, image input) | `run_multimodal_test_qaic.sh` | **QAIC-specific**: `is_internvl()`, `is_qwenvl()`, `is_gemma()`, `is_granite()`, `is_llama4()` model name checks; `update_qaic_config()` QAIC config merging for VLM | 🆕 `multimodal/test_multimodal_utils.py` |
| 7.10 | Prefix caching | `run_prefixcaching_test_qaic.sh` | **QAIC-specific**: prefix caching forcibly disabled in AOT mode (`platform_base.py`); `mamba_block_size` reset | 🆕 `prefixcaching/test_prefix_caching.py` |
| 7.11 | On-device sampling | `run_on_device_sampling_test_qaic.sh` | **QAIC-specific**: `aic_include_sampler` bool parsing in `_clean_config()`; ODS+SPD mutual exclusion assertion | 🆕 `on_device_sampling/test_on_device_sampling.py` |
| 7.12 | Chunked prefill | `ci_fasttest_qaic.sh` | **QAIC-specific**: `prefill_seq_len` calculation from `long_prefill_token_threshold`; chunked prefill disabled for pooling models | 🆕 `generic/test_chunked_prefill.py` |
| 7.13 | Async scheduling | `ci_fasttest_qaic.sh` | **QAIC-specific**: `async_scheduling=False` enforced in eager mode; warning logged | 🆕 `generic/test_chunked_prefill.py` (TestAsyncScheduling class) |
| 7.14 | Offline benchmarks | `run_benchmark_test_qaic.sh` | No QAIC-specific logic | ❌ SKIP |

---

## 8. Summary: Unit Tests Created

| Priority | File | Feature(s) covered | ~Tests |
|----------|------|---------------------|--------|
| P0 ✅ | `generic/test_qaic_utils.py` | `_clean_config()` normalisation + samplers/output consistency on device | 70 |
| P0 ✅ | `generic/test_envs.py` | VLLM_QAIC_* env vars | 25 |
| P0 ✅ | `lora/test_lora_unit.py` | search_adapters_in_cache, verify_adaptername_to_id_consistency, JSON round-trip | 20 |
| P1 ✅ | `generic/test_patch_config.py` | QaicCacheConfig (mxint8), QaicDeviceConfig (cpu), monkey-patch | 10 |
| P1 ✅ | `generic/test_quant_config.py` | QaicQuantConfig, QAIC_QUANTIZATION_LIST | 10 |
| P1 ✅ | `generic/test_platform.py` | QaicPlatform CPU-only methods, additional_config parsing, dynamic resolution, disagg constraints | 36 |
| P1 ✅ | `generic/test_patch_utils.py` | mxint8 dtype registration | 5 |
| P2 ✅ | `samplers/test_rejection_sampler.py` | Greedy-path skip-clone/skip-softmax optimizations | 8 |
| P2 ✅ | `custom_ops/test_ops_topk_router.py` | `_supports_qaic_regular_topk()`, `_has_valid_expert_grouping()` | 12 |
| P2 ✅ | `spd/test_spec_decode_unit.py` | Draft model config extraction, async_scheduling assertion, ngram SpD on device | 15 |
| P2 ✅ | `generic/test_chunked_prefill.py` | prefill_seq_len calculation, chunked prefill + async scheduling on device | 20 |
| P2 ✅ | `embedding/test_embedding_utils.py` | _check_vector, model lists, PoolerConfig, embedding on device | 50 |
| P2 ✅ | `multimodal/test_multimodal_utils.py` | VLM model detection, update_qaic_config | 35 |
| P2 ✅ | `disaggregated/test_prefill_streaming_utils.py` | _compute_stats, _throughput | 17 |

**Totals above are rough per-file estimates from planning time and are not
reliable — several files (e.g. `test_ops_topk_router.py`, `test_rms_norm_kernel.py`,
`test_qaic_utils.py`) are parametrized and collect many more cases than a
single "~Tests" number suggests. Several rows also predate the `unit/` /
`device_unit_and_e2e/` split (see the note at the top of this file) —
the on-device portions of `test_spec_decode_unit.py`,
`test_chunked_prefill.py`, `test_embedding_utils.py`, and
`test_multimodal_utils.py`'s planned coverage now live under
`vllm-qaic/tests/device_unit_and_e2e/`. For accurate,
`pytest --collect-only`-verified counts per file and per folder, see
`README.md`'s Quick Reference and Folder Details sections (904 tests
collected in `unit/`, plus 79 in `device_unit_and_e2e/`).**
