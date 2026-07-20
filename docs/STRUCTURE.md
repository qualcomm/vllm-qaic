# qaic_docs — Documentation Structure

**vLLM QAIC Plugin Documentation** | MkDocs Material | 35 pages

Built for eventual deployment at `https://quic.github.io/vllm-qaic/` and later merge into vLLM upstream `docs/`.

---

## Directory Tree

```
qaic_docs/
├── mkdocs.yml                              # MkDocs Material configuration
├── requirements-docs.txt                   # Doc build dependencies
└── docs/
    ├── index.md                            # Landing page (value prop, Docker quickstart)
    │
    ├── getting_started/
    │   ├── index.md                        # Mode selection guide (AOT vs PYT)
    │   ├── quickstart.md                   # Docker-first + source-based hello world
    │   ├── installation/
    │   │   ├── index.md                    # Installation method overview
    │   │   ├── prerequisites.md            # Hardware, SDK, Python 3.12 setup
    │   │   ├── aot_mode.md                 # AOT scripted + manual install
    │   │   ├── pyt_mode.md                 # PYT scripted + manual install
    │   │   └── verification.md             # Post-install smoke tests
    │   └── faq.md                          # Common issues and fixes
    │
    ├── user_guide/
    │   ├── index.md                        # Redirect to top-level sections
    │   ├── configuration/
    │   │   ├── environment_variables.md    # VLLM_QAIC_* env vars reference
    │   │   ├── engine_args.md              # additional_config, CLI flags
    │   │   └── device_management.md        # QID topology, multi-card, core allocation
    │   ├── features/
    │   │   ├── index.md                    # Feature support matrix (AOT vs Eager)
    │   │   ├── speculative_decoding.md     # N-gram, suffix, draft model SpD
    │   │   ├── quantization.md             # mxfp6, mxint8 KV cache (hardware-native)
    │   │   ├── multimodal.md               # VLM kv_offload architecture
    │   │   ├── disaggregated_serving.md    # xPyD prefill/decode split
    │   │   ├── lora.md                     # LoRA adapter serving
    │   │   └── encoder_decoder.md          # Whisper / encoder-decoder
    │   ├── models/
    │   │   ├── supported_models_aot.md     # Full QEfficient vLLM model matrix
    │   │   └── supported_models_eager.md   # Eager validated models table
    │   ├── serving/
    │   │   ├── offline_inference.md        # vllm.LLM batch inference
    │   │   └── online_serving.md           # vllm serve + OpenAI API
    │   └── performance.md                  # Performance tuning guidelines
    │
    ├── developer_guide/
    │   ├── index.md                        # Developer overview
    │   ├── architecture.md                 # Plugin system, platform class, data flow
    │   ├── profiling.md                    # Profiling instrumentation + analysis
    │   ├── testing.md                      # CI scripts, running tests locally
    │   └── contributing.md                 # License, scope, PR process (from OSR-22956)
    │
    ├── community/
    │   ├── release_notes.md                # Version history + compatibility matrix
    │   └── roadmap.md                      # Upcoming features per mode
    │
    └── profiling/
        └── index.md                        # Top-level profiling quick reference (legacy)
```

---

## Navigation Structure

The site uses **7 top-level sections** (visible as tabs):

| Nav Tab | Contents | Purpose |
|---------|----------|---------|
| **Home** | Landing page | Value prop, Docker quickstart, hardware overview |
| **Getting Started** | Quick Start, Installation (5 pages), FAQ | Onboarding funnel for new users |
| **Models & Features** | Feature matrix, supported models (AOT + Eager), 6 feature guides | "What can I run and what does it support?" |
| **Deployment** | Online serving, offline inference, performance tuning | Production operations |
| **Configuration** | Engine args, env vars, device management | Reference material for tuning |
| **Developer Guide** | Architecture, profiling, testing, contributing | Internal contributors |
| **Release Notes** / **Roadmap** | Version history, upcoming features | Flat top-level entries |

---

## Design Principles

1. **Hybrid Ascend + TPU structure**: Ascend's section hierarchy for "Getting Started", TPU's combined "Models & Features" approach, and a flat top-level for reference sections

2. **QEfficient cross-referencing**: Compilation, model export, and supported model lists link to QEfficient docs rather than duplicating — clearly marked with `!!! info "QEfficient Reference"` admonition blocks

3. **Two-mode documentation**: Every applicable page uses Material for MkDocs content tabs (`=== "AOT Mode"` / `=== "PYT Mode"`) to cleanly separate mode-specific content

4. **Docker-first quickstart**: Pre-built containers from GHCR (`ghcr.io/quic/cloud_ai_inference_vllm`) for fastest path to first inference

5. **Standalone → mergeable**: Lives at `qaic_docs/` now; when merging into vLLM upstream:
   - Installation pages → `docs/getting_started/installation/qaic.md`
   - Model tables → `docs/models/hardware_supported_models/qaic.md`
   - Features → sections within existing feature pages
   - Fragment markers (`<!-- --8<-- -->`) in prerequisites.md ready for snippet inclusion

6. **Visual hierarchy**: Material grid cards for landing/quickstart navigation, Mermaid diagrams for architecture/flows, collapsible `???` admonitions for progressive disclosure

7. **Professional UX**: Dark/light toggle, search suggest, navigation tabs, footer links, content tooltips, code annotations

---

## Key Cross-References to QEfficient

| Topic | Our Page | QEfficient Link |
|-------|----------|-----------------|
| Model compilation | `engine_args.md` | `source/quick_start.html` |
| Supported models | `supported_models_aot.md` | `source/validate.html` |
| SpD compilation | `speculative_decoding.md` | `speculative_decoding.html` |
| Feature enablement | `engine_args.md` | `source/features_enablement.html` |
| SDK installation | `prerequisites.md` | Cloud AI SDK Getting Started |

---

## Building the Docs

```bash
cd qaic_docs
pip install -r requirements-docs.txt
mkdocs serve    # Local preview at http://127.0.0.1:8000
mkdocs build    # Static site in site/
```

---

## Content Sources

| Page | Source Material |
|------|----------------|
| `index.md` | `vllm-qaic/README.md` + `Open_Source_vLLM_Blog.docx` |
| `installation/*` | `vllm-qaic/docs/installation.md` (split) |
| `quickstart.md` | Blog Docker examples + README snippets |
| `environment_variables.md` | `vllm-qaic/vllm_qaic/envs.py` |
| `features/index.md` | `vllm/platforms/qaic_base.py` support checks |
| `speculative_decoding.md` | Blog + `examples/offline_inference/qaic_spd.py` |
| `disaggregated_serving.md` | Blog diagrams + Docker commands |
| `multimodal.md` | `examples/offline_inference/qaic_vision_language.py` |
| `profiling.md` | `docs/qaic/QAIC_PROFILING.md` |
| `contributing.md` | `OSR-22956-Contribution Approval (QC GitHub).docx` |
| `performance.md` | New — performance tuning guidance and SpD expectations |
| `supported_models_aot.md` | Full extraction from QEfficient `validate.html` (vLLM ✔️ rows) |

---

## v0.15.0 Documentation Decisions

Decisions applied during documentation review for this release:

| Decision | Rationale |
|----------|-----------|
| **On-device sampling removed** | Not supported as of v0.15.0; feature page deleted, all references removed. Supported in v0.23.0. |
| **SpD + LoRA removed from limitations** | Upstream vLLM also does not support this (GitHub issue #6137). Not QAIC-specific. |
| **SpD + Multimodal removed from limitations** | Upstream vLLM raises `NotImplementedError` for multimodal + spec decode (`v1/spec_decode/draft_model.py`). Not QAIC-specific. |
| **Beam search removed from limitations** | Not planning to support; no need to list. |
| **Prefix caching marked as Planned** | On the roadmap, not permanently unsupported. |
| **Pipeline parallelism IS supported** | Removed incorrect "not supported" limitation entry. |
| **Quantization: only mxfp6 and mxint8 supported** | AWQ, GPTQ, FP8, mxfp4, and compressed-tensors removed from documentation. Only hardware-native formats (mxfp6 compute, mxint8 KV cache) are documented. |
| **PYT only supports Qwen-3-VL and Qwen-2.5-VL** | Both are experimental. InternVL entries removed from PYT models table. |
| **URL namespace** | Target is `QUALCOMM.github.io/vllm-qaic` (migration deferred). |
