# multimodal — Unit Tests

Tests for QAIC multimodal (vision-language) model support.

## What is tested

The multimodal CI script (`run_multimodal_test_qaic.sh`) tests VLMs (Qwen2.5VL, Qwen3VL, InternVL, LLaVA, Gemma, Granite, LLaMA4) on QAIC hardware. These unit tests cover the QAIC-specific model detection and config merging logic.

## Test files

| File | Tests | Description |
|------|-------|-------------|
| `test_multimodal_utils.py` | ~35 | `is_internvl()`, `is_llama4()`, `is_qwenvl()`, `is_gemma()`, `is_granite()` model name detection; `update_qaic_config()` merging (base preserved, updates applied, None ignored, Qwen VL gets height/width, non-Qwen does not, base not mutated); `DYNAMIC_RESOLUTION_MODELS` list |

## Running

```bash
# All tests (no hardware required)
pytest multimodal/ -v
```

## Existing upstream vLLM tests (informational)

| Upstream file | What it tests |
|---------------|---------------|
| `tests/models/multimodal/` | Multimodal model correctness (image/video/audio) |
| `tests/models/multimodal/test_qwen2_vl.py` | Qwen2-VL image encoding |
| `tests/multimodal/` | Multimodal input processing, registry |

These are generic vLLM tests. QAIC-specific VLM handling (dynamic resolution, custom MM processor) is tested here.
