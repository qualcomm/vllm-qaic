# embedding — Unit Tests

**Embedding model support — vector validation, model lists, override_qaic_config pooling keys (pure Python)**

> **API note:** On-device embedding tests use `qaic_model.embed(prompts)` — **not** `encode()`.
> The vLLM API requires `pooling_task` for `encode()`; use `embed()` for embeddings.

## Tests

| File | Tests |
|------|-------|
| `test_embedding_utils.py` | 46 |

**Total: 46 tests, pure Python (no hardware)**

## `test_embedding_utils.py`

- `TestCheckVectorNormalized` / `TestCheckVectorSoftmax` / `TestCheckVectorRaw` — `_check_vector()` helper logic
- `TestEmbeddingModelLists` — supported embedding/cross-encoder model lists
- `TestTaskClassification` — SEQWISE_TASKS / TOKWISE_TASKS
- `TestOverrideQaicConfigPooling` — `_clean_config()` normalization of pooling_device/pooling_method/normalize/softmax/task/embed_seq_len
- `TestPrefixCachingEnabled` — prefix_caching_enabled env-flag logic

> On-device embedding tests (output shape, unit norm, determinism, self-similarity,
> distinct-prompts-differ) live in
> `vllm-qaic/tests/device_unit_and_e2e/test_unit_qaic_embedding.py`
> (`TestEmbeddingOnDevice`), using the `device/` fixture system (`qaic_model`) — this
> folder's own `test_embedding_integration.py` (which depended on a nonexistent
> `embed_llm` fixture) has been removed as redundant.

## Running

```bash
# Pure Python tests only
pytest embedding/test_embedding_utils.py -v

# On-device tests (requires QAIC hardware + embedding model)
pytest device_unit_and_e2e/test_unit_qaic_embedding.py -v \
    --model-name BAAI/bge-base-en-v1.5 \
    --ctx-len 512 --decode-bsz 4 --device-group 1 --device-id 0
```
