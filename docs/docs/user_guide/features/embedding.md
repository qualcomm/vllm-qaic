# Embedding Models

Embedding networks transform high-dimensional inputs — text, images, items — into dense, low-dimensional vectors that capture semantic relationships. These vectors enable tasks such as similarity search, reranking, classification, and recommendation.

## Supported Models

| Architecture | Model Family | Representative Models |
|---|---|---|
| **BertModel** | BGE | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5), [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5), [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), [intfloat/e5-large](https://huggingface.co/intfloat/e5-large) |
| **XLMRobertaModel** | XLM-RoBERTa | [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large), [ibm-granite/granite-embedding-107m-multilingual](https://huggingface.co/ibm-granite/granite-embedding-107m-multilingual), [ibm-granite/granite-embedding-278m-multilingual](https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual) |
| **RobertaModel** | RoBERTa (Granite) | [ibm-granite/granite-embedding-30m-english](https://huggingface.co/ibm-granite/granite-embedding-30m-english), [ibm-granite/granite-embedding-125m-english](https://huggingface.co/ibm-granite/granite-embedding-125m-english) |
| **XLMRobertaForSequenceClassification** | XLM-RoBERTa (Reranker) | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| **MistralModel** | E5-Mistral | [intfloat/e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct) |
| **NomicBertModel** | Nomic | [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) |
| **BertModel (Jina)** | Jina | [jinaai/jina-embeddings-v2-base-en](https://huggingface.co/jinaai/jina-embeddings-v2-base-en), [jinaai/jina-embeddings-v2-base-code](https://huggingface.co/jinaai/jina-embeddings-v2-base-code) |

!!! note "Limitation"
    `sentence-transformers/gtr-t5-large` is not supported. Some tasks may not be compatible with certain models.
    Jina and nomic-ai models require `trust_remote_code=True`.

## Usage

"embed" example
```python
from vllm import LLM

prompts = ["Hello, my name is"] * 10
# CPU pooling — compile for multiple sequence lengths
model = LLM(
    model="intfloat/multilingual-e5-large",
    runner="pooling",
    enforce_eager=True,
    max_num_seqs=4,
    max_model_len=256,
    additional_config={
        "device_group": [0],
        "override_qaic_config": {
            "pooling_device": "cpu",
            "embed_seq_len": [32, 256],  # always include max_model_len
        },
    },
)
outputs = model.embed(prompts)

for prompt, output in zip(prompts, outputs):
    embeds = output.outputs.embedding
    print(f"Prompt: {prompt!r}, Embedding size: {len(embeds)}")

# QAIC pooling — single sequence length
model = LLM(
    model="intfloat/multilingual-e5-large",
    runner="pooling",
    enforce_eager=True,
    max_num_seqs=4,
    max_model_len=256,
    additional_config={
        "device_group": [0],
        "override_qaic_config": {
            "pooling_device": "qaic",
            "pooling_method": "mean",
            "normalize": True,
        },
    },
)
outputs = model.embed(prompts)

for prompt, output in zip(prompts, outputs):
    embeds = output.outputs.embedding
    print(f"Prompt: {prompt!r}, Embedding size: {len(embeds)}")
```

Run the full example:

```bash
python examples/offline_inference/basic/qaic_embed.py
```

```python
# classify - CPU
from vllm import LLM

prompts = ["Hello, my name is"] * 10

model = LLM(
    model="BAAI/bge-reranker-v2-m3",
    runner="pooling",
    enforce_eager=True,
    max_num_seqs=4,
    max_model_len=512,
    additional_config={
        "device_group": [0],
        "override_qaic_config": {"pooling_device": "cpu", "task": "classify"},
    },
)
outputs = model.classify(prompts)
```

Run the full example:

```bash
python examples/offline_inference/basic/qaic_classify.py
```

```python
# score - QAIC
from vllm import LLM

query = "What is the capital of France?"
passages = [
    "Paris is the capital and most populous city of France.",
    "The Eiffel Tower is located in Paris.",
]

model = LLM(
    model="BAAI/bge-reranker-v2-m3",
    runner="pooling",
    enforce_eager=True,
    max_num_seqs=4,
    max_model_len=512,
    additional_config={
        "device_group": [0],
        "override_qaic_config": {"pooling_device": "qaic", "task": "score"},
    },
)
outputs = model.score(query, passages)
```

Run the full example:

```bash
python examples/offline_inference/basic/qaic_score.py
```

## Configuration

| Parameter | Description |
|---|---|
| `runner` | Set to `"pooling"` for embedding models |
| `task` | `"embed"`, `"encode"`, `"reward"`, `"classify"`, or `"score"` |
| `override_qaic_config.pooling_device` | `"qaic"` to run pooler on device, `"cpu"` to run on CPU |
| `override_qaic_config.pooling_method` | Pooling method for `qaic` device: `"mean"`, `"avg"`, `"cls"`, `"max"`, or custom |
| `override_qaic_config.normalize` | `True` to apply L2 normalization to pooled outputs (`qaic` only) |
| `override_qaic_config.softmax` | `True` to apply softmax to pooled outputs (`qaic` only) |
| `override_qaic_config.embed_seq_len` | List of sequence lengths to compile for, e.g. `[32, 256]`. Must include `max_model_len` |
| `pooler_config` | Pass a `PoolerConfig` object with `pooling_type`, `use_activation`|

## Notes

- Set `max_seq_len_to_capture` equal to the context length. For multi-sequence-length compilation, `max_model_len` must be one of the values in `embed_seq_len`.
- Use the correct API for the task: `embed()`, `encode()`, `classify()`, or `score()`.
- Jina and nomic-ai models require `trust_remote_code=True`.

!!! warning "jina-embeddings-v2-base-en accuracy patch"
    This model requires a one-time patch for accuracy:

    ```python
    from QEfficient import QEFFAutoModel
    import os, subprocess, requests

    qeff_model = QEFFAutoModel.from_pretrained(
        "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
    )
    os.chdir(os.path.join(
        os.environ.get("HF_HOME"),
        "modules/transformers_modules/jinaai/jina-bert-implementation/"
        "f3ec4cf7de7e561007f27c9efc7148b0bd713f81/"
    ))
    response = requests.get(
        "https://huggingface.co/jinaai/jina-bert-implementation/discussions/7/files.diff"
    )
    with open("pr7.diff", "wb") as f:
        f.write(response.content)
    subprocess.run(["patch", "-p1", "-i", "pr7.diff"], check=True)
    ```
