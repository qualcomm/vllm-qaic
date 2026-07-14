# Supported Models — AOT Mode

Models validated for AOT mode on Cloud AI 100 with vLLM.

!!! tip "Architecture compatibility"
    Models sharing the same architecture (e.g., all `LlamaForCausalLM` variants) are generally
    compatible. The representative models below are explicitly validated, but other models of the
    same architecture family may also work.

## Text-Only Language Models

| Architecture | Model Family | Representative Models | Params |
|---|---|---|---|
| **Olmo2ForCausalLM** | OLMo-2 | [allenai/OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B) | 1B |
| **FalconForCausalLM** | Falcon | [tiiuae/falcon-40b](https://huggingface.co/tiiuae/falcon-40b) | 40B |
| **Qwen3MoeForCausalLM** | Qwen3 MoE | [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | 30B (3B active) |
| **GemmaForCausalLM** | CodeGemma | [google/codegemma-2b](https://huggingface.co/google/codegemma-2b), [google/codegemma-7b](https://huggingface.co/google/codegemma-7b) | 2B, 7B |
| | Gemma | [google/gemma-2b](https://huggingface.co/google/gemma-2b), [google/gemma-7b](https://huggingface.co/google/gemma-7b), [google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b), [google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b), [google/gemma-2-27b](https://huggingface.co/google/gemma-2-27b) | 2B–27B |
| **GptOssForCausalLM** | GPT-OSS | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | 20B |
| **GPTBigCodeForCausalLM** | StarCoder 1.5 | [bigcode/starcoder](https://huggingface.co/bigcode/starcoder) | 15B |
| | StarCoder 2 | [bigcode/starcoder2-15b](https://huggingface.co/bigcode/starcoder2-15b) | 15B |
| **GPTJForCausalLM** | GPT-J | [EleutherAI/gpt-j-6b](https://huggingface.co/EleutherAI/gpt-j-6b) | 6B |
| **GPT2LMHeadModel** | GPT-2 | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | 124M |
| **GraniteForCausalLM** | Granite 3.1 | [ibm-granite/granite-3.1-8b-instruct](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct), [ibm-granite/granite-guardian-3.1-8b](https://huggingface.co/ibm-granite/granite-guardian-3.1-8b) | 8B |
| | Granite 20B | [ibm-granite/granite-20b-code-base-8k](https://huggingface.co/ibm-granite/granite-20b-code-base-8k), [ibm-granite/granite-20b-code-instruct-8k](https://huggingface.co/ibm-granite/granite-20b-code-instruct-8k) | 20B |
| **InternVLChatModel** | InternVL | [OpenGVLab/InternVL2_5-1B](https://huggingface.co/OpenGVLab/InternVL2_5-1B), [OpenGVLab/InternVL3_5-1B](https://huggingface.co/OpenGVLab/InternVL3_5-1B) | 1B |
| **LlamaForCausalLM** | CodeLlama | [codellama/CodeLlama-7b-hf](https://huggingface.co/codellama/CodeLlama-7b-hf), [codellama/CodeLlama-13b-hf](https://huggingface.co/codellama/CodeLlama-13b-hf), [codellama/CodeLlama-34b-hf](https://huggingface.co/codellama/CodeLlama-34b-hf) | 7B–34B |
| | DeepSeek-R1-Distill-Llama | [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) | 70B |
| | InceptionAI (JAIS) | [inceptionai/jais-adapted-7b](https://huggingface.co/inceptionai/jais-adapted-7b), [inceptionai/jais-adapted-13b-chat](https://huggingface.co/inceptionai/jais-adapted-13b-chat), [inceptionai/jais-adapted-70b](https://huggingface.co/inceptionai/jais-adapted-70b) | 7B–70B |
| | Llama 3.3 | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | 70B |
| | Llama 3.2 | [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B), [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) | 1B, 3B |
| | Llama 3.1 | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B) | 8B, 70B |
| | Llama 3 | [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B), [meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B) | 8B, 70B |
| | Llama 2 | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf), [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) | 7B–70B |
| | Vicuna | [lmsys/vicuna-13b-delta-v0](https://huggingface.co/lmsys/vicuna-13b-delta-v0), [lmsys/vicuna-13b-v1.3](https://huggingface.co/lmsys/vicuna-13b-v1.3), [lmsys/vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5) | 13B |
| **MistralForCausalLM** | Mistral | [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) | 7B |
| **MixtralForCausalLM** | Codestral | [mistralai/Codestral-22B-v0.1](https://huggingface.co/mistralai/Codestral-22B-v0.1) | 22B |
| | Mixtral | [mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) | 8x7B |
| **Phi3ForCausalLM** | Phi-3 / Phi-3.5 | [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) | 3.8B |
| **QwenForCausalLM** | DeepSeek-R1-Distill-Qwen | [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) | 32B |
| | Qwen2 / Qwen2.5 | [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) | 1.5B+ |

??? example "Quick start — Llama 3.1 8B with SpD"
    ```bash
    vllm serve meta-llama/Llama-3.1-8B-Instruct \
      --max-num-seqs 8 \
      --max-model-len 2048 \
      --quantization mxfp6 \
      --kv-cache-dtype mxint8 \
      --speculative-config '{"method":"ngram","num_speculative_tokens":5}'
    ```

---

## Multimodal / Vision-Language Models

Support varies by QPC mode (Single vs Dual). Dual QPC uses the `kv_offload` architecture where the vision encoder and language model run on separate device groups.

| Architecture | Model Family | Representative Models | Single QPC | Dual QPC |
|---|---|---|---|---|
| **LlavaForConditionalGeneration** | LLaVA-1.5 | [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | :white_check_mark: | :white_check_mark: |
| **MllamaForConditionalGeneration** | Llama 3.2 Vision | [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct), [meta-llama/Llama-3.2-90B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct) | :white_check_mark: | :white_check_mark: |
| **LlavaNextForConditionalGeneration** | Granite Vision | [ibm-granite/granite-vision-3.2-2b](https://huggingface.co/ibm-granite/granite-vision-3.2-2b) | :x: | :white_check_mark: |
| **Llama4ForConditionalGeneration** | Llama-4-Scout | [meta-llama/Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) | :white_check_mark: | :white_check_mark: |
| **Qwen2_5_VLForConditionalGeneration** | Qwen2.5-VL | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | :x: | :white_check_mark: |
| **Qwen3VLForConditionalGeneration** | Qwen3-VL | [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) | :x: | :white_check_mark: |
| **Qwen3VLMoeForConditionalGeneration** | Qwen3-VL MoE | [Qwen/Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) | :x: | :white_check_mark: |

!!! warning "Multimodal limitations"
    - Multimodal models use the `kv_offload` architecture (vision encoder on separate device group)
    - Cannot combine with speculative decoding
    - See [Multimodal Guide](../features/multimodal.md) for configuration details

---

## Embedding Models

| Architecture | Model Family | Representative Models |
|---|---|---|
| **BertModel** | BGE | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5), [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5), [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), [intfloat/e5-large-v2](https://huggingface.co/intfloat/e5-large-v2) |
| **MPNetForMaskedLM** | MPNet | [sentence-transformers/multi-qa-mpnet-base-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-cos-v1) |
| **RobertaModel** | RoBERTa (Granite) | [ibm-granite/granite-embedding-30m-english](https://huggingface.co/ibm-granite/granite-embedding-30m-english), [ibm-granite/granite-embedding-125m-english](https://huggingface.co/ibm-granite/granite-embedding-125m-english) |
| **XLMRobertaForSequenceClassification** | XLM-RoBERTa (Reranker) | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| **XLMRobertaModel** | XLM-RoBERTa | [ibm-granite/granite-embedding-107m-multilingual](https://huggingface.co/ibm-granite/granite-embedding-107m-multilingual), [ibm-granite/granite-embedding-278m-multilingual](https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual), [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) |

??? example "Quick start — BGE embedding"
    ```bash
    vllm serve BAAI/bge-base-en-v1.5 \
      --task embedding \
      --max-num-seqs 16 \
      --max-model-len 512 \
      --quantization mxfp6 \
      --kv-cache-dtype mxint8
    ```

---

## Audio / Encoder-Decoder Models

| Architecture | Model Family | Representative Models |
|---|---|---|
| **Whisper** | Whisper | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny), [openai/whisper-base](https://huggingface.co/openai/whisper-base), [openai/whisper-small](https://huggingface.co/openai/whisper-small), [openai/whisper-medium](https://huggingface.co/openai/whisper-medium), [openai/whisper-large](https://huggingface.co/openai/whisper-large), [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) |

---

## Quantization Support

All text-only models listed above support:

| Quantization | Status | Notes |
|-------------|--------|-------|
| mxfp6 (compute) | :white_check_mark: Recommended | Hardware-native, best throughput/quality balance |
| mxint8 (KV cache) | :white_check_mark: Recommended | Pair with mxfp6 for optimal config |

---

## Notes

- :material-information-outline: Set `trust_remote_code=True` for Falcon, Phi-3 family models
- :material-information-outline: Pass `disable_sliding_window` for Gemma family models when using vLLM

---

!!! info "QEfficient Reference"
    The full validated model matrix (including quantization variants, SwiftKV, and PEFT/LoRA support)
    is maintained in the QEfficient documentation:
    [Validated Models](https://quic.github.io/efficient-transformers/source/validate.html)
