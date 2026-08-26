# Release Notes

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
