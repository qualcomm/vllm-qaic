# Disaggregated Serving

Independently optimize latency (TTFT) and throughput (TPOT) by running prefill and decode on separate hardware.

Disaggregated serving splits inference into separate **prefill** and **decode** stages, each running on independent device groups and optimized for their workload characteristics.

## Why Disaggregated Serving?

| Phase | Workload | Optimized For |
|-------|----------|---------------|
| **Prefill** | Compute-bound, bursty | Time to First Token (TTFT) |
| **Decode** | Memory-bandwidth-bound, sustained | Time Per Output Token (TPOT) |

Running both on the same hardware forces a compromised configuration. Disaggregation lets each be tuned independently.

## Architecture

```text
Client (OpenAI-compatible request)
        |
        v
qaic_disagg Router / API Server
        |
   +----+----+
   |         |
   v         v
Prefill    Decode
Stage      Stage
(Group A)  (Group B)
   |         |
   +-- KV -->+
   transfer  |
             v
      Streaming tokens
      back to client
```

The disaggregation is invisible to the client — a single OpenAI-compatible endpoint is exposed.

## Docker Deployment

Qualcomm ships a dedicated container with a pre-configured `qaic_disagg` entrypoint:

```bash
docker pull ghcr.io/quic/cloud_ai_inference_vllm_disagg:1.21.2.0
```

Launch prefill-decode disaggregation:

```bash
docker run --rm -it \
  --privileged \
  --shm-size=4gb \
  --network host \
  -e HF_TOKEN=<your_hf_token> \
  -e HF_HOME=/workspace/hf_cache \
  -e QEFF_HOME=/workspace/qeff_cache \
  -v $PWD/hf_cache:/workspace/hf_cache \
  -v $PWD/qeff_cache:/workspace/qeff_cache \
  ghcr.io/quic/cloud_ai_inference_vllm_disagg:1.21.2.0 \
  --model "meta-llama/Llama-3.2-1B-Instruct" \
  --prefill-port 9900 \
  --prefill-device-group 0..15 \
  --decode-port 9800 \
  --decode-device-group 16..23 \
  --port 8000 \
  --prefill-max-num-seqs 1 \
  --decode-max-num-seqs 1 \
  --prefill-max-seq-len-to-capture 256 \
  --decode-max-seq-len-to-capture 256 \
  --max-model-len 11264 \
  --prefill-override-qaic-config "split_retained_state_io:True mxfp6_matmul:True" \
  --decode-override-qaic-config "split_retained_state_io:True mxfp6_matmul:True" \
  --generation-config vllm \
  --kv-cache-dtype mxint8 -v -v -v
```

## Speculative Decoding with Disaggregated Serving

SpD can be layered onto disaggregated serving — proposals are generated and verified entirely on the decode node:

```text
Prefill Stage (Device Group 0..15)
  +-- Generates KV cache (no SpD overhead)

Decode Stage (Device Group 16..23)
  +-- Draft model proposes tokens
  +-- Target model verifies and streams output
```

Enable with `--decode-speculative-config`:

```bash
docker run --rm -it \
  --privileged --shm-size=4gb --network host \
  -e HF_TOKEN=<your_hf_token> \
  ghcr.io/quic/cloud_ai_inference_vllm_disagg:1.21.2.0 \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --prefill-port 9900 \
  --prefill-device-group 0..15 \
  --decode-port 9800 \
  --decode-device-group 16..23 \
  --port 8000 \
  --prefill-max-num-seqs 1 \
  --decode-max-num-seqs 4 \
  --max-model-len 2048 \
  --prefill-override-qaic-config "mxfp6_matmul:True" \
  --decode-override-qaic-config "mxfp6_matmul:True num_cores:10" \
  --decode-speculative-config '{"speculative_model":"meta-llama/Llama-3.2-1B-Instruct","num_speculative_tokens":3,"draft_override_qaic_config":{"device_group":[0],"num_cores":6}}' \
  --kv-cache-dtype mxint8 \
  --generation-config vllm
```

For n-gram without a draft model:

```bash
--decode-speculative-config '{"method":"ngram","num_speculative_tokens":5}'
```

## Disaggregation Modes

| Configuration | Encode | Prefill | Decode | Status |
|---------------|--------|---------|--------|--------|
| xPyD | — | x **P**refill nodes | y **D**ecode nodes | :white_check_mark: |
| xEyPD | x **E**ncode nodes (Vision) | — | y **PD** (Prefill+Decode) nodes | :white_check_mark: |
| xEyPzD | x **E**ncode nodes (Vision) | y **P**refill nodes | z **D**ecode nodes | :white_check_mark: |

## LMCache Integration

LMCache turns the KV cache into a fleet-wide resource, enabling cache sharing across engines:

- KV cache extracted from device memory to CPU/disk/distributed stores
- Reused across requests (long system prompts, RAG context, multi-turn chats)

## Key Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `--prefill-device-group` | QID range for prefill (e.g., `0..15`) |
| `--decode-device-group` | QID range for decode (e.g., `16..23`) |
| `--prefill-max-num-seqs` | Batch size for prefill stage |
| `--decode-max-num-seqs` | Batch size for decode stage |
| `--prefill-override-qaic-config` | Compilation flags for prefill QPCs |
| `--decode-override-qaic-config` | Compilation flags for decode QPCs |
