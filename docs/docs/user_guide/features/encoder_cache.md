# Encoder Cache (EC)

Share vision encoder embeddings across prefill and decode instances through a single shared-memory store, instead of re-sending them over HTTP on every request.

The **Encoder Cache (EC) connector** is the multimodal counterpart to KV transfer in [disaggregated serving](disaggregated_serving.md). In an xEyPzD deployment, the encode instances produce vision embeddings that the prefill/decode instances consume. EC writes those embeddings once into a shared-memory `ECStore` and hands consumers a lightweight reference (keyed by the multimodal hash), avoiding per-request base64/HTTP copies.

## Why Encoder Cache?

| Without EC | With EC |
|------------|---------|
| Embeddings serialized (base64) and POSTed encode → proxy → prefill/decode | Embeddings written once to shared memory, referenced by hash |
| Proxy RAM grows with concurrency and image size | Flat proxy RAM; single fp16 copy in `/dev/shm` |
| Duplicate images re-sent per request | Deduplicated by multimodal hash |

EC is primarily a **memory-efficiency and scalability** win. The benefit grows with embedding size (large hidden dim, deepstack models) and with fan-out (more prefill/decode instances consuming the same image).

## Architecture

```text
Encode instances (Vision)        Prefill instances
  produce embeddings               consume embeddings
        |                                 ^
        |  PUT (write once)               |  GET (by mm_hash)
        v                                 |
  +-----------------------------------------------+
  |     ECStore  (shared memory, /dev/shm)        |
  |     fp16 embeddings, TTL-evicted, ref-counted |
  +-----------------------------------------------+
            ^
            |  ZMQ control plane (ec_handoff service)
            v
     qaic_disagg proxy / router
```

The control plane (PUT/GET metadata) runs over a ZMQ `ec_handoff` service on its own port; the embedding payload itself lives in shared memory and is never copied through the proxy.

## Usage

EC layers onto a disaggregated xEyPzD launch. Add the EC flags alongside the existing KV handoff flags:

```bash
python3 -m qaic_disagg \
  --model "Qwen/Qwen3.6-35B-A3B" \
  --encode-port 9001,9002 \
  --encode-device-group 0 1 \
  --prefill-port 9101,9102 \
  --prefill-device-group 2 3 \
  --decode-port 9200 \
  --decode-device-group 4:8 \
  --port 8882 \
  --limit-mm-per-prompt '{"image": 1}' \
  --kv-handOff-port 5766 \
  --kv-store-size 512 \
  --ec-connector QaicECConnector \
  --ec-handOff-port 5767 \
  --ec-store-size-gb 64 \
  --ec-store-ttl 3600 \
  --mm-processor-cache-gb 4 \
  --mm-processor-cache-type lru \
  --generation-config vllm \
  --quantization mxfp6 \
  --kv-cache-dtype mxint8
```

EC is enabled purely by `--ec-connector QaicECConnector`; without it the encode→prefill path falls back to the legacy HTTP transfer.

## Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ec-connector` | `None` | EC connector to use. `QaicECConnector` enables shared-memory transfer. |
| `--ec-handOff-port` | `5858` | ZMQ port for the EC handoff control plane. Must differ from `--kv-handOff-port`. |
| `--ec-store-size-gb` | `64` | Maximum `ECStore` size in GB. This is a ceiling, not a reservation; shared memory is allocated on demand. |
| `--ec-store-ttl` | `3600` | Per-entry Time To Live in seconds. Entries older than the TTL are evicted once no instance is using them. |
| `--mm-processor-cache-gb` | `4` | Size (GB) of the per-instance multimodal processor cache that memoizes preprocessing of repeated media. |
| `--mm-processor-cache-type` | `lru` | Eviction policy for the processor cache (`lru` or `shm`). |
| `--mm-shm-cache-max-object-size-mb` | `128` | Max object size (MB) stored in the shared-memory processor cache (when `--mm-processor-cache-type shm`). |
| `--enable-mm-embeds` | — | Set automatically on decode/PD instances in E-PD topologies so they accept pre-encoded embeddings; not set on encode instances. |

!!! note "EC vs mm-processor-cache"
    These are two different caches. `--mm-processor-cache-*` caches **input preprocessing** (tokenized/transformed media) per engine instance; the EC `ECStore` caches **encoder output embeddings** shared across instances. They are complementary.

## Eviction and Reference Counting

Entries are evicted only when they are **both** older than `--ec-store-ttl` **and** not currently referenced by any in-flight request. In-use embeddings are never reaped mid-request. A re-requested image that was already evicted is simply re-encoded.

!!! tip "Sizing /dev/shm"
    The `ECStore` lives in `/dev/shm`. Ensure `/dev/shm` is large enough for the configured `--ec-store-size-gb`, plus the KV store. Over-allocating the store cap is harmless (it is a ceiling), but `/dev/shm` must have the physical room for peak live embeddings.

!!! warning "Constraints"
    - `--ec-handOff-port` must differ from `--kv-handOff-port`.
    - EC requires a disaggregated (xEyPzD) launch with encode instances; it is not used in aggregated serving.
