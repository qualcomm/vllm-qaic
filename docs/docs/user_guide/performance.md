# Performance

Guidelines and expectations for vLLM QAIC performance tuning.

!!! note "These are qualitative guidelines"
    Actual performance depends on model size, batch configuration, prompt length, and
    hardware generation. Use the [Profiling Guide](../developer_guide/profiling.md) for
    precise measurements on your workload.

---

## Key Metrics

| Metric | Definition | What affects it |
|--------|-----------|-----------------|
| **TTFT** | Time to First Token — latency from request to first token | Prefill speed, prompt length, model size |
| **TPOT** | Time Per Output Token — latency between generated tokens | Decode speed, batch size, KV cache pressure |
| **Throughput** | Total tokens generated per second across all requests | Batch size, decode efficiency, SpD acceptance |

---

## Performance Levers

### Speculative Decoding

SpD provides the largest single-feature improvement to decode throughput:

| Method | Expected Speedup | Best Workloads |
|--------|-----------------|----------------|
| N-gram | 1.3-2.0x | Summarization, repetitive output, code |
| Suffix | 1.5-2.5x | Code completion, long-context echo |
| Draft model | 1.5-2.5x | General text generation |

!!! tip "When SpD helps most"
    - Low batch sizes (1-4 sequences) — hardware isn't already saturated
    - Predictable output — summarization, structured data extraction
    - Acceptance rate > 60% — monitor via vLLM logs or profiling

    SpD overhead can hurt at high batch sizes where the device is already saturated.

### Quantization

| Quantization | Compute Precision | KV Cache | Trade-off |
|-------------|-------------------|----------|-----------|
| mxfp6 + mxint8 | mxfp6 | mxint8 | **Recommended** — best throughput/quality balance |
| FP16 (PYT) | fp16 | fp16 | Highest quality, lowest throughput |

### Batch Configuration

| Parameter | Impact | Guidance |
|-----------|--------|----------|
| `max_num_seqs` | Higher = more throughput, but higher TPOT | Start with 8-16, increase until TPOT exceeds SLO |
| `max_model_len` | Higher = more memory, limits batch size | Set to actual max prompt+completion length needed |

---

## Disaggregated Serving

Separating prefill and decode onto different nodes allows independent optimization:

```text
Request → Prefill Node (TTFT-optimized) → KV transfer → Decode Node (TPOT-optimized) → Response
```

| Topology | Use Case |
|----------|----------|
| 1P1D | Balanced latency and throughput |
| 1P2D | Throughput-optimized (decode-bound workloads) |
| 2P1D | Latency-optimized (long-prompt workloads) |

---

## Optimization Checklist

- [ ] Use `mxfp6` quantization with `mxint8` KV cache (default recommendation)
- [ ] Enable speculative decoding for decode-bound workloads
- [ ] Set `max_model_len` to actual max needed (don't over-allocate)
- [ ] Tune `max_num_seqs` to balance throughput vs latency
- [ ] Consider disaggregated serving for strict SLO requirements
- [ ] Profile to identify bottlenecks (see [Profiling](../developer_guide/profiling.md))

---

## Profiling

For detailed step-by-step performance analysis, see the [Profiling Guide](../developer_guide/profiling.md) which covers:

- Per-step timing breakdown (prefill, decode, sampling, scheduling)
- Interactive visualization with snakeviz
- Comparative analysis between configurations
- Identifying hardware vs software bottlenecks
