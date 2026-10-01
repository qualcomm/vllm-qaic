# Qwen3-ASR QAIC Results

## Configuration

| Item | Value |
|---|---|
| Model | `Qwen/Qwen3-ASR-0.6B-hf` |
| Hardware | One Qualcomm Cloud AI 100 |
| SDK | `AIC.1.22.0.99` Platform and Apps |
| Precision | FP16 |
| Batch size | 1 |
| Chunk size | 30 seconds |
| Chunk overlap | 0 seconds |
| Prefill sequence length | 512 |
| Encoder context length | 3000 feature frames |
| Decoder context length | 512 tokens |
| Generation limit | 128 tokens |
| EOS stopping | Enabled |

The QPC session was created once and reused for all files. One warmup request was excluded from timing.

## Performance

The host-SDK benchmark covered 81 files and 174 chunks. Total measured wall time was 1451.740 seconds, with 0.641 seconds of excluded warmup.

RTFx is input audio duration divided by measured inference wall time.

| Language | Duration | Files | Mean clip E2E | P50 | P90 | RTFx |
|---|---:|---:|---:|---:|---:|---:|
| English | 40 s | 16 | 13.465 s | 13.251 s | 15.406 s | 2.9893x |
| English | 50 s | 16 | 15.475 s | 14.994 s | 17.218 s | 3.2605x |
| English | 60 s | 16 | 19.516 s | 19.879 s | 21.481 s | 3.0902x |
| Arabic | 40 s | 10 | 16.585 s | 16.624 s | 17.536 s | 2.4423x |
| Arabic | 50 s | 16 | 21.578 s | 21.259 s | 23.989 s | 2.3352x |
| Arabic | 60 s | 7 | 23.622 s | 23.692 s | 23.695 s | 2.5216x |

| Aggregate metric | Value |
|---|---:|
| Mean chunk E2E | 8.334 s |
| P50 chunk E2E | 9.061 s |
| P90 chunk E2E | 11.844 s |
| Mean whole-clip E2E | 17.923 s |
| P50 whole-clip E2E | 17.125 s |
| P90 whole-clip E2E | 23.579 s |

## Accuracy

### English: LibriSpeech Test-Clean

| Clips | Reference words | Edit operations | WER |
|---:|---:|---:|---:|
| 2,620 | 52,576 | 1,290 | **2.4536%** |

### Arabic: UBC-NLP/Casablanca

The evaluation used 128 clips: 16 clips for each dialect.

| Dialect | Clips | Reference words | Edits | WER |
|---|---:|---:|---:|---:|
| Algeria | 16 | 85 | 82 | 96.4706% |
| Egypt | 16 | 114 | 87 | 76.3158% |
| Jordan | 16 | 169 | 101 | 59.7633% |
| Mauritania | 16 | 124 | 114 | 91.9355% |
| Morocco | 16 | 201 | 185 | 92.0398% |
| Palestine | 16 | 204 | 132 | 64.7059% |
| UAE | 16 | 191 | 129 | 67.5393% |
| Yemen | 16 | 266 | 159 | 59.7744% |
| **Aggregate** | **128** | **1,354** | **989** | **73.0428%** |

The Casablanca WER has substantial dialect variation and should not be treated as a customer-quality claim without additional language-prompting, normalization, and reference-alignment validation.

## Measurement Notes

- E2E latency includes audio preprocessing, QPC inference, and decoding.
- QPC session initialization is excluded.
- Results are batch-1, single-device measurements and are not contractual SLA values.
- Long audio was split into 30-second chunks; application-level overlap is independent of QPC compilation.
