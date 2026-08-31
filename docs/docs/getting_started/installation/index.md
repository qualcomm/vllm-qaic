# Installation Overview

`vllm-qaic` can be installed in two modes. Each requires its own Python environment — they cannot coexist.

## Installation Methods

| Method | Description | Time |
|--------|-------------|------|
| **Docker** (recommended for evaluation) | Pre-built container from GHCR | ~1 min |
| **Scripted** (recommended for development) | `./scripts/install.sh aot \| pyt` | ~5 min |
| **Manual** | Step-by-step pip commands | ~10 min |
| **Wheel** | Pre-built wheels from SDK or CI | ~3 min |

## Choose Your Mode

=== "AOT Mode"

    Best for production. Uses QEfficient to compile models into static QPCs.

    - [Prerequisites](prerequisites.md)
    - [AOT Installation](aot_mode.md)
    - [Verification](verification.md)

=== "PYT Mode"

    Best for development. Uses torch_qaic for dynamic execution.

    - [Prerequisites](prerequisites.md)
    - [PYT Installation](pyt_mode.md)
    - [Verification](verification.md)
