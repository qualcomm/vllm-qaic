# Prerequisites

<!-- --8<-- [start:installation] -->

Qualcomm Cloud AI 100 accelerator with QAIC Platform/Apps SDK >= 1.22.0 on Linux (Ubuntu 22.04+), Python 3.12.

<!-- --8<-- [end:installation] -->

<!-- --8<-- [start:requirements] -->

## Platform Requirements

| Requirement | AOT Mode | PYT Mode |
|-------------|----------|----------|
| OS | Ubuntu 22.04+ | Ubuntu 22.04+ |
| Python | 3.12 | 3.12 |
| QAIC Platform SDK | >= 1.22.0 | >= 1.22.0 |
| QAIC Apps SDK | >= 1.22.0 | >= 1.22.0 (with `--install-torch-qaic`) |
| torch | 2.7.0+cpu | 2.10.0+cpu |
| vLLM | 0.15.0 | 0.15.0 |
| QEfficient | main | — |

## Hardware

| Platform | Description |
|----------|-------------|
| Qualcomm Cloud AI 100 Ultra | 4 QIDs per card, 16 NSP cores per QID |
| Qualcomm Cloud AI 100 Standard | Data center inference accelerator |
| Qualcomm Cloud AI 080 | Entry-level Cloud AI accelerator |

<!-- --8<-- [end:requirements] -->

<!-- --8<-- [start:set-up-using-python] -->

## Step 1: Install Cloud AI SDK

Follow the official SDK installation guide:

1. [Install Cloud AI SDK and Drivers](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/index.html)
2. [Verify your installation](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/verification.html)

!!! note "PYT mode only"
    Run the Apps SDK installer with `--install-torch-qaic` to build `torch_qaic` wheels
    into `/opt/qti-aic/integrations/torch_qaic/py312/`.

## Step 2: Create Python 3.12 Environment

=== "conda"

    ```bash
    conda create -n vllm-qaic python=3.12
    conda activate vllm-qaic
    ```

=== "uv (fastest)"

    ```bash
    uv venv .venv --python 3.12
    source .venv/bin/activate
    ```

=== "venv"

    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate
    ```

## Step 3: Verify Device Access

```bash
# Check QAIC devices are visible
qaic-util | grep -i "qid\|status"

# Set device visibility
export QAIC_VISIBLE_DEVICES=0   # comma-separated QIDs
```

!!! info "QEfficient Reference"
    The QEfficient library handles model compilation for AOT mode. See the
    [QEfficient Installation Guide](https://quic.github.io/efficient-transformers/source/installation.html)
    for additional context on the compilation stack.

<!-- --8<-- [end:set-up-using-python] -->
