# FAQ

Common questions and solutions for vLLM QAIC.

---

## Installation & Setup

??? question "Can I run AOT and PYT modes in the same environment?"
    No. The two modes require different torch versions and conflicting packages.
    Use separate virtual environments.

    ```bash
    # Create isolated environments
    python -m venv ~/.venvs/vllm-aot
    python -m venv ~/.venvs/vllm-pyt
    ```

??? question "`ModuleNotFoundError: No module named 'torch_qaic'`"
    This means you're trying to use PYT mode without `torch_qaic` installed.

    **Fix:** Ensure the Apps SDK was installed with `--install-torch-qaic` and run:
    ```bash
    pip install /opt/qti-aic/integrations/torch_qaic/py312/torch_qaic-*.whl
    ```

??? question "`uv` rejects `torch==2.7.0+cpu` during install"
    `uv` does not support PEP 440 `+local` version labels from remote indexes.

    **Fix:** Use `python -m pip install` for torch instead of `uv pip install`.

---

## Runtime & Errors

??? question "QAIC device not found"
    Verify device access step-by-step:

    1. Check SDK/drivers are installed:
       ```bash
       qaic-util | grep -i status
       ```
    2. Check device visibility:
       ```bash
       echo $QAIC_VISIBLE_DEVICES
       ```
    3. Ensure `/dev/accel/` exists and has correct permissions:
       ```bash
       ls -la /dev/accel/
       ```

??? question "Out of memory on device"
    Reduce workload parameters:

    | Parameter | Effect |
    |-----------|--------|
    | `max_num_seqs` | Lower decode batch size |
    | `max_model_len` | Reduce context length |
    | `num_cores` in `additional_config` | Free cores if using SpD |

    See [Device Management](../user_guide/configuration/device_management.md) for core allocation details.

??? question "Model compilation takes too long"
    QPC compilation is a **one-time cost**. Pre-compiled QPCs are cached in `$QEFF_HOME`
    (default: `~/.cache/qefficient/`).

    **Skip compilation entirely** by pointing to a pre-compiled QPC directory:
    ```bash
    export VLLM_QAIC_QPC_PATH=/path/to/precompiled/qpc/
    ```

??? question "Do I need to pass `enforce_eager`?"
    No. The QAIC platform sets it for you: `True` in **PYT mode**, `False` in **AOT mode**,
    where execution is always through pre-compiled QPCs. Passing it yourself is unnecessary,
    and a value that disagrees with the selected mode is overridden (in AOT mode, overriding
    an explicit `enforce_eager=True` also logs a warning).

---

## Performance Tuning

??? question "Why is speculative decoding slower than expected?"
    SpD is most effective when:

    - :white_check_mark: Output tokens are predictable (summarization, code completion with long context)
    - :white_check_mark: Acceptance rate is high (> 60%)
    - :white_check_mark: Batch size is moderate (1-4 sequences)

    At high batch sizes, the hardware may already be saturated and SpD adds overhead.

    **Diagnose:** See [Profiling](../developer_guide/profiling.md) for step-by-step analysis.

??? question "How do I check which QPC was loaded?"
    Enable verbose logging:

    ```bash
    # Server mode
    vllm serve ... -v -v -v

    # Or via environment variable
    export QAIC_DEBUG=1
    ```

    The log output will show the QPC path and compilation parameters used.

---

## Still stuck?

If your issue isn't covered here:

1. Check the [Verification](installation/verification.md) page for post-install diagnostics
2. Review [Environment Variables](../user_guide/configuration/environment_variables.md) for relevant settings
3. Open an issue on [GitHub](https://github.com/quic/vllm-qaic/issues) with:
    - Your SDK version (`qaic-util -V`)
    - Python environment info (`pip list | grep -E "vllm|torch|qaic"`)
    - Full error traceback
