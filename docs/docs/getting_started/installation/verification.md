# Verification

After installation, verify your environment is correctly set up.

!!! tip "Run all checks"
    Complete all four steps below. A passing smoke test at the end confirms your full stack is working.

## Import Checks

```bash
# Check torch is CPU-only
python -c "import torch; print(torch.__version__)"
# AOT expected: 2.7.0+cpu
# PYT expected: 2.11.0+cpu

# Verify vllm-qaic plugin loads
python -c "import vllm_qaic; print('vllm_qaic OK')"

# PYT only: verify torch_qaic
python -c "import torch_qaic; print('torch_qaic OK')"
```

## Confirm No CUDA Packages

```bash
pip list | grep -i "nvidia\|cuda-toolkit\|cuda-bin"
# Should return no output
```

## Device Check

```bash
# Verify QAIC devices are accessible
export QAIC_VISIBLE_DEVICES=0
python -c "
from vllm_qaic import envs
print('QAIC environment loaded successfully')
"
```

## Smoke Test Inference

=== "AOT Mode"

    ```python
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_num_seqs=4,
        max_model_len=256,
        quantization="mxfp6",
        kv_cache_dtype="mxint8",
    )
    outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=32))
    print(outputs[0].outputs[0].text)
    ```

=== "PYT Mode"

    ```python
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_num_seqs=4,
        max_model_len=256,
        enforce_eager=True,
        async_scheduling=False,
    )
    outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=32))
    print(outputs[0].outputs[0].text)
    ```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: torch_qaic` | Wrong mode or env | Ensure you're in PYT env with SDK `--install-torch-qaic` |
| `ModuleNotFoundError: vllm_qaic` | Plugin not installed | Run `pip install ./vllm-qaic` |
| CUDA packages found | Mixed environment | Create a fresh env without CUDA torch |
| Device not found | SDK/driver not installed | Follow [Prerequisites](prerequisites.md) |
