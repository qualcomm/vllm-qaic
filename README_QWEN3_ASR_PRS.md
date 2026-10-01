# Qwen3-ASR PR Setup

For the concise compile-then-serve instructions, use
[`README_QWEN3_ASR.md`](README_QWEN3_ASR.md).

This file covers only the Qwen3-ASR pull requests:

- QEfficient PR 1276: https://github.com/quic/efficient-transformers/pull/1276
- vLLM-QAIC PR 146: https://github.com/qualcomm/vllm-qaic/pull/146

Prerequisites: Python 3.12, Qualcomm Cloud AI SDK/drivers, one available
AI100 device, and a compiled Qwen3-ASR QPC containing `programqpc.bin`.

## 1. QEfficient PR

```bash
mkdir qwen3-asr
cd qwen3-asr

git clone https://github.com/quic/efficient-transformers.git efficient-transformers
cd efficient-transformers
git fetch origin pull/1276/head:qeff-pr-1276
git checkout qeff-pr-1276
cd ..

git clone https://github.com/qualcomm/vllm-qaic.git vllm-qaic
cd vllm-qaic
git fetch origin pull/146/head:vllm-qaic-pr-146
git checkout vllm-qaic-pr-146
cd ..

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install \
  -r vllm-qaic/requirements-qwen3-asr-aot.txt

python -m pip install \
  --editable ./efficient-transformers

# Qwen3-ASR requires this final Transformers version.
python -m pip install \
  --force-reinstall \
  --no-deps \
  transformers==5.14.1 \
  numpy==1.26.4 \
  scipy==1.14.1 \
  scikit-learn==1.5.2
```

## 2. vLLM-QAIC PR

```bash
cd vllm-qaic

./scripts/install.sh aot

python -m pip install \
  --editable . \
  --no-build-isolation

cd ..
```

`install.sh` can replace QEfficient and downgrade Transformers. Restore both:

```bash
python -m pip install \
  --editable ./efficient-transformers \
  --no-deps

python -m pip install \
  --force-reinstall \
  --no-deps \
  transformers==5.14.1 \
  numpy==1.26.4 \
  scipy==1.14.1 \
  scikit-learn==1.5.2
```

Verify:

```bash
python - <<'PY'
import QEfficient
import transformers
import vllm

print("QEfficient:", QEfficient.__file__)
print("Transformers:", transformers.__version__)
print("vLLM:", vllm.__file__)

assert transformers.__version__ == "5.14.1"
assert "efficient-transformers" in QEfficient.__file__
assert "vllm-qaic" in vllm.__file__
PY
```

## 3. Run Qwen3-ASR

```bash
export QAIC_VISIBLE_DEVICES=<available-ai100-device-id>
export VLLM_QAIC_QPC_PATH=<qpc-directory>
export VLLM_QAIC_EFFICIENT_TRANSFORMERS=$PWD/efficient-transformers

python vllm-qaic/examples/qaic_qwen3_asr.py \
  <audio-file> \
  --device-ids "$QAIC_VISIBLE_DEVICES" \
  --qpc-path "$VLLM_QAIC_QPC_PATH" \
  --efficient-transformers "$VLLM_QAIC_EFFICIENT_TRANSFORMERS" \
  --prefill-seq-len 512 \
  --encoder-ctx-len 3000 \
  --max-model-len 512 \
  --max-tokens 128
```
