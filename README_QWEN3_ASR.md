# Qwen3-ASR: Compile and Serve

QEfficient PR `1276` compiles Qwen3-ASR. vLLM-QAIC PR `146` loads the
generated QPC and serves it. The QPC must be compiled before serving.

## 1. Compile with QEfficient PR 1276

```bash
mkdir qwen3-asr
cd qwen3-asr

git clone https://github.com/quic/efficient-transformers.git efficient-transformers
cd efficient-transformers
git fetch origin pull/1276/head:qeff-pr-1276
git checkout qeff-pr-1276

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install -e .
python -m pip install --force-reinstall --no-deps transformers==5.14.1
```

Run the Qwen3-ASR compile script included in this PR:

```bash
python <QEFF_COMPILE_SCRIPT> \
  --model Qwen/Qwen3-ASR-0.6B-hf \
  --output-dir <QPC_OUTPUT_DIR>
```

The compile output must contain:

```text
<QPC_OUTPUT_DIR>/programqpc.bin
```

## 2. Serve with vLLM-QAIC PR 146

From the `qwen3-asr` directory:

```bash
git clone https://github.com/qualcomm/vllm-qaic.git vllm-qaic
cd vllm-qaic
git fetch origin pull/146/head:vllm-qaic-pr-146
git checkout vllm-qaic-pr-146

python -m pip install \
  -r requirements-qwen3-asr-aot.txt

./scripts/install.sh aot
python -m pip install -e . --no-build-isolation

# install.sh may downgrade Transformers; restore the Qwen version.
python -m pip install --force-reinstall --no-deps transformers==5.14.1
```

Set the compiled QPC and device:

```bash
export QAIC_VISIBLE_DEVICES=<AI100_DEVICE_ID>
export VLLM_QAIC_QPC_PATH=<QPC_OUTPUT_DIR>
export VLLM_QAIC_EFFICIENT_TRANSFORMERS=$PWD/../efficient-transformers
```

Run one audio file:

```bash
python examples/qaic_qwen3_asr.py \
  <AUDIO_FILE> \
  --device-ids "$QAIC_VISIBLE_DEVICES" \
  --qpc-path "$VLLM_QAIC_QPC_PATH" \
  --efficient-transformers "$VLLM_QAIC_EFFICIENT_TRANSFORMERS" \
  --prefill-seq-len 512 \
  --encoder-ctx-len 3000 \
  --max-model-len 512 \
  --max-tokens 128
```

Verify the runtime before serving:

```bash
python - <<'PY'
import transformers
print("Transformers:", transformers.__version__)
assert transformers.__version__ == "5.14.1"
PY
```

