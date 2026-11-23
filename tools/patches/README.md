# DFlash2 Native Registry Prerequisite

DFlash and DFlash2 use the same QAIC speculative-decoding path (`method="dflash"`).
DFlash2 requires QEfficient's native DFlash2 support and a vLLM registry entry for
the checkpoint architecture `DFlash2DraftModel`.

For vLLM v0.23.0, apply `vllm-dflash2-registry.patch` to the installed vLLM package
in the environment used to run the example. From the vllm-qaic repository root:

```bash
VLLM_PACKAGE=$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("vllm").submodule_search_locations)))')
patch --dry-run --forward -d "$VLLM_PACKAGE" -p2 < tools/patches/vllm-dflash2-registry.patch
patch --forward -d "$VLLM_PACKAGE" -p2 < tools/patches/vllm-dflash2-registry.patch
```

Skip this patch when the installed registry already contains the entry. Recheck
after reinstalling vLLM. The alias shares DFlash's capability inspection; QAIC
loads the native DFlash2 graph through QEfficient. It does not add GPU execution
support for DFlash2. No runtime registration in `platform.py` is required.

## Standalone VLM Example

The existing example supports both DFlash and DFlash2. Without QPC paths, it
exports and compiles missing artifacts using the normal cache:

```bash
export VLLM_TARGET_DEVICE=qaic TRITON_CPU_BACKEND=1
python examples/qaic_spd_dflash_vlm.py \
  --model Qwen/Qwen3.8-27B --draft-model z-lab/Qwen3.8-27B-DFlash2 \
  --image /path/to/image.png --question "Describe this image."
```

To reuse compatible artifacts, add `--tlm-qpc "$TLM_QPC"`,
`--dlm-qpc "$DLM_QPC"`, and `--vision-qpc "$VISION_QPC"`. Their shapes and core
counts must match the example configuration: BS1, context 4096, prefill 128,
448-pixel image, and TS4 per component. Core counts default to 8 each; override
them with `--tlm-cores`, `--dlm-cores`, and `--vision-cores`. The script uses
devices 0-3 for language, 4-7 for draft, and 8-11 for vision.

Thinking is disabled by default, with `--enable-thinking` available explicitly.
The BS1 Qwen3.5-family target disables continuous-batching cache indexing at
export; draft and vision settings are unchanged. Existing QPC graphs are not
modified by this setting. DFlash2 uses argmax drafting and supports
non-disaggregated execution only in this integration.
