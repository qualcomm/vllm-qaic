"""
Standalone QAIC validation for `_dequant_gather_kernel`.

Source under test:
vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_embedding.py
  - _dequant_gather_kernel   (@triton.jit)
  - _dequant_gather_triton   (launcher)

Fused embedding gather + INT unpack + dequant for a pack-quantized
VocabParallelEmbedding. Given token ids, INT32-packed weights and per-row (or
per-group) scales, it produces, for each requested id and column `col`:
    packed_idx = col // PACK_FACTOR
    shift      = (col % PACK_FACTOR) * NUM_BITS
    q          = ((packed[id, packed_idx] >> shift) & ((1<<NUM_BITS)-1))
                 - (1 << (NUM_BITS-1))
    out        = q.float() * scale
This test exercises the CHANNEL-quantized path (num_groups == 1, GROUP_SIZE 0),
NUM_BITS=4, PACK_FACTOR=8.

Packing note: to avoid signed-vs-unsigned right-shift ambiguity, we constrain
the highest-order nibble of each packed int32 to < 8 so bit 31 is never set
(the packed value stays a positive int32). The pytorch_ref packs/unpacks with
identical arithmetic. Comparison is float assert_close.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_dequant_gather_kernel.txt")
KERNEL_FILE_PATH = (
    "vllm/model_executor/layers/quantization/compressed_tensors/"
    "compressed_tensors_embedding.py"
)
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_embedding import (  # noqa: E402,E501
    _dequant_gather_triton,
)

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
VOCAB = 12
HIDDEN = 64
NUM_BITS = 4
PACK_FACTOR = 32 // NUM_BITS  # 8
IDS = torch.tensor([0, 3, 7, 3, 11, 5], dtype=torch.int64, device=DEVICE)


def _build_packed():
    """Build int32-packed 4-bit weights (channel scales) on CPU.

    Returns (q_unsigned [V,H] int64, weight_packed [V,H/PF] int32,
             weight_scale [V,1] float32) all on CPU.
    """
    # Unsigned nibbles in [0, 16); constrain the top nibble of each group < 8
    # so the packed int32 keeps bit 31 clear (stays positive).
    q = torch.randint(0, 16, (VOCAB, HIDDEN), dtype=torch.int64)
    top_cols = torch.arange(PACK_FACTOR - 1, HIDDEN, PACK_FACTOR)
    q[:, top_cols] = torch.randint(0, 8, (VOCAB, top_cols.numel()), dtype=torch.int64)

    packed_cols = HIDDEN // PACK_FACTOR
    packed = torch.zeros(VOCAB, packed_cols, dtype=torch.int64)
    for j in range(PACK_FACTOR):
        cols = torch.arange(j, HIDDEN, PACK_FACTOR)
        packed |= q[:, cols] << (j * NUM_BITS)
    packed_i32 = packed.to(torch.int32)

    scale = torch.rand(VOCAB, 1, dtype=torch.float32) * 0.1 + 0.01
    return q, packed_i32, scale


_Q_UNSIGNED, _PACKED_CPU, _SCALE_CPU = _build_packed()
WEIGHT_PACKED = _PACKED_CPU.to(DEVICE)
WEIGHT_SCALE = _SCALE_CPU.to(DEVICE)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def _bench(fn, warmup=3, iters=10):
    """Device-synced wall-clock benchmark. Returns latency stats (ms)."""
    import time

    import numpy as np

    def _sync():
        try:
            torch.qaic.synchronize()
        except Exception:
            pass

    for _ in range(warmup):
        fn()
    _sync()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - start) * 1000.0)
    arr = np.array(times)
    return {
        "avg_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def pytorch_ref(ids):
    """Pure PyTorch gather + dequant using the pre-packed unsigned nibbles."""
    ids_cpu = ids.cpu().to(torch.int64)
    q = _Q_UNSIGNED[ids_cpu].to(torch.float32)  # [n, hidden] unsigned nibbles
    q = q - (1 << (NUM_BITS - 1))  # signed offset
    scale = _SCALE_CPU[ids_cpu]  # [n, 1] channel scale
    return q * scale


def kernel_impl(ids):
    return _dequant_gather_triton(
        ids, WEIGHT_PACKED, WEIGHT_SCALE, HIDDEN, NUM_BITS
    )


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(IDS)
        kernel_out = kernel_impl(IDS)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        torch.testing.assert_close(
            kernel_cpu.float(), ref_cpu.float(), rtol=1e-3, atol=1e-3
        )

        diff = (kernel_cpu.float() - ref_cpu.float()).abs()
        denom = ref_cpu.float().abs().clamp_min(1e-6)
        stats = {
            "input_shape": tuple(IDS.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(IDS.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(IDS.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": (diff / denom).max().item(),
        }

        pt_stats = _bench(lambda: pytorch_ref(IDS))
        kern_stats = _bench(lambda: kernel_impl(IDS))
        speedup = (kern_stats["avg_ms"] / pt_stats["avg_ms"]
                   if pt_stats["avg_ms"] > 0 else float("nan"))
        stats["pytorch_latency_ms"] = pt_stats
        stats["kernel_latency_ms"] = kern_stats
        stats["speedup_kernel_over_pytorch"] = speedup

        status = "SUCCESS"
        print("SUCCESS")
        print(stats)
        print(f"Speedup (Kernel/PyTorch): {speedup:.4f}x")
    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE")
        print(error_text)
    finally:
        lines = [
            f"{timestamp}\n",
            "Kernel: _dequant_gather_kernel (channel, 4-bit)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- ids shape: {stats['input_shape']}, ids={IDS.cpu().tolist()}\n")
            lines.append(f"- vocab={VOCAB}, hidden={HIDDEN}, num_bits={NUM_BITS}\n")
            lines.append(f"- weight_packed shape: {tuple(WEIGHT_PACKED.shape)} (int32)\n")
            lines.append(f"- weight_scale shape: {tuple(WEIGHT_SCALE.shape)} (channel)\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- out dtype: {stats['out_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            lines.append(f"- max_rel_err: {stats['rel_err']}\n")
            if "pytorch_latency_ms" in stats:
                lines.append("Timing:\n")
                lines.append(
                    f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                    f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                    f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n")
                lines.append(
                    f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                    f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                    f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n")
                lines.append(
                    f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n")
        else:
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
