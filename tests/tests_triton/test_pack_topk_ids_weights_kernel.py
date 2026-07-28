"""
Standalone QAIC validation for `_pack_topk_ids_weights_kernel`.

Source under test:
vllm/model_executor/layers/fused_moe/utils.py
  - _pack_topk_ids_weights_kernel  (bit-packs MoE routing for TRT-LLM)

Packs each token's top-k expert id and routing weight into a single int32 for
the TRT-LLM MoE path. Bit layout (from source):

    expert_id_shifted = expert_id << 16
    weight_bf16       = weight.to(bfloat16)
    weight_int16      = bitcast(weight_bf16, int16)
    weight_int32      = weight_int16.to(int32) & 0xFFFF
    packed            = expert_id_shifted | weight_int32

i.e. expert id occupies the high 16 bits and the raw bf16 weight bit-pattern
occupies the low 16 bits. Launched via `trtllm_moe_pack_topk_ids_weights`.

Reference: pure PyTorch replicating the exact bit-packing. Bit-exact compare.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.fused_moe.utils import (
    trtllm_moe_pack_topk_ids_weights,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_pack_topk_ids_weights_kernel.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fused_moe/utils.py"

DEVICE = "qaic"
torch.manual_seed(42)

NUM_TOKENS = 8
TOP_K = 2
NUM_EXPERTS = 4

TOPK_IDS = torch.randint(
    0, NUM_EXPERTS, (NUM_TOKENS, TOP_K), dtype=torch.int32, device=DEVICE
).contiguous()
TOPK_WEIGHTS = torch.rand(
    NUM_TOKENS, TOP_K, dtype=torch.float32, device=DEVICE
).contiguous()


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def _bench(fn, warmup=3, iters=10):
    """Device-synced wall-clock benchmark. Returns dict of latency stats (ms).

    Uses time.perf_counter with torch.qaic.synchronize() because
    torch.Event-based timing is broken on the QAIC backend in this env.
    """
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
    times.sort()
    arr = np.array(times)
    return {
        "avg_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def pytorch_ref(topk_ids, topk_weights):
    """Pure PyTorch replication of the exact int32 bit-packing.

    packed = (expert_id << 16) | (bf16_bits(weight) & 0xFFFF)
    """
    ids = topk_ids.cpu().to(torch.int32)
    weights = topk_weights.cpu().to(torch.float32)

    expert_id_shifted = ids.to(torch.int32) << 16

    weight_bf16 = weights.to(torch.bfloat16)
    # bitcast bf16 -> int16, then keep the low 16 bits.
    weight_int16 = weight_bf16.view(torch.int16)
    weight_int32 = weight_int16.to(torch.int32) & 0xFFFF

    packed = expert_id_shifted | weight_int32
    return packed.to(torch.int32)


def kernel_impl(topk_ids, topk_weights):
    return trtllm_moe_pack_topk_ids_weights(topk_ids, topk_weights)


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(TOPK_IDS, TOPK_WEIGHTS)
        kernel_out = kernel_impl(TOPK_IDS, TOPK_WEIGHTS)

        kernel_cpu = kernel_out.cpu().to(torch.int32)

        assert torch.equal(kernel_cpu, ref_out), "packed bits mismatch"

        stats = {
            "input_shape": (tuple(TOPK_IDS.shape), tuple(TOPK_WEIGHTS.shape)),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": (str(TOPK_IDS.dtype), str(TOPK_WEIGHTS.dtype)),
            "out_dtype": str(kernel_out.dtype),
            "device": str(TOPK_IDS.device),
            "max_abs_diff": 0,
            "mean_abs_diff": 0.0,
        }
        pt_stats = _bench(lambda: pytorch_ref(TOPK_IDS, TOPK_WEIGHTS))
        kern_stats = _bench(lambda: kernel_impl(TOPK_IDS, TOPK_WEIGHTS))
        speedup = kern_stats["avg_ms"] / pt_stats["avg_ms"] if pt_stats["avg_ms"] > 0 else float("nan")
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
            "Kernel: _pack_topk_ids_weights_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append(f"Input shapes (ids,weights): {stats['input_shape']}\n")
            lines.append(f"Input dtypes: {stats['in_dtype']}\n")
            lines.append(f"Output shape: {stats['output_shape']} ({stats['out_dtype']})\n")
            lines.append("Bit layout: (expert_id << 16) | (bf16_bits(weight) & 0xFFFF)\n")
            lines.append(f"Device: {stats['device']}\n")
            lines.append(f"Max abs diff: {stats['max_abs_diff']}\n")
            lines.append(f"Mean abs diff: {stats['mean_abs_diff']}\n")
            lines.append("Rel error: exact bit-pattern compare\n")
            if "pytorch_latency_ms" in stats:
                lines.append("Timing:\n")
                lines.append(f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                             f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                             f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                             f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n")
                lines.append(f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                             f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                             f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                             f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n")
                lines.append(f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n")
        else:
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    main()
