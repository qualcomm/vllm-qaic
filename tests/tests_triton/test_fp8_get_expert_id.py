"""
Standalone QAIC validation for `_get_expert_id` (FP8 MoE-LoRA device helper).

Source under test:
vllm/lora/ops/triton_ops/fused_moe_lora_fp8_op.py
  - _get_expert_id  (@triton.jit DEVICE HELPER, FP8-specific copy)

Resolves the MoE expert id for the current FP8 MoE-LoRA tile. Two modes
(constexpr `naive_block_assignment`):
  - naive: expert_id = expert_ids[pid_m]            (direct table)
  - sorted: ind = lora_id * stride_el + pid_m ;
            expert_id = load(expert_ids[ind], mask = ind < max_loras*stride_el,
                             other=-1)               (lora-indexed table)

Device helper -> wrapped in a tiny standalone @triton.jit kernel (one program
per tile) that calls the helper and stores the resolved expert id. Integer /
index output -> EXACT comparison. We exercise the naive (direct) path.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton
from vllm.lora.ops.triton_ops.fused_moe_lora_fp8_op import _get_expert_id

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_fp8_get_expert_id.txt")
KERNEL_FILE_PATH = "vllm/lora/ops/triton_ops/fused_moe_lora_fp8_op.py"

DEVICE = "qaic"
torch.manual_seed(42)

NUM_TILES = 6          # number of pid_m tiles / programs
NUM_EXPERTS = 4
MAX_LORAS = 3
STRIDE_EL = 0          # naive mode: stride unused
NAIVE = True

# direct expert-id table (naive mode source of truth), one entry per tile
EXPERT_IDS = torch.tensor(
    [0, 3, 1, 2, 0, 3], dtype=torch.int32, device=DEVICE
)


@triton.jit
def _wrap_get_expert_id(
    out_ptr,
    expert_ids_ptr,
    stride_el,
    max_loras,
    naive_block_assignment: tl.constexpr,
):
    i = tl.program_id(0)
    eid = _get_expert_id(
        expert_ids_ptr,
        0,          # lora_id (unused in naive path)
        i,          # pid_m
        stride_el,
        max_loras,
        naive_block_assignment,
    )
    tl.store(out_ptr + i, eid)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def pytorch_ref(expert_ids):
    eids = expert_ids.cpu()
    out = torch.empty(NUM_TILES, dtype=torch.int32)
    for i in range(NUM_TILES):
        if NAIVE:
            out[i] = int(eids[i].item())
        else:
            out[i] = int(eids[i].item())
    return out


def kernel_impl(expert_ids):
    out = torch.empty(NUM_TILES, dtype=torch.int32, device=DEVICE)
    _wrap_get_expert_id[(NUM_TILES,)](
        out,
        expert_ids,
        STRIDE_EL,
        MAX_LORAS,
        NAIVE,
    )
    return out


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


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref(EXPERT_IDS)
        out = kernel_impl(EXPERT_IDS).cpu().to(torch.int32)
        assert torch.equal(out, ref), f"mismatch: {out.tolist()} vs {ref.tolist()}"
        stats = {
            "input_shape": tuple(EXPERT_IDS.shape),
            "output_shape": tuple(out.shape),
            "in_dtype": str(EXPERT_IDS.dtype),
            "out_dtype": str(out.dtype),
            "device": DEVICE,
            "max_abs_diff": int((out - ref).abs().max().item()),
            "mean_abs_diff": float((out - ref).abs().float().mean().item()),
        }
        pt_stats = _bench(lambda: pytorch_ref(EXPERT_IDS))
        kern_stats = _bench(lambda: kernel_impl(EXPERT_IDS))
        speedup = kern_stats["avg_ms"] / pt_stats["avg_ms"] if pt_stats["avg_ms"] > 0 else float("nan")
        stats["pytorch_latency_ms"] = pt_stats
        stats["kernel_latency_ms"] = kern_stats
        stats["speedup_kernel_over_pytorch"] = speedup
        status = "SUCCESS"
        print("SUCCESS", stats)
        print(f"Speedup (Kernel/PyTorch): {speedup:.4f}x")
    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE\n" + error_text)
    finally:
        lines = [
            f"{timestamp}\n",
            "Kernel: _get_expert_id (FP8 MoE-LoRA device helper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- expert_ids: {EXPERT_IDS.cpu().tolist()}\n",
                f"- num_tiles={NUM_TILES}, max_loras={MAX_LORAS}, naive={NAIVE}\n",
                f"- device: {stats['device']}\n\n",
                "Output (integer/index EXACT comparison):\n",
                f"- expert_ids shape: {stats['output_shape']} dtype {stats['out_dtype']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
            ]
            if "pytorch_latency_ms" in stats:
                lines += [
                    "Timing:\n",
                    f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                    f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                    f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n",
                    f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                    f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                    f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n",
                    f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n",
                ]
        else:
            lines += ["Error:\n", error_text + "\n"]
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
