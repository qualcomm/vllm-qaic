"""
Standalone QAIC validation for `_get_lora_id` (FP8 MoE-LoRA device helper).

Source under test:
vllm/lora/ops/triton_ops/fused_moe_lora_fp8_op.py
  - _get_lora_id  (@triton.jit DEVICE HELPER, FP8-specific copy)

This device helper resolves the active LoRA adapter id for the current FP8
MoE-LoRA program. Two modes (constexpr `naive_block_assignment`):
  - naive: token_idx = pid_m // top_k_num ; lora_id = token_lora_mapping[token_idx]
  - sorted: lora_id = lora_ids[lora_idx]   (sorted-token lora-id list)

Because it is a device helper (no launcher), we wrap it in a tiny standalone
@triton.jit kernel that invokes the helper once per program (naive mode: one
program per output row) and stores the resolved lora id. Integer / index
output -> EXACT comparison against the pure-PyTorch lookup reference.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton
from vllm.lora.ops.triton_ops.fused_moe_lora_fp8_op import _get_lora_id

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_fp8_get_lora_id.txt")
KERNEL_FILE_PATH = "vllm/lora/ops/triton_ops/fused_moe_lora_fp8_op.py"

DEVICE = "qaic"
torch.manual_seed(42)

NUM_ROWS = 8      # number of programs (pid_m values)
TOP_K_NUM = 2
NUM_TOKENS = NUM_ROWS // TOP_K_NUM
MAX_LORAS = 3
NAIVE = True

# per-token lora mapping (naive mode source of truth)
TOKEN_LORA_MAPPING = torch.tensor(
    [0, 2, 1, 0], dtype=torch.int32, device=DEVICE
)
# sorted-token lora-id list (unused in naive mode, still passed)
LORA_IDS = torch.arange(NUM_ROWS, dtype=torch.int32, device=DEVICE) % MAX_LORAS


@triton.jit
def _wrap_get_lora_id(
    out_ptr,
    lora_ids,
    token_lora_mapping_ptr,
    top_k_num,
    naive_block_assignment: tl.constexpr,
):
    i = tl.program_id(0)
    lid = _get_lora_id(
        lora_ids,
        token_lora_mapping_ptr,
        i,          # lora_idx
        i,          # pid_m
        top_k_num,
        naive_block_assignment,
    )
    tl.store(out_ptr + i, lid)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def pytorch_ref(token_lora_mapping, lora_ids):
    tlm = token_lora_mapping.cpu()
    lids = lora_ids.cpu()
    out = torch.empty(NUM_ROWS, dtype=torch.int32)
    for i in range(NUM_ROWS):
        if NAIVE:
            token_idx = i // TOP_K_NUM
            out[i] = int(tlm[token_idx].item())
        else:
            out[i] = int(lids[i].item())
    return out


def kernel_impl(token_lora_mapping, lora_ids):
    out = torch.empty(NUM_ROWS, dtype=torch.int32, device=DEVICE)
    _wrap_get_lora_id[(NUM_ROWS,)](
        out,
        lora_ids,
        token_lora_mapping,
        TOP_K_NUM,
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
        ref = pytorch_ref(TOKEN_LORA_MAPPING, LORA_IDS)
        out = kernel_impl(TOKEN_LORA_MAPPING, LORA_IDS).cpu().to(torch.int32)
        assert torch.equal(out, ref), f"mismatch: {out.tolist()} vs {ref.tolist()}"
        stats = {
            "input_shape": tuple(TOKEN_LORA_MAPPING.shape),
            "output_shape": tuple(out.shape),
            "in_dtype": str(TOKEN_LORA_MAPPING.dtype),
            "out_dtype": str(out.dtype),
            "device": DEVICE,
            "max_abs_diff": int((out - ref).abs().max().item()),
            "mean_abs_diff": float((out - ref).abs().float().mean().item()),
        }
        pt_stats = _bench(lambda: pytorch_ref(TOKEN_LORA_MAPPING, LORA_IDS))
        kern_stats = _bench(lambda: kernel_impl(TOKEN_LORA_MAPPING, LORA_IDS))
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
            "Kernel: _get_lora_id (FP8 MoE-LoRA device helper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- token_lora_mapping: {TOKEN_LORA_MAPPING.cpu().tolist()}\n",
                f"- top_k_num={TOP_K_NUM}, naive={NAIVE}, num_rows={NUM_ROWS}\n",
                f"- device: {stats['device']}\n\n",
                "Output (integer/index EXACT comparison):\n",
                f"- lora_ids shape: {stats['output_shape']} dtype {stats['out_dtype']}\n",
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
