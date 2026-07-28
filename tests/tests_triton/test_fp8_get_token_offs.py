"""
Standalone QAIC validation for `_get_token_offs` (FP8 MoE-LoRA device helper).

Source under test:
vllm/lora/ops/triton_ops/fused_moe_lora_fp8_op.py
  - _get_token_offs  (@triton.jit DEVICE HELPER, FP8-specific copy)

Computes the flattened token offsets for the current FP8 MoE-LoRA tile. Two
modes (constexpr `naive_block_assignment`):
  - naive:  offs_token = where(offs == 0, pid_m, num_valid_tokens)
            (lane 0 = this tile's single token, all other lanes point past the
            valid range so they are masked out downstream)
  - sorted: offs_token_id = pid_m*BLOCK_SIZE_M + offs ;
            token_ind = stride_tl*lora_id + offs_token_id ;
            load(sorted_token_ids[token_ind], mask=token_ind<max_loras*stride_tl,
                 other=0)

Device helper (returns a BLOCK_SIZE_M vector) -> wrapped in a tiny standalone
@triton.jit kernel: one program per pid_m, stores the BLOCK_SIZE_M offsets row.
Integer / index output -> EXACT comparison. Exercises the naive path.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton
from vllm.lora.ops.triton_ops.fused_moe_lora_fp8_op import _get_token_offs

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_fp8_get_token_offs.txt")
KERNEL_FILE_PATH = "vllm/lora/ops/triton_ops/fused_moe_lora_fp8_op.py"

DEVICE = "qaic"
torch.manual_seed(42)

NUM_TILES = 4          # number of pid_m tiles / programs
BLOCK_SIZE_M = 8
NUM_VALID_TOKENS = 100
STRIDE_TL = 0
MAX_LORAS = 3
NAIVE = True

# placeholder sorted_token_ids table (unused in naive path but must be passed)
SORTED_TOKEN_IDS = torch.arange(
    NUM_TILES * BLOCK_SIZE_M, dtype=torch.int32, device=DEVICE
)


@triton.jit
def _wrap_get_token_offs(
    out_ptr,
    sorted_token_ids_ptr,
    stride_tl,
    max_loras,
    num_valid_tokens,
    naive_block_assignment: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = _get_token_offs(
        sorted_token_ids_ptr,
        0,          # lora_id (unused in naive path)
        pid_m,
        offs,
        stride_tl,
        max_loras,
        num_valid_tokens,
        naive_block_assignment,
        BLOCK_SIZE_M,
    )
    tl.store(out_ptr + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M), offs_token)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def pytorch_ref():
    out = torch.empty(NUM_TILES, BLOCK_SIZE_M, dtype=torch.int64)
    for pid_m in range(NUM_TILES):
        for j in range(BLOCK_SIZE_M):
            if NAIVE:
                out[pid_m, j] = pid_m if j == 0 else NUM_VALID_TOKENS
            else:
                out[pid_m, j] = pid_m * BLOCK_SIZE_M + j
    return out.reshape(-1)


def kernel_impl():
    out = torch.empty(
        NUM_TILES * BLOCK_SIZE_M, dtype=torch.int64, device=DEVICE
    )
    _wrap_get_token_offs[(NUM_TILES,)](
        out,
        SORTED_TOKEN_IDS,
        STRIDE_TL,
        MAX_LORAS,
        NUM_VALID_TOKENS,
        NAIVE,
        BLOCK_SIZE_M,
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
        ref = pytorch_ref()
        out = kernel_impl().cpu().to(torch.int64)
        assert torch.equal(out, ref), f"mismatch: {out.tolist()} vs {ref.tolist()}"
        stats = {
            "output_shape": tuple(out.shape),
            "out_dtype": str(out.dtype),
            "device": DEVICE,
            "max_abs_diff": int((out - ref).abs().max().item()),
            "mean_abs_diff": float((out - ref).abs().float().mean().item()),
        }
        pt_stats = _bench(lambda: pytorch_ref())
        kern_stats = _bench(lambda: kernel_impl())
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
            "Kernel: _get_token_offs (FP8 MoE-LoRA device helper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- num_tiles={NUM_TILES}, BLOCK_SIZE_M={BLOCK_SIZE_M}, "
                f"num_valid_tokens={NUM_VALID_TOKENS}, naive={NAIVE}\n",
                f"- device: {stats['device']}\n\n",
                "Output (integer/index EXACT comparison):\n",
                f"- token_offs shape: {stats['output_shape']} dtype {stats['out_dtype']}\n",
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
