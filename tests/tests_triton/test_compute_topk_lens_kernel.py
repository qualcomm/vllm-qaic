"""
Standalone QAIC validation for `_compute_topk_lens_kernel`.

Source under test:
vllm/models/deepseek_v4/amd/rocm.py
  - _compute_topk_lens_kernel

Counts the number of valid (non-negative) top-k indices per token, producing a
per-token `topk_lens`. Padding tokens (is_valid_token == 0) are forced to 0.

    count = (topk_indices >= 0).sum(dim=-1)
    topk_lens = where(is_valid_token, count, 0)

Integer metadata kernel -> EXACT-equality comparison. The kernel is launched
directly (grid = (num_tokens,)).
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.models.deepseek_v4.amd.rocm import _compute_topk_lens_kernel

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_compute_topk_lens_kernel.txt")
KERNEL_FILE_PATH = "vllm/models/deepseek_v4/amd/rocm.py"

DEVICE = "qaic"
torch.manual_seed(42)

NUM_TOKENS = 6
TOP_K = 16
# Valid entries (>=0) interspersed with -1 sentinels.
TOPK_INDICES = torch.randint(
    -2, 40, (NUM_TOKENS, TOP_K), dtype=torch.int32, device=DEVICE
).clamp(min=-1)
IS_VALID_TOKEN = torch.tensor(
    [1, 1, 0, 1, 0, 1], dtype=torch.int32, device=DEVICE
)


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


def pytorch_ref(topk_indices, is_valid_token):
    topk_indices = topk_indices.cpu()
    is_valid_token = is_valid_token.cpu()
    count = (topk_indices >= 0).sum(dim=-1).to(torch.int32)
    lens = torch.where(is_valid_token != 0, count, torch.zeros_like(count))
    return lens.to(torch.int32)


def kernel_impl(topk_indices, is_valid_token):
    num_tokens = topk_indices.shape[0]
    topk = topk_indices.shape[1]
    topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)
    _compute_topk_lens_kernel[(num_tokens,)](
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        topk,
        is_valid_token,
        TRITON_BLOCK_SIZE=1024,
    )
    return topk_lens


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref(TOPK_INDICES, IS_VALID_TOKEN)
        out = kernel_impl(TOPK_INDICES, IS_VALID_TOKEN).cpu()

        mismatch = int((out != ref).sum().item())
        assert mismatch == 0, f"topk_lens mismatches={mismatch}"

        stats = {
            "input_shape": tuple(TOPK_INDICES.shape),
            "output_shape": tuple(out.shape),
            "in_dtype": str(TOPK_INDICES.dtype),
            "out_dtype": str(out.dtype),
            "device": DEVICE,
            "mismatch": mismatch,
            "ref": ref.tolist(),
        }
        pt_stats = _bench(lambda: pytorch_ref(TOPK_INDICES, IS_VALID_TOKEN))
        kern_stats = _bench(lambda: kernel_impl(TOPK_INDICES, IS_VALID_TOKEN))
        speedup = (
            kern_stats["avg_ms"] / pt_stats["avg_ms"]
            if pt_stats["avg_ms"] > 0
            else float("nan")
        )
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
            "Kernel: _compute_topk_lens_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- topk_indices shape: {stats['input_shape']} dtype {stats['in_dtype']}\n",
                f"- is_valid_token: {IS_VALID_TOKEN.cpu().tolist()}\n",
                f"- device: {stats['device']}\n\n",
                "Output (EXACT-equality comparison):\n",
                f"- topk_lens shape: {stats['output_shape']} dtype {stats['out_dtype']}\n",
                f"- topk_lens: {stats['ref']}\n",
                f"- mismatches: {stats['mismatch']}\n",
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
