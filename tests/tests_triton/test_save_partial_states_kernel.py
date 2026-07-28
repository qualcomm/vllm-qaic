"""
Standalone QAIC validation for `_save_partial_states_kernel`.

Source under test:
vllm/models/deepseek_v4/common/ops/save_partial_states.py
  - _save_partial_states_kernel  (write packed [kv_state, score+ape] partial
    states into the DeepseekV4 compressor paged state cache). Launched via the
    public wrapper `save_partial_states`.

Exact source semantics (one program per token; slot_id == -1 tokens skipped):
    block_idx    = slot_id // block_size
    pos_in_block = slot_id %  block_size
    cache[block_idx, pos_in_block, 0:HEAD_SIZE]                    = kv[token]
    cache[block_idx, pos_in_block, STATE_WIDTH:STATE_WIDTH+HEAD]   =
        score[token] + ape[position % COMPRESS_RATIO]
The last cache dim packs [kv_state | score_state], each STATE_WIDTH wide.

Config tested: num_tokens=24 (some padded with slot=-1), head_size=64,
state_width=64 (cache last dim=128), block_size=16, compress_ratio=4, fp32.
Reference: pure PyTorch scatter of the same writes into a cloned cache; the
FULL cache buffer is compared (validates both the copy and the fused add).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_save_partial_states_kernel.txt")
KERNEL_FILE_PATH = "vllm/models/deepseek_v4/common/ops/save_partial_states.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.models.deepseek_v4.common.ops.save_partial_states import (  # noqa: E402
    save_partial_states,
)

torch.manual_seed(42)

NUM_TOKENS = 24
HEAD_SIZE = 64
STATE_WIDTH = 64
LAST_DIM = STATE_WIDTH + HEAD_SIZE  # 128
BLOCK_SIZE = 16
NUM_BLOCKS = 8
COMPRESS_RATIO = 4

KV = torch.randn(NUM_TOKENS, HEAD_SIZE, dtype=torch.float32, device=DEVICE)
SCORE = torch.randn(NUM_TOKENS, HEAD_SIZE, dtype=torch.float32, device=DEVICE)
APE = torch.randn(COMPRESS_RATIO, HEAD_SIZE, dtype=torch.float32, device=DEVICE)
POSITIONS = torch.randint(
    0, 100, (NUM_TOKENS,), dtype=torch.int32, device=DEVICE
)
# Distinct valid slots for most tokens; mark a few as padding (-1).
_slots = torch.randperm(NUM_BLOCKS * BLOCK_SIZE, device=DEVICE)[:NUM_TOKENS]
SLOT_MAPPING = _slots.to(torch.int64)
SLOT_MAPPING[3] = -1
SLOT_MAPPING[10] = -1
SLOT_MAPPING[17] = -1

STATE_CACHE_INIT = torch.zeros(
    NUM_BLOCKS, BLOCK_SIZE, LAST_DIM, dtype=torch.float32, device=DEVICE
)


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


def pytorch_ref():
    """Pure PyTorch scatter of the partial-state writes into a fresh cache."""
    cache = STATE_CACHE_INIT.clone()
    slots = SLOT_MAPPING.cpu()
    for tok in range(NUM_TOKENS):
        slot = int(slots[tok].item())
        if slot < 0:
            continue
        b = slot // BLOCK_SIZE
        p = slot % BLOCK_SIZE
        cache[b, p, 0:HEAD_SIZE] = KV[tok]
        ape_row = int(POSITIONS[tok].item()) % COMPRESS_RATIO
        cache[b, p, STATE_WIDTH:STATE_WIDTH + HEAD_SIZE] = SCORE[tok] + APE[ape_row]
    return cache


def kernel_impl():
    """Kernel launch only (mutates the cache in place; use a fresh clone)."""
    cache = STATE_CACHE_INIT.clone()
    save_partial_states(
        KV,
        SCORE,
        APE,
        POSITIONS,
        cache,
        SLOT_MAPPING,
        BLOCK_SIZE,
        STATE_WIDTH,
        COMPRESS_RATIO,
    )
    return cache


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref()
        kernel_out = kernel_impl()

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "input_shape": tuple(KV.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(KV.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(KV.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }

        pt_stats = _bench(pytorch_ref)
        kern_stats = _bench(kernel_impl)
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
            "Kernel: _save_partial_states_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- kv shape: {stats['input_shape']}\n",
                f"- in dtype: {stats['in_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- cache shape: {stats['output_shape']}\n",
                f"- out dtype: {stats['out_dtype']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
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
