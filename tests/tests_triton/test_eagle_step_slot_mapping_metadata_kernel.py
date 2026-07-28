"""
Standalone QAIC validation for `eagle_step_slot_mapping_metadata_kernel`.

Source under test:
vllm/v1/spec_decode/utils.py
  - eagle_step_slot_mapping_metadata_kernel
    (launched via `eagle_step_update_slot_mapping_and_metadata`)

Fused EAGLE autoregressive step. Launched with input_batch_size threads. Threads
with req_idx >= batch_size are cudagraph padding: they only write PAD_ID to
out_slot_mapping. For real requests:
  new_position    = position + 1
  exceeds_max     = new_position >= max_model_len
  clamped_pos     = 0 if exceeds_max else new_position
  block_number    = min(clamped_pos // block_size, n_blocks_per_req - 1)
  block_id        = block_table[req, block_number]
  slot_id         = block_id * block_size + clamped_pos % block_size
                    (PAD_ID if exceeds_max)
  new_seq_len     = min(1 if exceeds_max else seq_len + 1, max_model_len)
Writes out_clamped_positions, out_slot_mapping, and updates seq_lens in place.

Config tested: batch_size=4 real requests + 2 cudagraph padding slots
(input_batch_size=6), block_size=8, max_model_len=64, includes one request whose
new position exceeds max_model_len. We call the python launcher. Integer outputs
-> EXACT (torch.equal) comparison. Reference: pure PyTorch replication.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_eagle_step_slot_mapping_metadata_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/spec_decode/utils.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.spec_decode.utils import (  # noqa: E402
    PADDING_SLOT_ID,
    eagle_step_update_slot_mapping_and_metadata,
)

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
BATCH_SIZE = 4
INPUT_BATCH_SIZE = 6  # includes 2 cudagraph padding slots
BLOCK_SIZE = 8
MAX_MODEL_LEN = 64
N_BLOCKS_PER_REQ = 8

# One position (63) will exceed max_model_len after +1.
POSITIONS = torch.tensor([5, 15, 63, 30], dtype=torch.int32, device=DEVICE)
BLOCK_TABLE = torch.randint(
    0, 100, (BATCH_SIZE, N_BLOCKS_PER_REQ), dtype=torch.int32, device=DEVICE
)
SEQ_LENS = torch.tensor([6, 16, 64, 31], dtype=torch.int32, device=DEVICE)


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
    positions = POSITIONS.cpu().tolist()
    bt = BLOCK_TABLE.cpu()
    seq_lens = SEQ_LENS.cpu().tolist()

    out_clamped = torch.zeros(BATCH_SIZE, dtype=torch.int32)
    out_slot = torch.zeros(INPUT_BATCH_SIZE, dtype=torch.int32)
    new_seq_lens = torch.zeros(BATCH_SIZE, dtype=torch.int32)

    for req in range(INPUT_BATCH_SIZE):
        if req >= BATCH_SIZE:
            out_slot[req] = PADDING_SLOT_ID
            continue
        new_position = positions[req] + 1
        exceeds_max = new_position >= MAX_MODEL_LEN
        clamped = 0 if exceeds_max else new_position
        block_number = min(clamped // BLOCK_SIZE, N_BLOCKS_PER_REQ - 1)
        block_id = int(bt[req, block_number].item())
        slot_id = block_id * BLOCK_SIZE + clamped % BLOCK_SIZE
        slot_id = PADDING_SLOT_ID if exceeds_max else slot_id

        new_sl = 1 if exceeds_max else seq_lens[req] + 1
        new_sl = min(new_sl, MAX_MODEL_LEN)

        out_clamped[req] = clamped
        out_slot[req] = slot_id
        new_seq_lens[req] = new_sl
    return out_clamped, out_slot, new_seq_lens


def kernel_impl():
    # seq_lens is updated in place; work on a fresh clone each call.
    seq_lens = SEQ_LENS.clone()
    out_clamped = torch.zeros(BATCH_SIZE, dtype=torch.int32, device=DEVICE)
    out_slot = torch.zeros(INPUT_BATCH_SIZE, dtype=torch.int32, device=DEVICE)
    eagle_step_update_slot_mapping_and_metadata(
        POSITIONS,
        BLOCK_TABLE,
        seq_lens,
        BLOCK_SIZE,
        MAX_MODEL_LEN,
        out_clamped,
        out_slot,
        input_batch_size=INPUT_BATCH_SIZE,
    )
    return out_clamped, out_slot, seq_lens


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref()
        kern = kernel_impl()
        names = ["clamped_positions", "slot_mapping", "seq_lens"]
        for name, r, k in zip(names, ref, kern):
            kc = k.cpu().to(r.dtype)
            assert torch.equal(kc, r), (
                f"{name} mismatch: ref={r.tolist()} kern={kc.tolist()}"
            )

        stats = {
            "input_shape": tuple(POSITIONS.shape),
            "output_shape": tuple(kern[1].shape),
            "in_dtype": str(POSITIONS.dtype),
            "out_dtype": str(kern[1].dtype),
            "device": str(POSITIONS.device),
            "max_abs_diff": 0,
            "mean_abs_diff": 0,
        }
        pt_stats = _bench(pytorch_ref)
        kern_stats = _bench(kernel_impl)
        speedup = (
            kern_stats["avg_ms"] / pt_stats["avg_ms"]
            if pt_stats["avg_ms"] > 0
            else float("nan")
        )
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
            "Kernel: eagle_step_slot_mapping_metadata_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- input shape: {stats['input_shape']}\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- out dtype: {stats['out_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            lines.append("- relative_error: 0.0 (exact integer match)\n")
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
