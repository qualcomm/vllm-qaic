"""
Standalone QAIC validation for `_prepare_decode_inputs_kernel`.

Source under test:
vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
  - _prepare_decode_inputs_kernel  (launched via `prepare_decode_inputs`)

Prepares the draft-decode step inputs. Grid is (num_reqs + 1,). For each real
request (ADVANCE_DRAFT_POSITIONS=True):
  input_ids[req]  = draft_tokens[req, 0]
  positions[req]  = min(positions[req] + 1, max_model_len - 1)
  seq_lens[req]   = min(target_seq_lens[req] - num_rejected[req] + 1, max_model_len)
The final program (req_idx == num_reqs) fills query_start_loc for CUDA graphs:
  query_start_loc[j] = j if j < num_reqs else num_reqs   (j in [0, max_num_reqs])
and pads seq_lens[num_reqs:max_num_reqs] = 0.

Config tested: num_reqs=4, max_num_reqs=8, max_model_len=64, draft_tokens is
[num_reqs, W] (uses column 0 via stride), includes a request whose position is
at the clamp boundary. The python launcher requires an InputBuffers object, so
we launch the @triton.jit kernel directly. Integer buffers -> EXACT
(torch.equal) comparison over input_ids, positions, query_start_loc, seq_lens.
Reference: pure PyTorch replication.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_prepare_decode_inputs_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (  # noqa: E402
    _prepare_decode_inputs_kernel,
)

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
NUM_REQS = 4
MAX_NUM_REQS = 8
MAX_MODEL_LEN = 64
DRAFT_W = 3  # draft_tokens is [num_reqs, DRAFT_W]; kernel reads column 0
KERNEL_BLOCK = 1024
ADVANCE_DRAFT_POSITIONS = True

DRAFT_TOKENS = torch.tensor(
    [[111, 1, 2], [222, 3, 4], [333, 5, 6], [444, 7, 8]],
    dtype=torch.int32,
    device=DEVICE,
)
# One position (63) is at the clamp boundary: +1 clamps to max_model_len - 1.
POSITIONS_INIT = torch.tensor(
    [10, 20, 63, 40, 7, 7, 7, 7], dtype=torch.int64, device=DEVICE
)
INPUT_IDS_INIT = torch.full((MAX_NUM_REQS,), -9, dtype=torch.int32, device=DEVICE)
TARGET_SEQ_LENS = torch.tensor(
    [11, 21, 64, 41], dtype=torch.int32, device=DEVICE
)
NUM_REJECTED = torch.tensor([1, 0, 2, 3], dtype=torch.int32, device=DEVICE)


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
    draft = DRAFT_TOKENS.cpu()
    positions = POSITIONS_INIT.cpu().clone()
    input_ids = INPUT_IDS_INIT.cpu().clone()
    tsl = TARGET_SEQ_LENS.cpu().tolist()
    nrej = NUM_REJECTED.cpu().tolist()

    seq_lens = torch.zeros(MAX_NUM_REQS, dtype=torch.int32)
    query_start_loc = torch.zeros(MAX_NUM_REQS + 1, dtype=torch.int32)

    for req in range(NUM_REQS):
        input_ids[req] = int(draft[req, 0].item())
        positions[req] = min(int(positions[req].item()) + 1, MAX_MODEL_LEN - 1)
        seq_len = tsl[req] - nrej[req]
        seq_lens[req] = min(seq_len + 1, MAX_MODEL_LEN)

    # Final-program CUDA graph padding.
    for j in range(MAX_NUM_REQS + 1):
        query_start_loc[j] = j if j < NUM_REQS else NUM_REQS
    for j in range(NUM_REQS, MAX_NUM_REQS):
        seq_lens[j] = 0

    return input_ids, positions, query_start_loc, seq_lens


def kernel_impl():
    positions = POSITIONS_INIT.clone()
    input_ids = INPUT_IDS_INIT.clone()
    seq_lens = torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device=DEVICE)
    query_start_loc = torch.zeros(MAX_NUM_REQS + 1, dtype=torch.int32, device=DEVICE)

    _prepare_decode_inputs_kernel[(NUM_REQS + 1,)](
        DRAFT_TOKENS,
        DRAFT_TOKENS.stride(0),
        TARGET_SEQ_LENS,
        NUM_REJECTED,
        input_ids,
        positions,
        query_start_loc,
        seq_lens,
        MAX_MODEL_LEN,
        MAX_NUM_REQS,
        BLOCK_SIZE=KERNEL_BLOCK,
        ADVANCE_DRAFT_POSITIONS=ADVANCE_DRAFT_POSITIONS,
    )
    return input_ids, positions, query_start_loc, seq_lens


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref()
        kern = kernel_impl()
        names = ["input_ids", "positions", "query_start_loc", "seq_lens"]
        for name, r, k in zip(names, ref, kern):
            kc = k.cpu().to(r.dtype)
            assert torch.equal(kc, r), (
                f"{name} mismatch: ref={r.tolist()} kern={kc.tolist()}"
            )

        stats = {
            "input_shape": tuple(DRAFT_TOKENS.shape),
            "output_shape": tuple(kern[0].shape),
            "in_dtype": str(DRAFT_TOKENS.dtype),
            "out_dtype": str(kern[0].dtype),
            "device": str(DRAFT_TOKENS.device),
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
            "Kernel: _prepare_decode_inputs_kernel\n",
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
