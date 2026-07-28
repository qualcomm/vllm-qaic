"""
Standalone QAIC validation for `_prepare_pos_seq_lens_kernel`.

Source under test:
vllm/v1/worker/gpu/input_batch.py
  - _prepare_pos_seq_lens_kernel  (launched via `prepare_pos_seq_lens`)

The kernel is launched with a grid of (num_reqs + 1,). For each real request
program it computes:
  - seq_len[req_id]  = num_computed_tokens[req_state_idx] + query_len
  - positions[query_start + i] = num_computed_tokens + i   (for i in query_len)
The extra final program (req_id == num_reqs) zero-pads the unused tail of the
seq_lens buffer (indices [num_reqs, max_num_reqs)) for full CUDA graph capture.

Integer-exact validation of both mutated buffers (positions and seq_lens,
including the zero-padded tail).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_prepare_pos_seq_lens_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/input_batch.py"

DEVICE = "qaic"

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from vllm.v1.worker.gpu.input_batch import prepare_pos_seq_lens

torch.manual_seed(42)

# ---- Global shared inputs -------------------------------------------------
MAX_NUM_REQS = 6
NUM_REQS = 4

IDX_MAPPING = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device=DEVICE)
QUERY_LENS = [4, 3, 5, 2]
_qsl = [0]
for q in QUERY_LENS:
    _qsl.append(_qsl[-1] + q)
QUERY_START_LOC = torch.tensor(_qsl, dtype=torch.int32, device=DEVICE)
TOTAL_TOKENS = _qsl[-1]

NUM_COMPUTED_TOKENS = torch.tensor(
    [3, 7, 0, 11, 0, 0], dtype=torch.int32, device=DEVICE
)

# Positions is int64 in vLLM (InputBuffers.positions). seq_lens is int32,
# sized max_num_reqs so the pad program has slots to zero.
POS = torch.full((TOTAL_TOKENS,), -1, dtype=torch.int64, device=DEVICE)
SEQ_LENS = torch.full((MAX_NUM_REQS,), -1, dtype=torch.int32, device=DEVICE)


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


def pytorch_ref(idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens):
    pos = pos.cpu().clone()
    seq_lens = seq_lens.cpu().clone()
    idx_mapping = idx_mapping.cpu()
    qsl = query_start_loc.cpu()
    nct = num_computed_tokens.cpu()
    num_reqs = idx_mapping.shape[0]
    max_num_reqs = seq_lens.shape[0]
    for b in range(num_reqs):
        rs = int(idx_mapping[b].item())
        nc = int(nct[rs].item())
        qs = int(qsl[b].item())
        qe = int(qsl[b + 1].item())
        ql = qe - qs
        seq_lens[b] = nc + ql
        pos[qs:qs + ql] = torch.arange(nc, nc + ql, dtype=pos.dtype)
    # Pad program: zero out [num_reqs, max_num_reqs)
    seq_lens[num_reqs:max_num_reqs] = 0
    return pos, seq_lens


def kernel_impl(idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens):
    pos = pos.clone()
    seq_lens = seq_lens.clone()
    prepare_pos_seq_lens(
        idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens
    )
    return pos, seq_lens


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_pos, ref_sl = pytorch_ref(
            IDX_MAPPING, QUERY_START_LOC, NUM_COMPUTED_TOKENS, POS, SEQ_LENS
        )
        k_pos, k_sl = kernel_impl(
            IDX_MAPPING, QUERY_START_LOC, NUM_COMPUTED_TOKENS, POS, SEQ_LENS
        )
        k_pos = k_pos.cpu()
        k_sl = k_sl.cpu()

        mism_pos = int((k_pos != ref_pos).sum().item())
        mism_sl = int((k_sl != ref_sl).sum().item())
        assert mism_pos == 0, f"positions mismatch count={mism_pos}"
        assert mism_sl == 0, f"seq_lens mismatch count={mism_sl}"

        stats = {
            "pos_shape": tuple(POS.shape),
            "seq_lens_shape": tuple(SEQ_LENS.shape),
            "pos_dtype": str(POS.dtype),
            "seq_lens_dtype": str(SEQ_LENS.dtype),
            "device": str(POS.device),
            "mismatch_pos": mism_pos,
            "mismatch_seq_lens": mism_sl,
            "max_abs_diff": 0,
            "grid": f"({NUM_REQS + 1},)",
        }
        pt_stats = _bench(lambda: pytorch_ref(
            IDX_MAPPING, QUERY_START_LOC, NUM_COMPUTED_TOKENS, POS, SEQ_LENS
        ))
        kern_stats = _bench(lambda: kernel_impl(
            IDX_MAPPING, QUERY_START_LOC, NUM_COMPUTED_TOKENS, POS, SEQ_LENS
        ))
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
            "Kernel: _prepare_pos_seq_lens_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append(f"positions shape: {stats['pos_shape']} dtype {stats['pos_dtype']}\n")
            lines.append(f"seq_lens shape: {stats['seq_lens_shape']} dtype {stats['seq_lens_dtype']}\n")
            lines.append(f"device: {stats['device']}\n")
            lines.append(f"grid: {stats['grid']}\n")
            lines.append(f"mismatch(pos): {stats['mismatch_pos']}\n")
            lines.append(f"mismatch(seq_lens): {stats['mismatch_seq_lens']}\n")
            lines.append(f"max_abs_diff: {stats['max_abs_diff']}\n")
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
