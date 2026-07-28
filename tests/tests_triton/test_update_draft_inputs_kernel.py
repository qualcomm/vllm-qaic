"""
Standalone QAIC validation for `_update_draft_inputs_kernel`.

Source under test:
vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
  - _update_draft_inputs_kernel  (launched via `update_draft_inputs`).

For each request the kernel:
  1. writes draft_tokens[req] into output_draft_tokens[req, step]
     (step = current_draft_step, read from a 0-d tensor);
  2. if step < num_speculative_steps - 1 (not the final step):
       - writes draft_tokens[req] into input_ids[req];
       - copies hidden_states[req, :] into next_input_hidden_states[req, :];
       - if ADVANCE_DRAFT_POSITIONS: positions[req] = min(pos+1, max_model_len-1),
         seq_lens[req] = min(seq_len+1, max_model_len).

Fully deterministic integer/float copies + clamped increments; NO RNG. We exercise
the non-final step (step=0, num_speculative_steps=3) so the hidden-state copy and
position/seq_len advance paths run, and compare every mutated buffer against a pure
PyTorch reference EXACTLY (integer buffers with torch.equal, hidden states with
assert_close).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_update_draft_inputs_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (  # noqa: E402
    update_draft_inputs,
)

torch.manual_seed(42)


class _InputBuffers:
    """Minimal stand-in exposing only the fields the kernel reads/writes."""

    def __init__(self, input_ids, positions, seq_lens):
        self.input_ids = input_ids
        self.positions = positions
        self.seq_lens = seq_lens


# ---- Global shared inputs (used by BOTH implementations) ----
NUM_REQS = 5
HIDDEN_SIZE = 256
NUM_SPEC_STEPS = 3
STEP = 0  # non-final step -> full path exercised
MAX_MODEL_LEN = 4096
ADVANCE = True

DRAFT_TOKENS = torch.randint(0, 32000, (NUM_REQS,), dtype=torch.int64, device=DEVICE)
CURRENT_DRAFT_STEP = torch.tensor(STEP, dtype=torch.int64, device=DEVICE)
HIDDEN_STATES = torch.randn(NUM_REQS, HIDDEN_SIZE, dtype=torch.float32, device=DEVICE)
# Buffers seeded with distinct values so untouched entries are detectable.
INIT_POSITIONS = torch.randint(0, 100, (NUM_REQS,), dtype=torch.int64, device=DEVICE)
INIT_SEQ_LENS = torch.randint(1, 200, (NUM_REQS,), dtype=torch.int64, device=DEVICE)
INIT_INPUT_IDS = torch.zeros(NUM_REQS, dtype=torch.int64, device=DEVICE)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def _bench(fn, warmup=3, iters=10):
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
    """Pure PyTorch reproduction. Returns all mutated buffers on CPU."""
    draft = DRAFT_TOKENS.cpu()
    hs = HIDDEN_STATES.cpu()
    step = int(CURRENT_DRAFT_STEP.cpu().item())

    out_draft = torch.zeros(NUM_REQS, NUM_SPEC_STEPS + 1, dtype=torch.int64)
    next_hs = torch.zeros(NUM_REQS, HIDDEN_SIZE, dtype=torch.float32)
    input_ids = INIT_INPUT_IDS.cpu().clone()
    positions = INIT_POSITIONS.cpu().clone()
    seq_lens = INIT_SEQ_LENS.cpu().clone()

    for r in range(NUM_REQS):
        out_draft[r, step] = draft[r]
        if step >= NUM_SPEC_STEPS - 1:
            continue
        input_ids[r] = draft[r]
        next_hs[r, :] = hs[r, :]
        if ADVANCE:
            positions[r] = min(int(positions[r].item()) + 1, MAX_MODEL_LEN - 1)
            seq_lens[r] = min(int(seq_lens[r].item()) + 1, MAX_MODEL_LEN)
    return out_draft, next_hs, input_ids, positions, seq_lens


def kernel_impl():
    """Kernel launch only. Allocates output buffers and invokes the launcher."""
    out_draft = torch.zeros(
        NUM_REQS, NUM_SPEC_STEPS + 1, dtype=torch.int64, device=DEVICE
    )
    next_hs = torch.zeros(NUM_REQS, HIDDEN_SIZE, dtype=torch.float32, device=DEVICE)
    input_buffers = _InputBuffers(
        INIT_INPUT_IDS.clone(),
        INIT_POSITIONS.clone(),
        INIT_SEQ_LENS.clone(),
    )
    update_draft_inputs(
        DRAFT_TOKENS,
        CURRENT_DRAFT_STEP,
        HIDDEN_STATES,
        out_draft,
        next_hs,
        input_buffers,
        NUM_REQS,
        MAX_MODEL_LEN,
        NUM_SPEC_STEPS,
        advance_draft_positions=ADVANCE,
    )
    return (
        out_draft,
        next_hs,
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.seq_lens,
    )


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref()
        kern = kernel_impl()

        ref_cpu = [t.cpu() for t in ref]
        kern_cpu = [t.cpu() for t in kern]
        (r_draft, r_hs, r_ids, r_pos, r_seq) = ref_cpu
        (k_draft, k_hs, k_ids, k_pos, k_seq) = kern_cpu

        assert torch.equal(k_draft, r_draft), "output_draft_tokens mismatch"
        assert torch.equal(k_ids, r_ids), "input_ids mismatch"
        assert torch.equal(k_pos, r_pos), "positions mismatch"
        assert torch.equal(k_seq, r_seq), "seq_lens mismatch"
        torch.testing.assert_close(k_hs.float(), r_hs.float(), rtol=1e-3, atol=1e-3)

        diff = (k_hs.float() - r_hs.float()).abs()
        stats = {
            "input_shape": tuple(HIDDEN_STATES.shape),
            "output_shape": tuple(k_hs.shape),
            "in_dtype": str(HIDDEN_STATES.dtype),
            "out_dtype": str(k_hs.dtype),
            "device": str(HIDDEN_STATES.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
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
            "Kernel: _update_draft_inputs_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- input shape (hidden_states): {stats['input_shape']}\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape (next hidden): {stats['output_shape']}\n")
            lines.append(f"- out dtype: {stats['out_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
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
