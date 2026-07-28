"""
Standalone QAIC validation for `_min_p_kernel`.

Source under test:
vllm/v1/worker/gpu/sample/min_p.py
  - _min_p_kernel  (min-p logits filtering)

For each token row, compute max_logit = max(logits), threshold =
max_logit + log(min_p), then mask logits below threshold to -inf. Rows with
min_p == 0.0 are left unmodified. We validate the masked logits (including
-inf positions) against a pure PyTorch reference.

Reference: pure PyTorch max + threshold masking.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_min_p.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/sample/min_p.py"

# Global inputs
NUM_REQS = 8
VOCAB_SIZE = 4096
DEVICE = "qaic"

# The parent process (below, `python test_min_p.py`) must not touch the QAIC
# device at all: it only re-execs this file in a child so that an uncatchable
# Triton->Hexagon SIGABRT can still be recorded. Device tensors are therefore
# only created once we know we're the child, otherwise the parent claims the
# device's NSPs before the child gets a chance to.
_IS_CHILD = os.environ.get("MIN_P_CHILD") == "1"

if _IS_CHILD or __name__ != "__main__":
    import torch

    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm")
    )
    from vllm.v1.worker.gpu.sample.min_p import apply_min_p

    torch.manual_seed(42)
    LOGITS = torch.randn(NUM_REQS, VOCAB_SIZE, dtype=torch.float32, device=DEVICE)
    EXPANDED_IDX_MAPPING = torch.arange(NUM_REQS, dtype=torch.int32, device=DEVICE)
    # min_p per request (req_state indexed). 0.0 => no-op for that request.
    MIN_P = torch.tensor(
        [0.0, 0.1, 0.2, 0.5, 0.0, 0.3, 0.05, 0.8],
        dtype=torch.float32,
        device=DEVICE,
    )


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def pytorch_ref(logits, expanded_idx_mapping, min_p):
    """Pure PyTorch min-p filtering.

    Masks logits below max_logit + log(min_p) to -inf for each row whose
    mapped min_p is nonzero; rows with min_p == 0.0 are left unmodified.
    """
    out = logits.cpu().clone()
    expanded_idx_mapping = expanded_idx_mapping.cpu()
    min_p = min_p.cpu()
    num_tokens = out.shape[0]
    for token_idx in range(num_tokens):
        req_state_idx = int(expanded_idx_mapping[token_idx].item())
        mp = float(min_p[req_state_idx].item())
        if mp == 0.0:
            continue
        row = out[token_idx]
        threshold = row.max() + torch.log(torch.tensor(mp, dtype=torch.float32))
        row[row < threshold] = float("-inf")
    return out


def kernel_impl(logits, expanded_idx_mapping, min_p):
    logits = logits.clone()
    apply_min_p(logits, expanded_idx_mapping, min_p)
    return logits


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(LOGITS, EXPANDED_IDX_MAPPING, MIN_P)
        kernel_out = kernel_impl(LOGITS, EXPANDED_IDX_MAPPING, MIN_P)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()

        same_inf = torch.isinf(ref_cpu) == torch.isinf(kernel_cpu)
        finite_mask = ~torch.isinf(ref_cpu)

        torch.testing.assert_close(
            kernel_cpu[finite_mask], ref_cpu[finite_mask], rtol=1e-3, atol=1e-3
        )
        assert bool(same_inf.all()), "-inf mask mismatch"

        diff = (kernel_cpu[finite_mask] - ref_cpu[finite_mask]).abs()
        stats = {
            "input_shape": tuple(LOGITS.shape),
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(LOGITS.dtype),
            "device": str(LOGITS.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "inf_mask_match": bool(same_inf.all()),
            "grid": (f"({NUM_REQS},)", "BLOCK_SIZE=1024"),
        }
        status = "SUCCESS"
        print("SUCCESS")
        print(stats)

    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE")
        print(error_text)

    finally:
        lines = [
            f"{timestamp}\n",
            "Kernel: _min_p_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- logits shape: {stats['input_shape']}\n")
            lines.append(f"- dtype: {stats['dtype']}\n")
            lines.append(f"- device: {stats['device']}\n")
            lines.append(f"- min_p: {MIN_P.cpu().tolist()}\n\n")
            lines.append("Grid Configuration:\n")
            lines.append(f"- grid: {stats['grid'][0]}\n")
            lines.append(f"- block: {stats['grid'][1]}\n\n")
            lines.append("Output:\n")
            lines.append(f"- logits (masked) shape: {stats['output_shape']}\n")
            lines.append(f"- -inf mask match: {stats['inf_mask_match']}\n")
            lines.append(f"- max_abs_diff (finite): {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff (finite): {stats['mean_abs_diff']}\n")
        else:
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))

    return status


def _run_with_crash_guard():
    """Run main() in a child process so an uncatchable Triton->Hexagon
    compiler SIGABRT still gets recorded in the log by the parent."""
    import subprocess

    if os.environ.get("MIN_P_CHILD") == "1":
        sys.exit(0 if main() == "SUCCESS" else 1)

    env = dict(os.environ, MIN_P_CHILD="1")
    proc = subprocess.run([sys.executable, __file__], env=env)
    if proc.returncode < 0:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _log(
            f"{timestamp}\n"
            "Kernel: _min_p_kernel\n"
            f"Kernel file: {KERNEL_FILE_PATH}\n"
            f"Device target: QAIC (device='{DEVICE}')\n"
            "Status: FAILURE\n\n"
            "Error:\n"
            f"Child killed by signal (exit {proc.returncode}) during "
            "Triton->Hexagon compile/execution.\n"
            "\n------------------------------------\n\n"
        )
    sys.exit(proc.returncode if proc.returncode >= 0 else 1)


if __name__ == "__main__":
    _run_with_crash_guard()
