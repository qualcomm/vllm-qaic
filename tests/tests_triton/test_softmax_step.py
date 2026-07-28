"""
Standalone QAIC validation for the `softmax_step` @triton.jit device helper.

Source under test:
vllm/v1/attention/ops/triton_attention_helpers.py
  - softmax_step(S, M, L)

One online-softmax update step. Exact source recurrence:
    m_j   = maximum(M, max(S, axis=1))          # running max, per row
    m_j   = where(m_j > -inf, m_j, 0.0)         # guard fully-masked rows
    P     = exp(S - m_j[:, None])               # exponentiated probs
    l_j   = sum(P, axis=1)                       # this-tile sum
    alpha = exp(M - m_j)                         # rescale factor for old acc
    L_new = L * alpha + l_j                      # updated running sum
    return (m_j, L_new, P, alpha)               # == (M_new, L_new, P, alpha)

Shapes: S [BLOCK_M, TILE]; M, L [BLOCK_M]. Returns M_new [BLOCK_M],
L_new [BLOCK_M], P [BLOCK_M, TILE], alpha [BLOCK_M]. The caller is
responsible for rescaling its accumulator by alpha[:, None] outside.

We seed M/L from a prior tile so the rescale path (alpha != 1) is exercised.

Reference: pure PyTorch replication of the exact recurrence and ordering.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import softmax_step

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_softmax_step.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_attention_helpers.py"

DEVICE = "qaic"
BLOCK_M = 8
TILE = 16

torch.manual_seed(42)
S_IN = torch.randn(BLOCK_M, TILE, dtype=torch.float32, device=DEVICE) * 2.0
# Prior-tile running max / sum (from a hypothetical previous step) so that
# alpha = exp(M - m_j) is genuinely != 1 for several rows.
M_IN = torch.randn(BLOCK_M, dtype=torch.float32, device=DEVICE)
L_IN = torch.rand(BLOCK_M, dtype=torch.float32, device=DEVICE) + 1.0


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


def pytorch_ref(S, M, L):
    """Pure PyTorch online-softmax step, matching source ordering exactly."""
    S = S.cpu()
    M = M.cpu()
    L = L.cpu()
    m_j = torch.maximum(M, S.max(dim=1).values)  # [BLOCK_M]
    m_j = torch.where(m_j > float("-inf"), m_j, torch.zeros_like(m_j))
    P = torch.exp(S - m_j.reshape(BLOCK_M, 1))  # [BLOCK_M, TILE]
    l_j = P.sum(dim=1)  # [BLOCK_M]
    alpha = torch.exp(M - m_j)  # [BLOCK_M]
    L_new = L * alpha + l_j  # [BLOCK_M]
    return m_j, L_new, P, alpha


@triton.jit
def _softmax_step_launcher(
    s_ptr,
    m_ptr,
    l_ptr,
    m_new_ptr,
    l_new_ptr,
    p_ptr,
    alpha_ptr,
    BLOCK_M: tl.constexpr,
    TILE: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, TILE)
    idx = rows[:, None] * TILE + cols[None, :]
    S = tl.load(s_ptr + idx)  # [BLOCK_M, TILE]
    M = tl.load(m_ptr + rows)  # [BLOCK_M]
    L = tl.load(l_ptr + rows)  # [BLOCK_M]
    m_new, l_new, P, alpha = softmax_step(S, M, L)
    tl.store(m_new_ptr + rows, m_new)
    tl.store(l_new_ptr + rows, l_new)
    tl.store(alpha_ptr + rows, alpha)
    tl.store(p_ptr + idx, P)


def kernel_impl(S, M, L):
    m_new = torch.empty(BLOCK_M, dtype=torch.float32, device=DEVICE)
    l_new = torch.empty(BLOCK_M, dtype=torch.float32, device=DEVICE)
    alpha = torch.empty(BLOCK_M, dtype=torch.float32, device=DEVICE)
    P = torch.empty(BLOCK_M, TILE, dtype=torch.float32, device=DEVICE)
    _softmax_step_launcher[(1,)](
        S.reshape(-1),
        M,
        L,
        m_new,
        l_new,
        P.reshape(-1),
        alpha,
        BLOCK_M=BLOCK_M,
        TILE=TILE,
    )
    return m_new, l_new, P, alpha


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_m, ref_l, ref_p, ref_a = pytorch_ref(S_IN, M_IN, L_IN)
        ker_m, ker_l, ker_p, ker_a = kernel_impl(S_IN, M_IN, L_IN)

        pairs = {
            "M_new": (ker_m.cpu(), ref_m),
            "L_new": (ker_l.cpu(), ref_l),
            "P": (ker_p.cpu(), ref_p),
            "alpha": (ker_a.cpu(), ref_a),
        }
        max_diff = 0.0
        mean_diffs = []
        for name, (k, r) in pairs.items():
            torch.testing.assert_close(k, r, rtol=1e-3, atol=1e-3)
            d = (k - r).abs()
            max_diff = max(max_diff, d.max().item())
            mean_diffs.append(d.mean().item())

        stats = {
            "input_shape": tuple(S_IN.shape),
            "output_shapes": {
                "M_new": (BLOCK_M,),
                "L_new": (BLOCK_M,),
                "P": (BLOCK_M, TILE),
                "alpha": (BLOCK_M,),
            },
            "in_dtype": str(S_IN.dtype),
            "out_dtype": "torch.float32",
            "device": str(S_IN.device),
            "max_abs_diff": max_diff,
            "mean_abs_diff": sum(mean_diffs) / len(mean_diffs),
        }
        pt_stats = _bench(lambda: pytorch_ref(S_IN, M_IN, L_IN))
        kern_stats = _bench(lambda: kernel_impl(S_IN, M_IN, L_IN))
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
            "Kernel: softmax_step (device helper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- S shape: {stats['input_shape']}\n",
                f"- M shape: ({BLOCK_M},), L shape: ({BLOCK_M},)\n",
                f"- in dtype: {stats['in_dtype']}, device: {stats['device']}\n\n",
                "Outputs (M_new, L_new, P, alpha):\n",
                f"- shapes: {stats['output_shapes']}\n",
                f"- out dtype: {stats['out_dtype']}\n",
                f"- max_abs_diff (over all outputs): {stats['max_abs_diff']}\n",
                f"- mean_abs_diff (avg over outputs): {stats['mean_abs_diff']}\n",
            ]
            if "pytorch_latency_ms" in stats:
                lines.append("Timing:\n")
                lines.append(
                    f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                    f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                    f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n"
                )
                lines.append(
                    f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                    f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                    f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n"
                )
                lines.append(
                    f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n"
                )
        else:
            lines += ["Error:\n", error_text + "\n"]
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
