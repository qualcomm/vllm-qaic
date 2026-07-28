"""
Standalone QAIC validation for `_triton_mrope_forward`.

Source under test:
vllm/model_executor/layers/rotary_embedding/mrope.py
  - _triton_mrope_forward  (Qwen2/2.5-VL multimodal RoPE (MRoPE) applied to q and
    k, in place). Launched via the public wrapper `triton_mrope`.

The kernel builds a per-token (cos, sin) row of length rotary_dim//2 by selecting
the T/H/W frequency sections from a cos/sin cache of shape
(3, num_tokens, head_dim//2) according to `mrope_section = [t, h, w]`
(sum == rotary_dim//2), then applies a NeoX-style (split-half) rotation:
    out1 = q1 * cos - q2 * sin
    out2 = q2 * cos + q1 * sin
The pointer arithmetic in the kernel assumes head_dim == rotary_dim (per-token
cache stride is rotary_dim//2), so we test that config (head_size == rotary_dim).

Config tested: num_tokens=6, n_qh=4, n_kh=2, head_size=rotary_dim=64,
mrope_section=[16,8,8], non-interleaved. fp32 throughout.
Reference: pure PyTorch section-select + split-half rotation (matches
`forward_native`'s non-interleaved cos/sin concatenation).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_triton_mrope_forward.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/rotary_embedding/mrope.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope  # noqa: E402

torch.manual_seed(42)

NUM_TOKENS = 6
N_QH = 4
N_KH = 2
HEAD_SIZE = 64
ROTARY_DIM = 64
HALF = ROTARY_DIM // 2
MROPE_SECTION = [16, 8, 8]  # sums to HALF
INTERLEAVED = False

Q = torch.randn(NUM_TOKENS, N_QH * HEAD_SIZE, dtype=torch.float32, device=DEVICE)
K = torch.randn(NUM_TOKENS, N_KH * HEAD_SIZE, dtype=torch.float32, device=DEVICE)
# cos/sin caches shaped (3, num_tokens, head_dim // 2)
_angles = torch.randn(3, NUM_TOKENS, HALF, dtype=torch.float32, device=DEVICE)
COS = torch.cos(_angles)
SIN = torch.sin(_angles)


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


def pytorch_ref(q, k, cos, sin, mrope_section, head_size, rotary_dim):
    """Pure PyTorch MRoPE. No triton/vLLM kernel calls."""
    nt = q.shape[0]
    half = rotary_dim // 2
    t, h, w = mrope_section
    cos_sel = torch.cat(
        [cos[0][:, :t], cos[1][:, t:t + h], cos[2][:, t + h:half]], dim=-1
    ).float()
    sin_sel = torch.cat(
        [sin[0][:, :t], sin[1][:, t:t + h], sin[2][:, t + h:half]], dim=-1
    ).float()
    c = cos_sel[:, None, :]
    s = sin_sel[:, None, :]

    def rot(x, nh):
        x = x.view(nt, nh, head_size).float()
        x1 = x[..., :half]
        x2 = x[..., half:rotary_dim]
        xpass = x[..., rotary_dim:]
        n1 = x1 * c - x2 * s
        n2 = x2 * c + x1 * s
        out = torch.cat([n1, n2, xpass], dim=-1)
        return out.reshape(nt, nh * head_size)

    return rot(q, q.shape[1] // head_size), rot(k, k.shape[1] // head_size)


def kernel_impl(q, k, cos, sin, mrope_section, head_size, rotary_dim):
    """Kernel launch only (kernel mutates q/k in place; use clones)."""
    q_c = q.clone()
    k_c = k.clone()
    q_out, k_out = triton_mrope(
        q_c, k_c, cos, sin, mrope_section, head_size, rotary_dim, INTERLEAVED
    )
    return q_out, k_out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_q, ref_k = pytorch_ref(
            Q, K, COS, SIN, MROPE_SECTION, HEAD_SIZE, ROTARY_DIM
        )
        ker_q, ker_k = kernel_impl(
            Q, K, COS, SIN, MROPE_SECTION, HEAD_SIZE, ROTARY_DIM
        )

        ref_q_c, ref_k_c = ref_q.cpu(), ref_k.cpu()
        ker_q_c, ker_k_c = ker_q.cpu(), ker_k.cpu()
        torch.testing.assert_close(ker_q_c, ref_q_c, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ker_k_c, ref_k_c, rtol=1e-3, atol=1e-3)

        diff_q = (ker_q_c - ref_q_c).abs()
        diff_k = (ker_k_c - ref_k_c).abs()
        max_abs = max(diff_q.max().item(), diff_k.max().item())
        mean_abs = (diff_q.mean().item() + diff_k.mean().item()) / 2.0
        stats = {
            "input_shape": tuple(Q.shape),
            "output_shape": tuple(ker_q.shape),
            "in_dtype": str(Q.dtype),
            "out_dtype": str(ker_q.dtype),
            "device": str(Q.device),
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
        }

        pt_stats = _bench(
            lambda: pytorch_ref(Q, K, COS, SIN, MROPE_SECTION, HEAD_SIZE, ROTARY_DIM)
        )
        kern_stats = _bench(
            lambda: kernel_impl(Q, K, COS, SIN, MROPE_SECTION, HEAD_SIZE, ROTARY_DIM)
        )
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
            "Kernel: _triton_mrope_forward\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- q shape: {stats['input_shape']}\n",
                f"- in dtype: {stats['in_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- q out shape: {stats['output_shape']}\n",
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
