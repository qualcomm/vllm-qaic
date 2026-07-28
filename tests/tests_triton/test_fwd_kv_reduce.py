"""
Kernel test for `_fwd_kv_reduce` (sequential decayed reduction of per-block
KV outer products into a running KV history), from lightning (linear)
attention.

Source under test:
vllm/model_executor/layers/lightning_attn.py
  - _fwd_kv_reduce

This test launches the Triton kernel directly (not through the fused
`lightning_attention` pipeline) with b=1, h=1, n=64, d=16, e=16, BLOCK=32,
NUM_BLOCK=2 -- two blocks, so the sequential reduction across blocks is
actually exercised (unlike the other four lightning_attn tests which use a
single block).

Semantics (per source): kv_pre starts as kv_history (the incoming running
history). For each block i in 0..NUM_BLOCK-1: block_decay = exp(-s *
block_size) (block_size = min(n - i*BLOCK, BLOCK)); kv_cur = kv[i] (the raw
per-block K^T@V outer product, e.g. produced by `_fwd_kv_parallel`); the
kernel OVERWRITES kv[i] with kv_pre (the pre-block running history, not the
raw per-block sum -- this in-place rewrite is the key semantic under test);
then kv_pre = block_decay * kv_pre + kv_cur. After the loop, kv_pre is
stored into kv_history (the final post-sequence running history).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_fwd_kv_reduce.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/lightning_attn.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.model_executor.layers.lightning_attn import _fwd_kv_reduce  # noqa: E402

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Global inputs (shared by pytorch_ref and kernel_impl)
# ---------------------------------------------------------------------------
DEVICE = "qaic"
DTYPE = torch.float32

B, H, N, D, E = 1, 1, 64, 16, 16
BLOCK = 32
NUM_BLOCK = 2  # n // BLOCK, exact division
D_FBLOCK = D
E_FBLOCK = E
NUM_FBLOCK = 1

S = torch.full((H,), 0.05, dtype=DTYPE, device=DEVICE)
# Raw per-block K^T@V outer products (as would be produced by
# `_fwd_kv_parallel`), and the incoming running KV history.
KV_INPUT = torch.randn(B, H, NUM_BLOCK, D, E, dtype=DTYPE, device=DEVICE)
KV_HISTORY_INPUT = torch.randn(B, H, D, E, dtype=DTYPE, device=DEVICE)


def pytorch_ref(s, kv_input, kv_history_input):
    """Pure PyTorch reference for `_fwd_kv_reduce`.

    Returns (kv_rewritten, kv_history_final):
      - kv_rewritten: kv tensor with each block overwritten by the
        pre-block running history (matches the kernel's in-place rewrite).
      - kv_history_final: the running history after processing all blocks.
    """
    s_cpu = s.cpu()
    kv_cpu = kv_input.cpu().clone()
    kv_history_cpu = kv_history_input.cpu().clone()

    b, h, num_block, d, e = kv_cpu.shape
    kv_out = torch.zeros_like(kv_cpu)
    kv_history_out = torch.zeros_like(kv_history_cpu)

    for bi in range(b):
        for hi in range(h):
            s_val = float(s_cpu[hi].item())
            kv_pre = kv_history_cpu[bi, hi].clone()
            for i in range(num_block):
                block_size = min(N - i * BLOCK, BLOCK)
                block_decay = torch.exp(
                    torch.tensor(-s_val * block_size, dtype=torch.float32)
                )
                kv_cur = kv_cpu[bi, hi, i].clone()
                # Overwrite this block's slot with the PRE-block history.
                kv_out[bi, hi, i] = kv_pre
                kv_pre = block_decay * kv_pre + kv_cur
            kv_history_out[bi, hi] = kv_pre

    return kv_out, kv_history_out


def kernel_impl(s, kv_input, kv_history_input):
    """Kernel wrapper: launches `_fwd_kv_reduce` directly.

    Kernel launch only -- no reference logic, no validation logic. `kv` and
    `kv_history` are mutated in place by the kernel, matching source
    semantics; we clone the shared globals first so kernel_impl never
    mutates state that pytorch_ref also reads.
    """
    kv = kv_input.clone()
    kv_history = kv_history_input.clone()

    b, h, num_block, d, e = kv.shape
    n = N

    grid = (b * h, NUM_FBLOCK)
    _fwd_kv_reduce[grid](
        s,
        kv,
        kv_history,
        b,
        h,
        n,
        d,
        e,
        BLOCK=BLOCK,
        NUM_BLOCK=num_block,
        D_FBLOCK=D_FBLOCK,
        E_FBLOCK=E_FBLOCK,
    )
    return kv, kv_history


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


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_kv, ref_kv_history = pytorch_ref(S, KV_INPUT, KV_HISTORY_INPUT)
        kernel_kv, kernel_kv_history = kernel_impl(S, KV_INPUT, KV_HISTORY_INPUT)

        ref_kv_cpu = ref_kv.cpu()
        kernel_kv_cpu = kernel_kv.cpu()
        ref_hist_cpu = ref_kv_history.cpu()
        kernel_hist_cpu = kernel_kv_history.cpu()

        torch.testing.assert_close(kernel_kv_cpu, ref_kv_cpu, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(
            kernel_hist_cpu, ref_hist_cpu, rtol=1e-3, atol=1e-3
        )

        diff_kv = (kernel_kv_cpu - ref_kv_cpu).abs()
        diff_hist = (kernel_hist_cpu - ref_hist_cpu).abs()
        rel_err_kv = (diff_kv / (ref_kv_cpu.abs() + 1e-8)).mean().item()
        rel_err_hist = (diff_hist / (ref_hist_cpu.abs() + 1e-8)).mean().item()

        stats = {
            "kv_input_shape": tuple(KV_INPUT.shape),
            "kv_history_input_shape": tuple(KV_HISTORY_INPUT.shape),
            "kv_output_shape": tuple(kernel_kv.shape),
            "kv_history_output_shape": tuple(kernel_kv_history.shape),
            "input_dtype": str(KV_INPUT.dtype),
            "output_dtype": str(kernel_kv.dtype),
            "device": str(KV_INPUT.device),
            "max_abs_diff_kv": diff_kv.max().item(),
            "mean_abs_diff_kv": diff_kv.mean().item(),
            "relative_error_kv": rel_err_kv,
            "max_abs_diff_kv_history": diff_hist.max().item(),
            "mean_abs_diff_kv_history": diff_hist.mean().item(),
            "relative_error_kv_history": rel_err_hist,
            "grid": f"({B * H}, {NUM_FBLOCK})",
        }

        pt_stats = _bench(lambda: pytorch_ref(S, KV_INPUT, KV_HISTORY_INPUT))
        kern_stats = _bench(lambda: kernel_impl(S, KV_INPUT, KV_HISTORY_INPUT))
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
            "Kernel: _fwd_kv_reduce\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- kv (raw per-block) shape: {stats['kv_input_shape']}\n")
            lines.append(
                f"- kv_history (initial) shape: {stats['kv_history_input_shape']}\n"
            )
            lines.append(f"- input dtype: {stats['input_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n")
            lines.append(f"- s (decay): {S.cpu().tolist()}\n")
            lines.append(
                f"- BLOCK={BLOCK}, NUM_BLOCK={NUM_BLOCK}, D_FBLOCK={D_FBLOCK}, "
                f"E_FBLOCK={E_FBLOCK}\n\n"
            )
            lines.append("Grid Configuration:\n")
            lines.append(f"- grid: {stats['grid']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- kv (rewritten) shape: {stats['kv_output_shape']}\n")
            lines.append(
                f"- kv_history (final) shape: {stats['kv_history_output_shape']}\n"
            )
            lines.append(f"- output dtype: {stats['output_dtype']}\n")
            lines.append(f"- max_abs_diff (kv): {stats['max_abs_diff_kv']}\n")
            lines.append(f"- mean_abs_diff (kv): {stats['mean_abs_diff_kv']}\n")
            lines.append(f"- relative_error (kv): {stats['relative_error_kv']}\n")
            lines.append(
                f"- max_abs_diff (kv_history): {stats['max_abs_diff_kv_history']}\n"
            )
            lines.append(
                f"- mean_abs_diff (kv_history): {stats['mean_abs_diff_kv_history']}\n"
            )
            lines.append(
                f"- relative_error (kv_history): "
                f"{stats['relative_error_kv_history']}\n"
            )
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
    sys.exit(0 if main() == "SUCCESS" else 1)
