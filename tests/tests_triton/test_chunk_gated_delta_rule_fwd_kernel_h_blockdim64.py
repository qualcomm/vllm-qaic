"""
Standalone QAIC validation for `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`.

Source under test:
vllm/model_executor/layers/fla/ops/chunk_delta_h.py
  - chunk_gated_delta_rule_fwd_kernel_h_blockdim64  (recomputes and stores
    per-chunk recurrent hidden states for Gated DeltaNet.)

For a single chunk (B=1, T=64=chunk_size, Hg=1, H=1, K=64, V=32), with
initial_state=None (USE_INITIAL_STATE=False -> h starts at 0), g provided
(scalar per-head decay, USE_G=True), gk=None (USE_GK=False), use_exp2=False:

    h[:, 0] = h_0                              (state *before* this chunk)
    v_new   = u - w @ h_0^T                     (delta, stored BEFORE decay
                                                  correction)
    g_last  = g[last position of chunk]
    v_corr  = v_new * exp(g_last - g_i)          (per-token decay correction,
                                                  applied only for the state
                                                  update, not for v_new output)
    h_1     = h_0 * exp(g_last) + v_corr^T @ k    (outer-product accumulation)
    final_state = h_1  (since NT == 1)

Reference: pure PyTorch chunked loop over batch/head reproducing the above
exactly in fp32 (h/v_new/final_state dtypes matching the launcher's own
dtype choices), read directly from chunk_delta_h.py's kernel body.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(
    LOG_DIR, "log_chunk_gated_delta_rule_fwd_kernel_h_blockdim64.txt"
)
KERNEL_FILE_PATH = "vllm/model_executor/layers/fla/ops/chunk_delta_h.py"

DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402
import triton.testing  # noqa: E402


def _qaic_safe_do_bench(fn, warmup=25, rep=100, grad_to_none=None,
                         quantiles=None, return_mode="mean"):
    """Wall-clock replacement for `triton.testing.do_bench`.

    The stock implementation times kernels via `torch.Event.elapsed_time`,
    which is broken for the QAIC backend in this environment
    (`RuntimeError: expected other to be a torch.Event object`). Triton's
    `@triton.autotune` decorator calls this during its very first config
    search, so every autotuned FLA kernel hits this crash before any of our
    kernel-under-test logic even runs. We swap in a simple `time.perf_counter`
    based timer (device-synced) so autotuning can proceed; this only affects
    *which* config is picked / timed, not kernel correctness.
    """
    import time

    fn()
    torch.qaic.synchronize()
    n_repeat = 3
    times = []
    for _ in range(n_repeat):
        start = time.perf_counter()
        fn()
        torch.qaic.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    if quantiles is not None:
        import numpy as np

        return list(np.quantile(times, quantiles))
    return sum(times) / len(times)


triton.testing.do_bench = _qaic_safe_do_bench

from vllm.model_executor.layers.fla.ops.chunk_delta_h import (  # noqa: E402
    chunk_gated_delta_rule_fwd_h,
)

# ---------------------------------------------------------------------------
# Global inputs (single chunk: B=1, T=64=chunk_size, Hg=1, H=1, K=64, V=32)
# ---------------------------------------------------------------------------
B = 1
T = 64
Hg = 1
H = 1
K = 64
V = 32
CHUNK_SIZE = 64
DTYPE = torch.float32

torch.manual_seed(42)
KEY = torch.randn(B, T, Hg, K, dtype=DTYPE, device=DEVICE) * 0.1
W = torch.randn(B, T, H, K, dtype=DTYPE, device=DEVICE) * 0.1
U = torch.randn(B, T, H, V, dtype=DTYPE, device=DEVICE) * 0.1
# Per-(token, head) cumulative log-decay within the chunk (monotonically
# non-increasing). The kernel reads g_last (last position) and per-position
# g for the v decay-correction step.
_log_decay = -0.02 * torch.rand(B, T, H, dtype=DTYPE, device=DEVICE)
G = torch.cumsum(_log_decay, dim=1)


def pytorch_ref(k, w, u, g, chunk_size):
    """Pure PyTorch reference for chunk_gated_delta_rule_fwd_kernel_h_blockdim64.

    initial_state=None (state starts at 0), gk=None, use_exp2=False.
    """
    b_, t_, hg_, k_ = k.shape
    h_ = u.shape[2]
    v_ = u.shape[3]
    rep = h_ // hg_
    bt = chunk_size
    nt = (t_ + bt - 1) // bt

    kf = k.float()
    wf = w.float()
    uf = u.float()
    gf = g.float() if g is not None else None

    h_out = torch.zeros(b_, nt, h_, v_, k_, dtype=torch.float32, device=k.device)
    v_new = torch.zeros_like(uf)
    final_state = torch.zeros(b_, h_, v_, k_, dtype=torch.float32, device=k.device)

    for bi in range(b_):
        for hh in range(h_):
            hg_idx = hh // rep
            h_t = torch.zeros(v_, k_, dtype=torch.float32, device=k.device)
            for it in range(nt):
                lo, hi = it * bt, min((it + 1) * bt, t_)
                w_blk = wf[bi, lo:hi, hh, :]  # [bt, K]
                k_blk = kf[bi, lo:hi, hg_idx, :]  # [bt, K]
                u_blk = uf[bi, lo:hi, hh, :]  # [bt, V]

                h_out[bi, it, hh] = h_t

                v_delta = u_blk - w_blk @ h_t.transpose(0, 1)  # [bt, V]
                v_new[bi, lo:hi, hh, :] = v_delta

                if gf is not None:
                    g_blk = gf[bi, lo:hi, hh]  # [bt]
                    g_last = g_blk[-1]
                    v_corr = v_delta * torch.exp(g_last - g_blk)[:, None]
                    decay = torch.exp(g_last)
                else:
                    v_corr = v_delta
                    decay = torch.ones((), dtype=torch.float32, device=k.device)

                h_t = h_t * decay + v_corr.transpose(0, 1) @ k_blk  # [V, K]

            final_state[bi, hh] = h_t

    return h_out.to(k.dtype), v_new.to(u.dtype), final_state


def kernel_impl(k, w, u, g, chunk_size):
    return chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=None,
        initial_state=None,
        output_final_state=True,
        chunk_size=chunk_size,
        save_new_value=True,
        use_exp2=False,
    )


def _bench(fn, warmup=3, iters=10):
    """Device-synced wall-clock benchmark. Returns dict of latency stats (ms)."""
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
        ref_h, ref_v_new, ref_final = pytorch_ref(KEY, W, U, G, CHUNK_SIZE)
        kernel_h, kernel_v_new, kernel_final = kernel_impl(KEY, W, U, G, CHUNK_SIZE)

        ref_h_cpu = ref_h.cpu().float()
        ref_v_cpu = ref_v_new.cpu().float()
        ref_final_cpu = ref_final.cpu().float()
        kernel_h_cpu = kernel_h.cpu().float()
        kernel_v_cpu = kernel_v_new.cpu().float()
        kernel_final_cpu = kernel_final.cpu().float()

        torch.testing.assert_close(kernel_h_cpu, ref_h_cpu, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(kernel_v_cpu, ref_v_cpu, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(
            kernel_final_cpu, ref_final_cpu, rtol=1e-3, atol=1e-3
        )

        diff_h = (kernel_h_cpu - ref_h_cpu).abs()
        diff_v = (kernel_v_cpu - ref_v_cpu).abs()
        diff_final = (kernel_final_cpu - ref_final_cpu).abs()
        max_abs_diff = max(
            diff_h.max().item(), diff_v.max().item(), diff_final.max().item()
        )
        mean_abs_diff = (
            diff_h.mean().item() + diff_v.mean().item() + diff_final.mean().item()
        ) / 3.0
        rel_err = (
            diff_final / (ref_final_cpu.abs() + 1e-6)
        ).mean().item()

        stats = {
            "input_shapes": {
                "k": tuple(KEY.shape),
                "w": tuple(W.shape),
                "u": tuple(U.shape),
                "g": tuple(G.shape),
            },
            "output_shapes": {
                "h": tuple(kernel_h.shape),
                "v_new": tuple(kernel_v_new.shape),
                "final_state": tuple(kernel_final.shape),
            },
            "input_dtype": str(KEY.dtype),
            "output_dtype_h": str(kernel_h.dtype),
            "output_dtype_final_state": str(kernel_final.dtype),
            "device": str(KEY.device),
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "rel_err_final_state": rel_err,
        }
        pt_stats = _bench(lambda: pytorch_ref(KEY, W, U, G, CHUNK_SIZE))
        kern_stats = _bench(lambda: kernel_impl(KEY, W, U, G, CHUNK_SIZE))
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
            "Kernel: chunk_gated_delta_rule_fwd_kernel_h_blockdim64\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            for name, shape in stats["input_shapes"].items():
                lines.append(f"- {name} shape: {shape}\n")
            lines.append(f"- dtype: {stats['input_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Outputs:\n")
            for name, shape in stats["output_shapes"].items():
                lines.append(f"- {name} shape: {shape}\n")
            lines.append(f"- h dtype: {stats['output_dtype_h']}\n")
            lines.append(f"- final_state dtype: {stats['output_dtype_final_state']}\n")
            lines.append(f"- max_abs_diff (across h/v_new/final_state): {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff (avg across h/v_new/final_state): {stats['mean_abs_diff']}\n")
            lines.append(f"- rel_err (final_state, mean): {stats['rel_err_final_state']}\n")
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
    result = main()
    sys.exit(0 if result == "SUCCESS" else 1)
