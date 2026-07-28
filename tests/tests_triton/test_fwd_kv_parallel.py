"""
Kernel test for `_fwd_kv_parallel` (per-block K^T@V outer-product
accumulation with decay), from lightning (linear) attention.

Source under test:
vllm/model_executor/layers/lightning_attn.py
  - _fwd_kv_parallel

This test launches the Triton kernel directly with a single-block
configuration (b=1, h=1, n=64, d=16, e=16, BLOCK=64 => NUM_BLOCK=1,
D_FBLOCK=d, E_FBLOCK=e, NUM_FBLOCK=1, CBLOCK=1 => NUM_CBLOCK=64) so n is
exactly divisible by BLOCK (no last-block boundary handling) and BLOCK is
exactly divisible by CBLOCK (no left-shift/boundary handling inside the
sub-block loop either -- left_shift == 0 throughout).

COMPILER COMPROMISE: CBLOCK values of 64 (one sub-block), 32, 16, 8, 4 and 2
were all tried first and every one of them failed to compile on this
environment's QAIC/Hexagon Triton backend with
`error: op was not bufferized` / "Failed to convert Triton Linalg to LLVM
MLIR" inside the shared `translate_linalg_to_obj` lowering stage. This
reproduces with the *unmodified* upstream `_fwd_kv_parallel` kernel itself
(confirmed by launching the real kernel directly, and by bisecting a
hand-written copy of the kernel line-by-line to isolate which construct
triggers it) -- it is a backend/compiler limitation on this box for this
kernel's per-sub-block masked-load + `tl.dot` pattern at these tile sizes,
not a bug in our test harness or in the kernel's semantics. Only CBLOCK=1
(NUM_CBLOCK=64) compiles successfully here, so that configuration is used;
it was verified numerically correct (max abs diff ~1e-6) against a hand
reference before being adopted for this file.

Semantics (per source): for the single block, per-timestep decay weight
k_decay[t] = exp(-s*(BLOCK - (t+1))) (t is 0-indexed position within the
block), and kv_block = sum_t k_decay[t] * outer(k[t], v[t]) ==
(k_decay[:, None] * k_block).T @ v_block.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_fwd_kv_parallel.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/lightning_attn.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.model_executor.layers.lightning_attn import _fwd_kv_parallel  # noqa: E402

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Global inputs (shared by pytorch_ref and kernel_impl)
# ---------------------------------------------------------------------------
DEVICE = "qaic"
DTYPE = torch.float32

B, H, N, D, E = 1, 1, 64, 16, 16
BLOCK = 64
NUM_BLOCK = 1  # n // BLOCK, exact division -> no boundary handling needed
D_FBLOCK = D
E_FBLOCK = E
NUM_FBLOCK = 1
CBLOCK = 1  # only tile size that compiles on this QAIC/Hexagon backend
NUM_CBLOCK = BLOCK // CBLOCK  # 64

K = torch.randn(B, H, N, D, dtype=DTYPE, device=DEVICE)
V = torch.randn(B, H, N, E, dtype=DTYPE, device=DEVICE)
S = torch.full((H,), 0.05, dtype=DTYPE, device=DEVICE)

# k_decay construction exactly as in `_attention.forward`:
#   array = arange(0, BLOCK) + 1
#   k_decay = exp(-s * (BLOCK - array.reshape(1, -1)))
_array = torch.arange(0, BLOCK, dtype=DTYPE, device=DEVICE) + 1
K_DECAY = torch.exp(-S.view(-1, 1) * (BLOCK - _array.view(1, -1)))  # [H, BLOCK]


def pytorch_ref(k, v, k_decay):
    """Pure PyTorch reference for `_fwd_kv_parallel`.

    kv_block[b,h] = (k_decay[h][:, None] * k[b,h]).T @ v[b,h]
    (single block since NUM_BLOCK == 1 here).
    """
    k_cpu = k.cpu()
    v_cpu = v.cpu()
    k_decay_cpu = k_decay.cpu()

    b, h, n, d = k_cpu.shape
    e = v_cpu.shape[-1]
    kv = torch.zeros(b, h, NUM_BLOCK, d, e, dtype=torch.float32)

    for bi in range(b):
        for hi in range(h):
            decay = k_decay_cpu[hi]  # [BLOCK]
            k_weighted = k_cpu[bi, hi] * decay[:, None]  # [n, d]
            kv[bi, hi, 0] = k_weighted.t() @ v_cpu[bi, hi]

    return kv


def kernel_impl(k, v, k_decay):
    """Kernel wrapper: launches `_fwd_kv_parallel` directly.

    Kernel launch only -- no reference logic, no validation logic.
    """
    b, h, n, d = k.shape
    e = v.shape[-1]
    kv = torch.empty(b, h, NUM_BLOCK, d, e, dtype=torch.float32, device=k.device)

    grid = (b * h, NUM_BLOCK)
    _fwd_kv_parallel[grid](
        k,
        v,
        k_decay,
        kv,
        b,
        h,
        n,
        d,
        e,
        BLOCK=BLOCK,
        NUM_BLOCK=NUM_BLOCK,
        D_FBLOCK=D_FBLOCK,
        E_FBLOCK=E_FBLOCK,
        NUM_FBLOCK=NUM_FBLOCK,
        CBLOCK=CBLOCK,
        NUM_CBLOCK=NUM_CBLOCK,
    )
    return kv


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
        ref_out = pytorch_ref(K, V, K_DECAY)
        kernel_out = kernel_impl(K, V, K_DECAY)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()

        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        rel_err = (diff / (ref_cpu.abs() + 1e-8)).mean().item()

        stats = {
            "k_shape": tuple(K.shape),
            "v_shape": tuple(V.shape),
            "k_decay_shape": tuple(K_DECAY.shape),
            "output_shape": tuple(kernel_out.shape),
            "input_dtype": str(K.dtype),
            "output_dtype": str(kernel_out.dtype),
            "device": str(K.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "relative_error": rel_err,
            "grid": f"({B * H}, {NUM_BLOCK})",
        }

        pt_stats = _bench(lambda: pytorch_ref(K, V, K_DECAY))
        kern_stats = _bench(lambda: kernel_impl(K, V, K_DECAY))
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
            "Kernel: _fwd_kv_parallel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- k shape: {stats['k_shape']}\n")
            lines.append(f"- v shape: {stats['v_shape']}\n")
            lines.append(f"- k_decay shape: {stats['k_decay_shape']}\n")
            lines.append(f"- input dtype: {stats['input_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n")
            lines.append(f"- s (decay): {S.cpu().tolist()}\n")
            lines.append(
                f"- BLOCK={BLOCK}, NUM_BLOCK={NUM_BLOCK}, CBLOCK={CBLOCK}, "
                f"D_FBLOCK={D_FBLOCK}, E_FBLOCK={E_FBLOCK}\n\n"
            )
            lines.append("Grid Configuration:\n")
            lines.append(f"- grid: {stats['grid']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- output dtype: {stats['output_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            lines.append(f"- relative_error: {stats['relative_error']}\n")
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
