"""
Kernel test for `_fwd_none_diag_kernel` (non-diagonal / cross-block
attention contribution using the running KV history), from lightning
(linear) attention.

Source under test:
vllm/model_executor/layers/lightning_attn.py
  - _fwd_none_diag_kernel

This test launches the Triton kernel directly (not through the fused
`lightning_attention` pipeline) with b=1, h=1, n=64, d=16, e=16, BLOCK=32,
NUM_BLOCK=2, E_FBLOCK=16 (== e, so NUM_FBLOCK is implicitly 1 as in the
real launch site, and `tl.program_id(2)` is always 0 since we only launch a
2D grid), CBLOCK=16, NUM_CBLOCK=2.

For isolated testing (per the task spec) we seed `O` with a known random
"diagonal contribution" tensor (as if produced by `_fwd_diag_kernel`) and
verify the kernel adds the correct non-diagonal term on top of it, using a
`KV` tensor that stands in for the pre-block running history that would be
produced by `_fwd_kv_reduce`.

Semantics (per source): for sub-block c (of size CBLOCK) within block
off_n, q_decay = exp(-s * (off_c*CBLOCK + local_position)); out_none_diag =
(q @ kv[off_n]) * q_decay; o[block] = o_diag[block] (existing diagonal
contribution, already in `O`) + out_none_diag, stored back into `O`.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_fwd_none_diag_kernel.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/lightning_attn.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.model_executor.layers.lightning_attn import (  # noqa: E402
    _fwd_none_diag_kernel,
)

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Global inputs (shared by pytorch_ref and kernel_impl)
# ---------------------------------------------------------------------------
DEVICE = "qaic"
DTYPE = torch.float32

B, H, N, D, E = 1, 1, 64, 16, 16
BLOCK = 32
NUM_BLOCK = 2  # n // BLOCK, exact division
E_FBLOCK = E  # NUM_FBLOCK == 1 in the real launch site
CBLOCK = 16
NUM_CBLOCK = BLOCK // CBLOCK  # 2

S = torch.full((H,), 0.05, dtype=DTYPE, device=DEVICE)
Q = torch.randn(B, H, N, D, dtype=DTYPE, device=DEVICE)
# Pre-block running KV history per block, as produced by `_fwd_kv_reduce`.
KV = torch.randn(B, H, NUM_BLOCK, D, E, dtype=DTYPE, device=DEVICE)
# Known "diagonal contribution" seed, as if produced by `_fwd_diag_kernel`.
O_DIAG = torch.randn(B, H, N, E, dtype=DTYPE, device=DEVICE)


def pytorch_ref(q, o_diag, s, kv):
    """Pure PyTorch reference for `_fwd_none_diag_kernel`.

    o_out[block][c] = o_diag[block][c] + (q[block][c] @ kv[block]) * q_decay
    where q_decay = exp(-s * (off_c*CBLOCK + local_position)) and
    local_position is the position within the CBLOCK sub-block.
    """
    q_cpu = q.cpu()
    o_diag_cpu = o_diag.cpu()
    s_cpu = s.cpu()
    kv_cpu = kv.cpu()

    b, h, n, d = q_cpu.shape
    e = kv_cpu.shape[-1]
    out = o_diag_cpu.clone()

    for bi in range(b):
        for hi in range(h):
            s_val = float(s_cpu[hi].item())
            for off_n in range(NUM_BLOCK):
                n_offset = off_n * BLOCK
                kv_block = kv_cpu[bi, hi, off_n]  # [d, e]
                for off_c in range(NUM_CBLOCK):
                    c_offset = off_c * CBLOCK
                    block_offset = n_offset + c_offset
                    q_block = q_cpu[
                        bi, hi, block_offset : block_offset + CBLOCK
                    ]  # [CBLOCK, d]
                    local_pos = torch.arange(CBLOCK, dtype=torch.float32)
                    q_decay = torch.exp(
                        -s_val * (off_c * CBLOCK + local_pos)
                    ).unsqueeze(
                        -1
                    )  # [CBLOCK, 1]
                    qkv_none_diag = (q_block @ kv_block) * q_decay  # [CBLOCK, e]
                    out[
                        bi, hi, block_offset : block_offset + CBLOCK
                    ] += qkv_none_diag

    return out


def kernel_impl(q, o_diag, s, kv):
    """Kernel wrapper: launches `_fwd_none_diag_kernel` directly.

    Kernel launch only -- no reference logic, no validation logic. `O` is
    mutated in place by the kernel (it adds the non-diagonal contribution
    on top of whatever is already stored there), so we clone `o_diag`
    before launching.
    """
    o = o_diag.clone()
    b, h, n, d = q.shape
    e = kv.shape[-1]

    grid = (b * h, NUM_BLOCK * NUM_CBLOCK)
    _fwd_none_diag_kernel[grid](
        q,
        o,
        s,
        kv,
        b,
        h,
        n,
        d,
        e,
        BLOCK=BLOCK,
        NUM_BLOCK=NUM_BLOCK,
        E_FBLOCK=E_FBLOCK,
        CBLOCK=CBLOCK,
        NUM_CBLOCK=NUM_CBLOCK,
    )
    return o


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
        ref_out = pytorch_ref(Q, O_DIAG, S, KV)
        kernel_out = kernel_impl(Q, O_DIAG, S, KV)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()

        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        rel_err = (diff / (ref_cpu.abs() + 1e-8)).mean().item()

        stats = {
            "q_shape": tuple(Q.shape),
            "o_diag_shape": tuple(O_DIAG.shape),
            "kv_shape": tuple(KV.shape),
            "output_shape": tuple(kernel_out.shape),
            "input_dtype": str(Q.dtype),
            "output_dtype": str(kernel_out.dtype),
            "device": str(Q.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "relative_error": rel_err,
            "grid": f"({B * H}, {NUM_BLOCK * NUM_CBLOCK})",
        }

        pt_stats = _bench(lambda: pytorch_ref(Q, O_DIAG, S, KV))
        kern_stats = _bench(lambda: kernel_impl(Q, O_DIAG, S, KV))
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
            "Kernel: _fwd_none_diag_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- q shape: {stats['q_shape']}\n")
            lines.append(f"- o (diag seed) shape: {stats['o_diag_shape']}\n")
            lines.append(f"- kv (pre-block history) shape: {stats['kv_shape']}\n")
            lines.append(f"- input dtype: {stats['input_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n")
            lines.append(f"- s (decay): {S.cpu().tolist()}\n")
            lines.append(
                f"- BLOCK={BLOCK}, NUM_BLOCK={NUM_BLOCK}, CBLOCK={CBLOCK}, "
                f"NUM_CBLOCK={NUM_CBLOCK}, E_FBLOCK={E_FBLOCK}\n\n"
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
