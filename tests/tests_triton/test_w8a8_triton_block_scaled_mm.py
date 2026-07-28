"""
Standalone QAIC validation for `_w8a8_triton_block_scaled_mm`.

Source under test:
vllm/model_executor/layers/quantization/utils/fp8_utils.py
  - _w8a8_triton_block_scaled_mm  (@triton.jit)
  - w8a8_triton_block_scaled_mm   (launcher)

Block-wise-quantized FP8 (W8A8) GEMM. `A` (M, K) is FP8 with per-token-group
scales `As` (M, K/block_k); `B` (N, K) is FP8 with per-block scales `Bs`
(N/block_n, K/block_k). The kernel accumulates raw FP8 dot products per
K-block and rescales each tile by the corresponding A- and B-block scales:
    C[m, n] = sum_kb ( sum_{k in kb} A_code[m,k] * B_code[n,k] )
              * As[m, kb] * Bs[n//block_n, kb]
which equals dequant(A) @ dequant(B).T.

Reference: pure PyTorch block-dequantization of A and B followed by
`A_deq @ B_deq.T`. Because the reference reuses the SAME fp8 codes and scales
produced for the kernel, the only divergence is fp32 accumulation order, so we
compare at rtol/atol = 1e-2 (fp8 GEMM low precision).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_w8a8_triton_block_scaled_mm.txt")
KERNEL_FILE_PATH = (
    "vllm/model_executor/layers/quantization/utils/fp8_utils.py"
)

M = 64
N = 64
K = 64
BLOCK_N = 32
BLOCK_K = 32
BLOCK_SIZE = [BLOCK_N, BLOCK_K]
EPS = 1e-10
DEVICE = "qaic"

_IS_CHILD = os.environ.get("W8A8_FP8_MM_CHILD") == "1"

if _IS_CHILD or __name__ != "__main__":
    import torch

    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm")
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
        w8a8_triton_block_scaled_mm,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        get_fp8_min_max,
    )
    from vllm.platforms import current_platform

    torch.manual_seed(42)
    FP8_DTYPE = current_platform.fp8_dtype()
    FP8_MIN, FP8_MAX = get_fp8_min_max()

    A_F = torch.randn(M, K, dtype=torch.float32, device=DEVICE)
    B_F = torch.randn(N, K, dtype=torch.float32, device=DEVICE)

    # Quantize A with per-token-group (group == block_k) scales -> row-major As.
    A_CODE, AS = per_token_group_quant_fp8(
        A_F.contiguous(), BLOCK_K, eps=EPS, column_major_scales=False,
        use_ue8m0=False,
    )

    def _block_quant_b(bf):
        """Per (block_n, block_k) absmax FP8 quant of B -> (codes, scales)."""
        n, k = bf.shape
        nb = n // BLOCK_N
        kb = k // BLOCK_K
        codes = torch.empty(n, k, dtype=FP8_DTYPE, device=bf.device)
        scales = torch.empty(nb, kb, dtype=torch.float32, device=bf.device)
        eps_t = torch.tensor(EPS, dtype=torch.float32, device=bf.device)
        for i in range(nb):
            for j in range(kb):
                blk = bf[i * BLOCK_N:(i + 1) * BLOCK_N,
                         j * BLOCK_K:(j + 1) * BLOCK_K]
                absmax = torch.maximum(blk.abs().max(), eps_t)
                scale = absmax * (1.0 / FP8_MAX)
                q = torch.clamp(blk / scale, FP8_MIN, FP8_MAX).to(FP8_DTYPE)
                codes[i * BLOCK_N:(i + 1) * BLOCK_N,
                      j * BLOCK_K:(j + 1) * BLOCK_K] = q
                scales[i, j] = scale
        return codes.contiguous(), scales

    B_CODE, BS = _block_quant_b(B_F)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def pytorch_ref(a_code, as_, b_code, bs):
    """Pure PyTorch block-dequant GEMM: dequant(A) @ dequant(B).T."""
    a_code = a_code.to(torch.float32).cpu()
    b_code = b_code.to(torch.float32).cpu()
    as_ = as_.to(torch.float32).cpu()
    bs = bs.to(torch.float32).cpu()

    # Expand A scales (M, K/block_k) -> (M, K).
    a_scale_full = as_.repeat_interleave(BLOCK_K, dim=1)
    a_deq = a_code * a_scale_full
    # Expand B scales (N/block_n, K/block_k) -> (N, K).
    b_scale_full = bs.repeat_interleave(BLOCK_N, dim=0).repeat_interleave(
        BLOCK_K, dim=1
    )
    b_deq = b_code * b_scale_full

    return a_deq @ b_deq.t()


def kernel_impl(a_code, as_, b_code, bs):
    c = w8a8_triton_block_scaled_mm(
        a_code, b_code, as_, bs, BLOCK_SIZE, output_dtype=torch.float32
    )
    return c.to(torch.float32)


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


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_c = pytorch_ref(A_CODE, AS, B_CODE, BS)
        kern_c = kernel_impl(A_CODE, AS, B_CODE, BS)

        ref_c = ref_c.cpu()
        kern_c = kern_c.cpu()

        torch.testing.assert_close(kern_c, ref_c, rtol=1e-2, atol=1e-2)

        diff = (kern_c - ref_c).abs()
        rel = diff / (ref_c.abs() + 1e-6)
        stats = {
            "A_shape": tuple(A_CODE.shape),
            "B_shape": tuple(B_CODE.shape),
            "As_shape": tuple(AS.shape),
            "Bs_shape": tuple(BS.shape),
            "C_shape": tuple(kern_c.shape),
            "fp8_dtype": str(FP8_DTYPE),
            "device": str(A_CODE.device),
            "block_size": BLOCK_SIZE,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "max_rel_err": rel.max().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(A_CODE, AS, B_CODE, BS))
        kern_stats = _bench(lambda: kernel_impl(A_CODE, AS, B_CODE, BS))
        speedup = (kern_stats["avg_ms"] / pt_stats["avg_ms"]
                   if pt_stats["avg_ms"] > 0 else float("nan"))
        stats["pytorch_latency_ms"] = pt_stats
        stats["kernel_latency_ms"] = kern_stats
        stats["speedup_kernel_over_pytorch"] = speedup
        print(f"Speedup (Kernel/PyTorch): {speedup:.4f}x")
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
            "Kernel: _w8a8_triton_block_scaled_mm\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- A(fp8) shape: {stats['A_shape']}\n")
            lines.append(f"- B(fp8) shape: {stats['B_shape']}\n")
            lines.append(f"- As shape: {stats['As_shape']}\n")
            lines.append(f"- Bs shape: {stats['Bs_shape']}\n")
            lines.append(f"- block_size: {stats['block_size']}\n")
            lines.append(f"- fp8_dtype: {stats['fp8_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- C shape: {stats['C_shape']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            lines.append(f"- max_rel_err: {stats['max_rel_err']}\n")
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


def _run_with_crash_guard():
    import subprocess

    if os.environ.get("W8A8_FP8_MM_CHILD") == "1":
        sys.exit(0 if main() == "SUCCESS" else 1)

    env = dict(os.environ, W8A8_FP8_MM_CHILD="1")
    proc = subprocess.run([sys.executable, __file__], env=env)
    if proc.returncode < 0:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _log(
            f"{timestamp}\n"
            "Kernel: _w8a8_triton_block_scaled_mm\n"
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
