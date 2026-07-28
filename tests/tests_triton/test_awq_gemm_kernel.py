"""
Standalone QAIC validation for `awq_gemm_kernel`.

Source under test:
vllm/model_executor/layers/quantization/awq_triton.py
  - awq_gemm_kernel  (@triton.jit)
  - launcher: awq_gemm_triton(input, qweight, scales, qzeros, split_k_iters)

Fused GEMM against AWQ 4-bit packed weights, dequantizing weight tiles on the
fly (split-K reduction summed over the K dimension). Tensor layout:
    input   : [M, K]        float16
    qweight : [K, N // 8]   int32   (8 nibbles packed per int32)
    qzeros  : [K // G, N//8] int32  (packed zero-points)
    scales  : [K // G, N]   float16
Output    : [M, N] = input @ W, where W[k, n] = (nibble - zero) * scale.

AWQ NIBBLE-REORDER: identical convention to awq_dequantize_kernel — output
column n within each group of 8 reads the nibble at bit position order[n%8]*4
of the packed int32, order = [0, 4, 1, 5, 2, 6, 3, 7]. We generate the packed
int32s ourselves at those exact bit positions so the reference is unambiguous.

FLOAT/4-bit kernel. Reference: dequantize W (fp16) then matmul in fp16.
Compared at rtol/atol=1e-2 (fp16 GEMM accumulation + dequant rounding).
Documented.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import numpy as np
import torch

from vllm.model_executor.layers.quantization.awq_triton import awq_gemm_triton

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_awq_gemm_kernel.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/quantization/awq_triton.py"

DEVICE = "qaic"
torch.manual_seed(42)
np.random.seed(42)

_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

M = 8
K = 32
N = 16
GROUP_SIZE = 32               # K // G = 1
NUM_GROUPS = K // GROUP_SIZE   # 1
N_PACKED = N // 8             # 2
SPLIT_K = 1


def _pack_awq(nibbles):
    """nibbles: int array [R, C*8] in 0..15 -> packed uint32 [R, C]."""
    R, MM = nibbles.shape
    C = MM // 8
    packed = np.zeros((R, C), dtype=np.uint32)
    for r in range(R):
        for c in range(C):
            w = np.uint32(0)
            for j in range(8):
                nib = np.uint32(int(nibbles[r, c * 8 + j]) & 0xF)
                w |= nib << np.uint32(_ORDER[j] * 4)
            packed[r, c] = w
    return packed


_QW_NIB = np.random.randint(0, 16, size=(K, N)).astype(np.int64)
_ZERO_NIB = np.random.randint(0, 8, size=(NUM_GROUPS, N)).astype(np.int64)

_QW_PACKED_U32 = _pack_awq(_QW_NIB)          # [K, N_PACKED]
_ZERO_PACKED_U32 = _pack_awq(_ZERO_NIB)      # [NUM_GROUPS, N_PACKED]

INPUT = (torch.randn(M, K, dtype=torch.float32) * 0.5).to(torch.float16).to(DEVICE)
QWEIGHT = torch.from_numpy(_QW_PACKED_U32.view(np.int32)).to(DEVICE)
QZEROS = torch.from_numpy(_ZERO_PACKED_U32.view(np.int32)).to(DEVICE)
SCALES = (torch.rand(NUM_GROUPS, N, dtype=torch.float32) * 0.3 + 0.1).to(
    torch.float16
).to(DEVICE)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def pytorch_ref(inp, qw_nib, zero_nib, scales):
    scales = scales.cpu().to(torch.float16)
    qw = torch.from_numpy(qw_nib).to(torch.float16)          # [K, N]
    zero = torch.from_numpy(zero_nib).to(torch.float16)      # [NUM_GROUPS, N]
    w = torch.empty(K, N, dtype=torch.float16)
    for k in range(K):
        g = k // GROUP_SIZE
        w[k] = (qw[k] - zero[g]) * scales[g]
    return inp.cpu().to(torch.float16) @ w                   # [M, N]


def kernel_impl(inp, qweight, scales, qzeros):
    return awq_gemm_triton(inp, qweight, scales, qzeros, SPLIT_K)


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


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(INPUT, _QW_NIB, _ZERO_NIB, SCALES)
        kernel_out = kernel_impl(INPUT, QWEIGHT, SCALES, QZEROS)

        ref_cpu = ref_out.cpu().to(torch.float32)
        kernel_cpu = kernel_out.cpu().to(torch.float32)
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-2, atol=1e-2)

        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "input_shape": tuple(INPUT.shape),
            "qweight_shape": tuple(QWEIGHT.shape),
            "scales_shape": tuple(SCALES.shape),
            "qzeros_shape": tuple(QZEROS.shape),
            "output_shape": tuple(kernel_out.shape),
            "device": DEVICE,
            "group_size": GROUP_SIZE,
            "split_k": SPLIT_K,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(INPUT, _QW_NIB, _ZERO_NIB, SCALES))
        kern_stats = _bench(lambda: kernel_impl(INPUT, QWEIGHT, SCALES, QZEROS))
        speedup = kern_stats["avg_ms"] / pt_stats["avg_ms"] if pt_stats["avg_ms"] > 0 else float("nan")
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
            "Kernel: awq_gemm_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs (AWQ reverse-order nibble packing [0,4,1,5,2,6,3,7]):\n",
                f"- input shape: {stats['input_shape']} fp16\n",
                f"- qweight shape: {stats['qweight_shape']} int32\n",
                f"- scales shape: {stats['scales_shape']} fp16\n",
                f"- qzeros shape: {stats['qzeros_shape']} int32\n",
                f"- group_size: {stats['group_size']}, split_k: {stats['split_k']}\n",
                f"- device: {stats['device']}\n\n",
                "Output (fused dequant-GEMM fp16, rtol/atol=1e-2):\n",
                f"- out shape: {stats['output_shape']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
            ]
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
            lines += ["Error:\n", error_text + "\n"]
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
