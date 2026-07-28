"""
Standalone QAIC validation for `do_shrink_kernel_fp8`.

Source under test:
vllm/lora/ops/triton_ops/fp8_kernel_utils.py
  - do_shrink_kernel_fp8  (FP8 LoRA shrink, one CTA/slice/lora, dequant + scaling)

`do_shrink_kernel_fp8` is a @triton.jit device helper invoked inside
`_lora_shrink_kernel_fp8`. For rows in `ram` it computes
  out[row] = scaling * dequant_a_scale * dequant_b_scale * (x[row] @ A[lora].T)
via fp8_mm_k, then stores (SPLIT_K==1) or atomic-adds (SPLIT_K>1). We use the
simplest tensor-wise path (group_k==group_n==0, per_channel_quant=False,
SLICE_NUM==1, SPLIT_K==1) and wrap with ram = arange(M).

FP8 REPRESENTATION: x and A stored as torch.float8_e4m3fn; a_scale is a scalar
(tensor-wise activation scale), b_scale is per-lora [num_lora]. Reference
dequantizes with those scales.

Reference: scaling * (x.float()*a_scale) @ (A[lora].float()*b_scale).T
FLOAT compare.
"""

import datetime
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

from vllm.lora.ops.triton_ops.fp8_kernel_utils import do_shrink_kernel_fp8
from vllm.triton_utils import tl, triton

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_do_shrink_kernel_fp8.txt")
KERNEL_FILE_PATH = "vllm/lora/ops/triton_ops/fp8_kernel_utils.py"
DEVICE = "qaic"
FP8_DTYPE = torch.float8_e4m3fn

torch.manual_seed(42)

# Global shared inputs
M = 8
HIDDEN = 16      # K
RANK = 8         # N
NUM_LORA = 2
LORA_INDEX = 1
SCALING = 0.5
A_SCALE = 0.75          # tensor-wise activation scale
B_SCALES = [1.1, 0.9]   # per-lora weight scale

BLOCK_M = 8
BLOCK_N = 8
BLOCK_K = 16
EVEN_K = HIDDEN % BLOCK_K == 0

X = (torch.randn(M, HIDDEN, device=DEVICE) * 0.25).to(FP8_DTYPE)
LORA_A = (torch.randn(NUM_LORA, RANK, HIDDEN, device=DEVICE) * 0.25).to(FP8_DTYPE)


@triton.jit
def _shrink_fp8_wrapper(
    input_ptr,
    lora_ptr,
    out_ptr,
    a_scale_ptr,
    b_scale_ptr,
    N,
    K,
    M_LEN,
    input_d0_stride,
    input_d1_stride,
    lora_d0_stride,
    lora_d1_stride,
    lora_d2_stride,
    b_scale_l_stride,
    output_d1_stride,
    output_d2_stride,
    scaling,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    ram = tl.arange(0, BLOCK_M)
    do_shrink_kernel_fp8(
        0,          # pid_n
        0,          # pid_sk
        0,          # slice_id
        1,          # lora_index
        input_ptr,
        lora_ptr,
        out_ptr,
        a_scale_ptr,
        b_scale_ptr,
        N,
        K,
        M_LEN,
        ram,
        input_d0_stride,
        input_d1_stride,
        lora_d0_stride,
        lora_d1_stride,
        lora_d2_stride,
        0,              # a_scale_m_stride (tensor-wise)
        0,              # a_scale_k_stride
        b_scale_l_stride,
        0,              # b_scale_n_stride
        0,              # b_scale_k_stride
        0,              # output_d0_stride (unused SLICE_NUM==1)
        output_d1_stride,
        output_d2_stride,
        scaling,
        0,              # group_n -> tensor-wise
        0,              # group_k -> tensor-wise
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        1,              # SPLIT_K
        1,              # SLICE_NUM
        False,          # USE_GDC
        True,           # use_fp8_w8a8
        False,          # per_channel_quant
        False,          # launch_pdl
    )


def pytorch_ref(x, lora_a):
    xf = x.float() * A_SCALE
    af = lora_a[LORA_INDEX].float() * B_SCALES[LORA_INDEX]
    return SCALING * (xf @ af.t())


def kernel_impl(x, lora_a):
    out = torch.zeros(M, RANK, dtype=torch.float32, device=x.device)
    a_scale = torch.tensor([A_SCALE], dtype=torch.float32, device=x.device)
    b_scale = torch.tensor(B_SCALES, dtype=torch.float32, device=x.device)
    _shrink_fp8_wrapper[(1,)](
        x,
        lora_a,
        out,
        a_scale,
        b_scale,
        RANK,
        HIDDEN,
        M,
        x.stride(0),
        x.stride(1),
        lora_a.stride(0),
        lora_a.stride(1),
        lora_a.stride(2),
        b_scale.stride(0),
        out.stride(0),
        out.stride(1),
        SCALING,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
    )
    return out


def _log(text):
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


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(X, LORA_A)
        kernel_out = kernel_impl(X, LORA_A)
        ref_cpu = ref_out.cpu()
        ker_cpu = kernel_out.cpu()
        torch.testing.assert_close(ker_cpu, ref_cpu, rtol=1e-3, atol=1e-3)
        diff = (ker_cpu - ref_cpu).abs()
        stats = {
            "input_shapes": [tuple(X.shape), tuple(LORA_A.shape)],
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(X.dtype),
            "device": str(X.device),
            "scaling": SCALING,
            "a_scale": A_SCALE,
            "b_scale": B_SCALES[LORA_INDEX],
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": (diff.max() / (ref_cpu.abs().max() + 1e-8)).item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(X, LORA_A))
        kern_stats = _bench(lambda: kernel_impl(X, LORA_A))
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
        print("FAILURE\n", error_text)
    finally:
        lines = [
            f"{timestamp}\n",
            "Kernel: do_shrink_kernel_fp8 (wrapped, tensor-wise)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n",
        ]
        if status == "SUCCESS":
            for k, v in stats.items():
                lines.append(f"- {k}: {v}\n")
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
            lines.append("Error:\n" + error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    main()
