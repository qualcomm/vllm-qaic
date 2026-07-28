"""
Standalone QAIC validation for `do_shrink_kernel`.

Source under test:
vllm/lora/ops/triton_ops/kernel_utils.py
  - do_shrink_kernel  (LoRA shrink hidden->rank matmul, one CTA/slice/lora)

`do_shrink_kernel` is a @triton.jit device helper invoked inside
`_lora_shrink_kernel`. For the rows identified by `ram`, it computes
  out[row, :] = scaling * (x[row, :] @ A[lora_index].T)
where A (the LoRA "A"/shrink weight) has shape [num_lora, rank(N), hidden(K)]
and x has shape [M, hidden]. With SPLIT_K==1 it stores; with SPLIT_K>1 it
atomic-adds. We wrap it (SPLIT_K==1, SLICE_NUM==1) with ram = arange(M).

Reference: pure PyTorch scaling * (x @ A[lora].T). FLOAT compare.
"""

import datetime
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

from vllm.lora.ops.triton_ops.kernel_utils import do_shrink_kernel
from vllm.triton_utils import tl, triton

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_do_shrink_kernel.txt")
KERNEL_FILE_PATH = "vllm/lora/ops/triton_ops/kernel_utils.py"
DEVICE = "qaic"

torch.manual_seed(42)

# Global shared inputs
M = 8            # num tokens/rows
HIDDEN = 16      # K dimension
RANK = 8         # N dimension
NUM_LORA = 2
LORA_INDEX = 1
SCALING = 0.5

BLOCK_M = 8
BLOCK_N = 8
BLOCK_K = 16
EVEN_K = HIDDEN % BLOCK_K == 0

X = torch.randn(M, HIDDEN, dtype=torch.float32, device=DEVICE)
# LoRA A weight: [num_lora, rank, hidden]
LORA_A = torch.randn(NUM_LORA, RANK, HIDDEN, dtype=torch.float32, device=DEVICE)


@triton.jit
def _shrink_wrapper(
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    M_LEN,
    input_d0_stride,
    input_d1_stride,
    lora_d0_stride,
    lora_d1_stride,
    lora_d2_stride,
    output_d0_stride,
    output_d1_stride,
    output_d2_stride,
    scaling,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    ram = tl.arange(0, BLOCK_M)
    do_shrink_kernel(
        0,          # pid_n
        0,          # pid_sk
        0,          # slice_id
        1,          # lora_index (LORA_INDEX)
        input_ptr,
        lora_ptr,
        out_ptr,
        N,
        K,
        M_LEN,
        ram,
        input_d0_stride,
        input_d1_stride,
        lora_d0_stride,
        lora_d1_stride,
        lora_d2_stride,
        output_d0_stride,
        output_d1_stride,
        output_d2_stride,
        scaling,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        1,          # SPLIT_K
        1,          # SLICE_NUM
        False,      # USE_GDC
    )


def pytorch_ref(x, lora_a):
    return SCALING * (x @ lora_a[LORA_INDEX].t())  # [M, rank]


def kernel_impl(x, lora_a):
    # output_tensor conceptual shape: [slice, M, rank]; SLICE_NUM==1 -> [M, rank]
    out = torch.zeros(M, RANK, dtype=torch.float32, device=x.device)
    _shrink_wrapper[(1,)](
        x,
        lora_a,
        out,
        RANK,
        HIDDEN,
        M,
        x.stride(0),
        x.stride(1),
        lora_a.stride(0),
        lora_a.stride(1),
        lora_a.stride(2),
        0,               # output_d0_stride (unused for SLICE_NUM==1)
        out.stride(0),   # output_d1_stride -> row
        out.stride(1),   # output_d2_stride -> col
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
            "Kernel: do_shrink_kernel (wrapped)\n",
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
