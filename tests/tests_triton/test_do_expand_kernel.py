"""
Standalone QAIC validation for `do_expand_kernel`.

Source under test:
vllm/lora/ops/triton_ops/kernel_utils.py
  - do_expand_kernel  (LoRA expand rank->hidden matmul, one CTA/slice/lora)

`do_expand_kernel` is a @triton.jit device helper invoked inside
`_lora_expand_kernel`. For the rows identified by `ram`, it computes
  out[row, :] = input[row, :] @ B[lora_index].T
where B (the LoRA "B"/expand weight) has shape [num_lora, hidden, rank] and
input (the shrink output) has shape [M, rank]. It optionally adds into the
existing output (ADD_INPUTS). We wrap it in a tiny kernel `_expand_wrapper`
that supplies ram = arange(M) (identity token mapping) and a single slice.

Reference: pure PyTorch shrink_out @ B[lora].T (+ existing output if ADD_INPUTS).
FLOAT compare.
"""

import datetime
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

from vllm.lora.ops.triton_ops.kernel_utils import do_expand_kernel
from vllm.triton_utils import tl, triton

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_do_expand_kernel.txt")
KERNEL_FILE_PATH = "vllm/lora/ops/triton_ops/kernel_utils.py"
DEVICE = "qaic"

torch.manual_seed(42)

# Global shared inputs
M = 8           # num tokens/rows
RANK = 8        # K dimension
HIDDEN = 16     # N dimension
NUM_LORA = 2
LORA_INDEX = 1
ADD_INPUTS = False

BLOCK_M = 8
BLOCK_N = 16
BLOCK_K = 8
EVEN_K = RANK % BLOCK_K == 0

SHRINK_OUT = torch.randn(M, RANK, dtype=torch.float32, device=DEVICE)
# LoRA B weight: [num_lora, hidden, rank]
LORA_B = torch.randn(NUM_LORA, HIDDEN, RANK, dtype=torch.float32, device=DEVICE)


@triton.jit
def _expand_wrapper(
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    M_LEN,
    input_d1_stride,
    input_d2_stride,
    ls_d0,
    ls_d1,
    ls_d2,
    output_d0_stride,
    output_d1_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
):
    ram = tl.arange(0, BLOCK_M)
    do_expand_kernel(
        0,              # pid_n
        1,              # lora_index (LORA_INDEX)
        0,              # slice_id
        input_ptr,
        lora_ptr,
        out_ptr,
        N,
        K,
        M_LEN,
        ram,
        0,              # slice_start_loc (integer, SLICE_NUM==1)
        0,              # input_d0_stride (unused for SLICE_NUM==1)
        input_d1_stride,
        input_d2_stride,
        ls_d0,
        ls_d1,
        ls_d2,
        output_d0_stride,
        output_d1_stride,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        True,           # SAME_STRIDE
        1,              # SLICE_NUM
        EVEN_K,
        False,          # CAST_TYPE
        ADD_INPUTS,
        False,          # USE_GDC
    )


def pytorch_ref(shrink_out, lora_b, existing=None):
    out = shrink_out @ lora_b[LORA_INDEX].t()  # [M, hidden]
    if ADD_INPUTS and existing is not None:
        out = out + existing
    return out


def kernel_impl(shrink_out, lora_b, existing=None):
    out = torch.zeros(M, HIDDEN, dtype=torch.float32, device=shrink_out.device)
    if ADD_INPUTS and existing is not None:
        out.copy_(existing)
    _expand_wrapper[(1,)](
        shrink_out,
        lora_b,
        out,
        HIDDEN,
        RANK,
        M,
        shrink_out.stride(0),
        shrink_out.stride(1),
        lora_b.stride(0),
        lora_b.stride(1),
        lora_b.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        ADD_INPUTS,
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
        ref_out = pytorch_ref(SHRINK_OUT, LORA_B)
        kernel_out = kernel_impl(SHRINK_OUT, LORA_B)
        ref_cpu = ref_out.cpu()
        ker_cpu = kernel_out.cpu()
        torch.testing.assert_close(ker_cpu, ref_cpu, rtol=1e-3, atol=1e-3)
        diff = (ker_cpu - ref_cpu).abs()
        stats = {
            "input_shapes": [tuple(SHRINK_OUT.shape), tuple(LORA_B.shape)],
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(SHRINK_OUT.dtype),
            "device": str(SHRINK_OUT.device),
            "add_inputs": ADD_INPUTS,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": (diff.max() / (ref_cpu.abs().max() + 1e-8)).item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(SHRINK_OUT, LORA_B))
        kern_stats = _bench(lambda: kernel_impl(SHRINK_OUT, LORA_B))
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
            "Kernel: do_expand_kernel (wrapped)\n",
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
