"""
Standalone QAIC validation for `_get_c_ptrs`.

Source under test:
vllm/lora/ops/triton_ops/fused_moe_lora_op.py
  - _get_c_ptrs  (device helper: output pointers into the MoE-LoRA C tensor)

`_get_c_ptrs` is a @triton.jit device helper that RETURNS A POINTER TENSOR
(BLOCK_SIZE_M x BLOCK_N). Two constexpr-selected modes:
  * sort_c=True  (token-sorted layout for TMA):
        offs_token_id = pid_m*BLOCK_SIZE_M + offs
        c = cur_c + lora_id*EM*stride_cm + stride_cm*offs_token_id[:,None]
              + stride_cn*offs_cn[None,:]
  * sort_c=False (original prompt order):
        c = cur_c + stride_cm*offs_token[:,None] + stride_cn*offs_cn[None,:]

POINTER->OFFSET VALIDATION APPROACH:
Because the helper returns raw device addresses (not comparable across a
PyTorch reference), the wrapper reconstructs the *element offset* each
returned pointer encodes: it casts the returned pointer and the base pointer
to int64, subtracts, and divides by the element size (4 bytes for float32).
That element offset is exactly the integer arithmetic the helper performs, so
we validate the offset math (not the absolute addresses). We never dereference
the (possibly OOB) computed pointers -- only their integer value is used.
The base tensor `C_BASE` can therefore be size 1.

INTEGER offsets -> EXACT equality + mismatch count.
"""

import datetime
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

from vllm.lora.ops.triton_ops.fused_moe_lora_op import _get_c_ptrs
from vllm.triton_utils import tl, triton

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_get_c_ptrs.txt")
KERNEL_FILE_PATH = "vllm/lora/ops/triton_ops/fused_moe_lora_op.py"
DEVICE = "qaic"

torch.manual_seed(42)

# Global shared inputs
BLOCK_SIZE_M = 8
BLOCK_N = 8
EM = 16
LORA_ID = 1
PID_M = 1
STRIDE_CM = 32
STRIDE_CN = 1
ELEM_SIZE = 4  # float32
OFFS_CN = torch.arange(BLOCK_N, dtype=torch.int32, device=DEVICE)
# arbitrary token ids used by the sort_c=False path
OFFS_TOKEN = (torch.arange(BLOCK_SIZE_M, dtype=torch.int32, device=DEVICE) + 2)
C_BASE = torch.zeros(1, dtype=torch.float32, device=DEVICE)


@triton.jit
def _c_ptrs_wrapper(
    cur_c_ptr,
    offs_token_ptr,
    offs_cn_ptr,
    out_sort_ptr,
    out_nosort_ptr,
    lora_id,
    pid_m,
    stride_cm,
    stride_cn,
    elem_size,
    EM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(offs_token_ptr + offs)
    offs_cn = tl.load(offs_cn_ptr + tl.arange(0, BLOCK_N))

    base_i = cur_c_ptr.to(tl.int64)

    c_sort = _get_c_ptrs(
        cur_c_ptr, lora_id, pid_m, offs, offs_token, offs_cn,
        stride_cm, stride_cn, EM, BLOCK_SIZE_M, True,
    )
    c_nosort = _get_c_ptrs(
        cur_c_ptr, lora_id, pid_m, offs, offs_token, offs_cn,
        stride_cm, stride_cn, EM, BLOCK_SIZE_M, False,
    )

    off_sort = (c_sort.to(tl.int64) - base_i) // elem_size
    off_nosort = (c_nosort.to(tl.int64) - base_i) // elem_size

    row = tl.arange(0, BLOCK_SIZE_M)[:, None]
    col = tl.arange(0, BLOCK_N)[None, :]
    lin = row * BLOCK_N + col
    tl.store(out_sort_ptr + lin, off_sort)
    tl.store(out_nosort_ptr + lin, off_nosort)


def pytorch_ref():
    offs = torch.arange(BLOCK_SIZE_M, dtype=torch.int64, device=DEVICE)
    offs_token = OFFS_TOKEN.to(torch.int64)
    offs_cn = OFFS_CN.to(torch.int64)

    offs_token_id = PID_M * BLOCK_SIZE_M + offs
    sort = (
        LORA_ID * EM * STRIDE_CM
        + STRIDE_CM * offs_token_id[:, None]
        + STRIDE_CN * offs_cn[None, :]
    )
    nosort = STRIDE_CM * offs_token[:, None] + STRIDE_CN * offs_cn[None, :]
    return sort, nosort


def kernel_impl():
    out_sort = torch.zeros(BLOCK_SIZE_M, BLOCK_N, dtype=torch.int64, device=DEVICE)
    out_nosort = torch.zeros(BLOCK_SIZE_M, BLOCK_N, dtype=torch.int64, device=DEVICE)
    _c_ptrs_wrapper[(1,)](
        C_BASE,
        OFFS_TOKEN,
        OFFS_CN,
        out_sort,
        out_nosort,
        LORA_ID,
        PID_M,
        STRIDE_CM,
        STRIDE_CN,
        ELEM_SIZE,
        EM,
        BLOCK_SIZE_M,
        BLOCK_N,
    )
    return out_sort, out_nosort


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
        ref_sort, ref_nosort = pytorch_ref()
        ker_sort, ker_nosort = kernel_impl()
        rs, rn = ref_sort.cpu(), ref_nosort.cpu()
        ks, kn = ker_sort.cpu(), ker_nosort.cpu()
        mism_sort = int((rs != ks).sum().item())
        mism_nosort = int((rn != kn).sum().item())
        assert mism_sort == 0, f"sort_c offsets: {mism_sort} mismatches"
        assert mism_nosort == 0, f"prompt-order offsets: {mism_nosort} mismatches"
        stats = {
            "output_shape": tuple(ker_sort.shape),
            "dtype": "int64 (element offsets)",
            "device": str(ker_sort.device),
            "mismatches_sort_c": mism_sort,
            "mismatches_prompt_order": mism_nosort,
        }
        pt_stats = _bench(lambda: pytorch_ref())
        kern_stats = _bench(lambda: kernel_impl())
        speedup = kern_stats["avg_ms"] / pt_stats["avg_ms"] if pt_stats["avg_ms"] > 0 else float("nan")
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
            "Kernel: _get_c_ptrs (wrapped; pointer->element-offset validation)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n",
        ]
        if status == "SUCCESS":
            for k, v in stats.items():
                lines.append(f"- {k}: {v}\n")
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
            lines.append("Error:\n" + error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    main()
