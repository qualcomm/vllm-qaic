"""
Standalone QAIC validation for `_get_expert_id`.

Source under test:
vllm/lora/ops/triton_ops/fused_moe_lora_op.py
  - _get_expert_id  (device helper: resolve MoE expert id for the current tile)

`_get_expert_id` is a @triton.jit device helper. Two constexpr-selected modes:
  * naive_block_assignment=True : return expert_ids[pid_m]   (flat table)
  * naive_block_assignment=False: ind = lora_id*stride_el + pid_m;
                                  return load(expert_ids[ind], ind<max_loras*stride_el, -1)
We wrap it in `_expert_id_wrapper` which exercises BOTH modes and stores the
two resolved ids.

INTEGER helper -> EXACT equality + mismatch count.
"""

import datetime
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

from vllm.lora.ops.triton_ops.fused_moe_lora_op import _get_expert_id
from vllm.triton_utils import tl, triton

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_get_expert_id.txt")
KERNEL_FILE_PATH = "vllm/lora/ops/triton_ops/fused_moe_lora_op.py"
DEVICE = "qaic"

torch.manual_seed(42)

# Global shared inputs
MAX_LORAS = 2
STRIDE_EL = 4       # per-lora row length of expert_ids table
PID_M = 1
LORA_ID = 1
# Table laid out as [max_loras, stride_el] flattened (non-naive uses lora_id*stride_el+pid_m)
# and also serves as the flat table for the naive path (indexed by pid_m).
EXPERT_IDS = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32, device=DEVICE)


@triton.jit
def _expert_id_wrapper(
    expert_ids_ptr,
    out_ptr,
    lora_id,
    pid_m,
    stride_el,
    max_loras,
):
    naive_eid = _get_expert_id(expert_ids_ptr, lora_id, pid_m, stride_el, max_loras, True)
    tbl_eid = _get_expert_id(expert_ids_ptr, lora_id, pid_m, stride_el, max_loras, False)
    tl.store(out_ptr + 0, naive_eid)
    tl.store(out_ptr + 1, tbl_eid)


def pytorch_ref():
    naive_eid = int(EXPERT_IDS[PID_M].item())
    ind = LORA_ID * STRIDE_EL + PID_M
    if ind < MAX_LORAS * STRIDE_EL:
        tbl_eid = int(EXPERT_IDS[ind].item())
    else:
        tbl_eid = -1
    return torch.tensor([naive_eid, tbl_eid], dtype=torch.int32, device=DEVICE)


def kernel_impl():
    out = torch.zeros(2, dtype=torch.int32, device=DEVICE)
    _expert_id_wrapper[(1,)](EXPERT_IDS, out, LORA_ID, PID_M, STRIDE_EL, MAX_LORAS)
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
        ref_out = pytorch_ref()
        kernel_out = kernel_impl()
        ref_cpu = ref_out.cpu()
        ker_cpu = kernel_out.cpu()
        mism = int((ref_cpu != ker_cpu).sum().item())
        assert mism == 0, f"{mism} mismatches: ref={ref_cpu.tolist()} got={ker_cpu.tolist()}"
        stats = {
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(kernel_out.dtype),
            "device": str(kernel_out.device),
            "mismatches": mism,
            "ref": ref_cpu.tolist(),
            "got": ker_cpu.tolist(),
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
            "Kernel: _get_expert_id (wrapped, naive + table modes)\n",
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
