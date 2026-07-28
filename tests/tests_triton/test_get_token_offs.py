"""
Standalone QAIC validation for `_get_token_offs`.

Source under test:
vllm/lora/ops/triton_ops/fused_moe_lora_op.py
  - _get_token_offs  (device helper: flattened token offsets for a MoE-LoRA tile)

`_get_token_offs` is a @triton.jit device helper returning a BLOCK_SIZE_M
vector of token offsets. Two constexpr-selected modes:
  * naive_block_assignment=True : where(offs==0, pid_m, num_valid_tokens)
        i.e. lane 0 -> pid_m, all other lanes -> num_valid_tokens (masked out)
  * naive_block_assignment=False: offs_token_id = pid_m*BLOCK_SIZE_M + offs;
        token_ind = stride_tl*lora_id + offs_token_id;
        load(sorted_token_ids[token_ind], token_ind<max_loras*stride_tl, 0)
We wrap it in `_token_offs_wrapper` exercising BOTH modes into a [2, BLOCK] out.

INTEGER helper -> EXACT equality + mismatch count.
"""

import datetime
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

from vllm.lora.ops.triton_ops.fused_moe_lora_op import _get_token_offs
from vllm.triton_utils import tl, triton

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_get_token_offs.txt")
KERNEL_FILE_PATH = "vllm/lora/ops/triton_ops/fused_moe_lora_op.py"
DEVICE = "qaic"

torch.manual_seed(42)

# Global shared inputs
BLOCK_SIZE_M = 16
PID_M = 0
LORA_ID = 1
STRIDE_TL = 16          # per-lora row length of sorted_token_ids
MAX_LORAS = 2
NUM_VALID_TOKENS = 8
# sorted_token_ids laid out [max_loras, stride_tl] flattened.
SORTED_TOKEN_IDS = torch.arange(
    MAX_LORAS * STRIDE_TL, dtype=torch.int32, device=DEVICE
)


@triton.jit
def _token_offs_wrapper(
    sorted_token_ids_ptr,
    out_ptr,
    lora_id,
    pid_m,
    stride_tl,
    max_loras,
    num_valid_tokens,
    BLOCK_SIZE_M: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE_M)
    naive = _get_token_offs(
        sorted_token_ids_ptr, lora_id, pid_m, offs, stride_tl, max_loras,
        num_valid_tokens, True, BLOCK_SIZE_M,
    )
    sorted_offs = _get_token_offs(
        sorted_token_ids_ptr, lora_id, pid_m, offs, stride_tl, max_loras,
        num_valid_tokens, False, BLOCK_SIZE_M,
    )
    tl.store(out_ptr + offs, naive)
    tl.store(out_ptr + BLOCK_SIZE_M + offs, sorted_offs)


def pytorch_ref():
    offs = torch.arange(BLOCK_SIZE_M, dtype=torch.int32, device=DEVICE)
    naive = torch.where(
        offs == 0,
        torch.full_like(offs, PID_M),
        torch.full_like(offs, NUM_VALID_TOKENS),
    )
    offs_token_id = PID_M * BLOCK_SIZE_M + offs
    token_ind = STRIDE_TL * LORA_ID + offs_token_id
    valid = token_ind < MAX_LORAS * STRIDE_TL
    gathered = SORTED_TOKEN_IDS[token_ind.clamp(max=MAX_LORAS * STRIDE_TL - 1)]
    sorted_offs = torch.where(valid, gathered, torch.zeros_like(gathered))
    return torch.cat([naive, sorted_offs]).to(torch.int32)


def kernel_impl():
    out = torch.zeros(2 * BLOCK_SIZE_M, dtype=torch.int32, device=DEVICE)
    _token_offs_wrapper[(1,)](
        SORTED_TOKEN_IDS, out, LORA_ID, PID_M, STRIDE_TL, MAX_LORAS,
        NUM_VALID_TOKENS, BLOCK_SIZE_M,
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
            "Kernel: _get_token_offs (wrapped, naive + sorted modes)\n",
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
