"""
Standalone QAIC validation for `_fwd_kernel_stage2`.

Source under test:
vllm/v1/attention/ops/triton_decode_attention.py
  - _fwd_kernel_stage2          (split-KV flash-decoding stage 2: reduces the
    per-split partial outputs + per-split log-sum-exp produced by stage 1 into
    the final decode attention output, using a numerically-stable LSE combine)
  - _decode_softmax_reducev_fwd (Python launcher used here)

Stage 1 writes, for each (batch, head, split), a per-split
softmax-normalized value ``tv = Mid_O[b,h,split,:Lv]`` and a per-split
log-sum-exp ``tlogic = Mid_O[b,h,split,Lv]``. Stage 2 combines the splits:
    m       = max_s tlogic_s
    scale_s = exp(tlogic_s - m)
    out     = sum_s scale_s * tv_s / sum_s scale_s
    lse     = m + log(sum_s scale_s)
Empty splits (split_kv_end <= split_kv_start) contribute nothing.

Note: the already-implemented ``test_tq_decode_stage1.py`` reuses this same
``_fwd_kernel_stage2`` internally for its stage-2 reduction; here we exercise
it in isolation via ``_decode_softmax_reducev_fwd`` with synthetic per-split
partials (data preparation, not a kernel call).

Config: single request, single head, HEAD_DIM=32, NUM_KV_SPLITS=4 (all
splits non-empty for SEQ_LEN=16).

Reference: pure PyTorch LSE-weighted combination across splits.
"""

import datetime
import math
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.attention.ops.triton_decode_attention import (  # noqa: E402
    _decode_softmax_reducev_fwd,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_triton_decode_attention_fwd_kernel_stage2.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_decode_attention.py"

DEVICE = "qaic"
BATCH = 1
HEAD = 1
KV_HEAD = 1
HEAD_DIM = 32
SEQ_LEN = 16
NUM_KV_SPLITS = 4  # 16 / 4 = 4 tokens per split -> all splits non-empty

torch.manual_seed(42)

# Synthetic stage-1 partials: Mid_O shape [B, H, NUM_KV_SPLITS, HEAD_DIM + 1].
# [:HEAD_DIM] = per-split normalized value, [HEAD_DIM] = per-split LSE.
MID_O = torch.randn(
    BATCH, HEAD, NUM_KV_SPLITS, HEAD_DIM + 1, dtype=torch.float32, device=DEVICE
)
# q / v_buffer only supply shapes (batch, head, Lv) to the launcher.
Q = torch.randn(BATCH, HEAD, HEAD_DIM, dtype=torch.float32, device=DEVICE)
V_BUFFER = torch.randn(SEQ_LEN, KV_HEAD, HEAD_DIM, dtype=torch.float32, device=DEVICE)
B_SEQLEN = torch.tensor([SEQ_LEN], dtype=torch.int32, device=DEVICE)


def _log(text: str) -> None:
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


def pytorch_ref(mid_o):
    """Pure PyTorch LSE-weighted combination across splits (matches source).

    Returns final output of shape [BATCH, HEAD, HEAD_DIM].
    """
    mo = mid_o.float().cpu()
    tv = mo[..., :HEAD_DIM]  # [B, H, S, D]
    tlogic = mo[..., HEAD_DIM]  # [B, H, S]

    kv_len_per_split = math.ceil(SEQ_LEN / NUM_KV_SPLITS)
    active = torch.zeros(NUM_KV_SPLITS, dtype=torch.bool)
    for s in range(NUM_KV_SPLITS):
        start = kv_len_per_split * s
        end = min(start + kv_len_per_split, SEQ_LEN)
        active[s] = end > start

    neg_inf = torch.tensor(float("-inf"))
    tl_masked = torch.where(active.view(1, 1, -1), tlogic, neg_inf)
    m = tl_masked.max(dim=-1, keepdim=True).values  # [B, H, 1]
    scale = torch.where(
        active.view(1, 1, -1), torch.exp(tl_masked - m), torch.zeros_like(tlogic)
    )  # [B, H, S]
    e_sum = scale.sum(dim=-1)  # [B, H]
    acc = (scale.unsqueeze(-1) * tv).sum(dim=-2)  # [B, H, D]
    return acc / e_sum.unsqueeze(-1)


def kernel_impl(mid_o, q, v_buffer, b_seqlen):
    """Kernel wrapper: launch only."""
    o = torch.zeros(BATCH, HEAD, HEAD_DIM, dtype=torch.float32, device=DEVICE)
    lse = torch.zeros(BATCH, HEAD, dtype=torch.float32, device=DEVICE)
    _decode_softmax_reducev_fwd(
        mid_o,
        q,
        o,
        lse,
        v_buffer,
        b_seqlen,
        NUM_KV_SPLITS,
    )
    return o


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(MID_O)
        kernel_out = kernel_impl(MID_O, Q, V_BUFFER, B_SEQLEN)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "mid_o_shape": tuple(MID_O.shape),
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(MID_O.dtype),
            "device": str(MID_O.device),
            "num_kv_splits": NUM_KV_SPLITS,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": (diff.max() / (ref_cpu.abs().max() + 1e-8)).item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(MID_O))
        kern_stats = _bench(lambda: kernel_impl(MID_O, Q, V_BUFFER, B_SEQLEN))
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
        print("FAILURE\n" + error_text)
    finally:
        lines = [
            f"{timestamp}\n",
            "Kernel: _fwd_kernel_stage2 (split-KV decode stage 2 reduce)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- mid_o shape: {stats['mid_o_shape']}\n",
                f"- num_kv_splits: {stats['num_kv_splits']}\n",
                f"- dtype: {stats['dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- out shape: {stats['output_shape']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
                f"- rel_err: {stats['rel_err']}\n",
            ]
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
            lines += ["Error:\n", error_text + "\n"]
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
