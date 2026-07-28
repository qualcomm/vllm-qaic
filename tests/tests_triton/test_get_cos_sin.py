"""
Standalone QAIC validation for the `_get_cos_sin` @triton.jit device helper.

Source under test:
vllm/models/deepseek_v4/common/ops/fused_indexer_q.py
  - _get_cos_sin(cos_sin_cache_ptr, cos_sin_cache_stride, pos, HALF_ROT_DIM)

`_get_cos_sin` is a device-side helper (called from inside the fused indexer-Q
RoPE kernels) with no public launcher, so we wrap it in a minimal `@triton.jit`
launcher (`_get_cos_sin_launcher`): one program per token loads that token's
position, calls the helper, and stores the returned cos/sin rows.

Exact source semantics: for a cache row of width 2*HALF_ROT_DIM,
    cos = cache[pos, 0:HALF_ROT_DIM]          (cast to fp32)
    sin = cache[pos, HALF_ROT_DIM:2*HALF_ROT_DIM]  (cast to fp32)
i.e. a gather of the first / second half of the selected cache row.

Config tested: num_positions=32, HALF_ROT_DIM=32 (cache width 64), fp32.
Reference: pure PyTorch row gather + halves split.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_get_cos_sin.txt")
KERNEL_FILE_PATH = "vllm/models/deepseek_v4/common/ops/fused_indexer_q.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.triton_utils import tl, triton  # noqa: E402
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import _get_cos_sin  # noqa: E402

torch.manual_seed(42)

NUM_POS = 32          # rows in the cache
NUM_TOKENS = 20       # tokens querying the cache
HALF = 32             # HALF_ROT_DIM
ROT_DIM = 2 * HALF    # cache row width

_ang = torch.randn(NUM_POS, HALF, dtype=torch.float32, device=DEVICE)
COS_SIN_CACHE = torch.cat([torch.cos(_ang), torch.sin(_ang)], dim=-1).contiguous()
POS = torch.randint(0, NUM_POS, (NUM_TOKENS,), dtype=torch.int32, device=DEVICE)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


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


def pytorch_ref(cache, pos, half):
    """Pure PyTorch gather of cos (first half) and sin (second half)."""
    rows = cache[pos.long()].float()
    cos = rows[:, :half]
    sin = rows[:, half:2 * half]
    return cos, sin


@triton.jit
def _get_cos_sin_launcher(
    cache_ptr,
    cache_stride,
    pos_ptr,
    cos_out_ptr,
    sin_out_ptr,
    out_stride,
    HALF_ROT_DIM: tl.constexpr,
):
    tok = tl.program_id(0)
    pos = tl.load(pos_ptr + tok)
    cos, sin = _get_cos_sin(cache_ptr, cache_stride, pos, HALF_ROT_DIM)
    block = tl.arange(0, HALF_ROT_DIM)
    tl.store(cos_out_ptr + tok * out_stride + block, cos)
    tl.store(sin_out_ptr + tok * out_stride + block, sin)


def kernel_impl(cache, pos, half):
    """Kernel launch only."""
    n = pos.shape[0]
    cos_out = torch.empty(n, half, dtype=torch.float32, device=cache.device)
    sin_out = torch.empty(n, half, dtype=torch.float32, device=cache.device)
    _get_cos_sin_launcher[(n,)](
        cache,
        cache.stride(0),
        pos,
        cos_out,
        sin_out,
        cos_out.stride(0),
        HALF_ROT_DIM=half,
    )
    return cos_out, sin_out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_cos, ref_sin = pytorch_ref(COS_SIN_CACHE, POS, HALF)
        ker_cos, ker_sin = kernel_impl(COS_SIN_CACHE, POS, HALF)

        ref_cos_c, ref_sin_c = ref_cos.cpu(), ref_sin.cpu()
        ker_cos_c, ker_sin_c = ker_cos.cpu(), ker_sin.cpu()
        torch.testing.assert_close(ker_cos_c, ref_cos_c, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ker_sin_c, ref_sin_c, rtol=1e-3, atol=1e-3)

        diff_cos = (ker_cos_c - ref_cos_c).abs()
        diff_sin = (ker_sin_c - ref_sin_c).abs()
        max_abs = max(diff_cos.max().item(), diff_sin.max().item())
        mean_abs = (diff_cos.mean().item() + diff_sin.mean().item()) / 2.0
        stats = {
            "input_shape": tuple(COS_SIN_CACHE.shape),
            "output_shape": tuple(ker_cos.shape),
            "in_dtype": str(COS_SIN_CACHE.dtype),
            "out_dtype": str(ker_cos.dtype),
            "device": str(COS_SIN_CACHE.device),
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
        }

        pt_stats = _bench(lambda: pytorch_ref(COS_SIN_CACHE, POS, HALF))
        kern_stats = _bench(lambda: kernel_impl(COS_SIN_CACHE, POS, HALF))
        speedup = (kern_stats["avg_ms"] / pt_stats["avg_ms"]
                   if pt_stats["avg_ms"] > 0 else float("nan"))
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
            "Kernel: _get_cos_sin (device helper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- cache shape: {stats['input_shape']}\n",
                f"- in dtype: {stats['in_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- cos out shape: {stats['output_shape']}\n",
                f"- out dtype: {stats['out_dtype']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
                "Timing:\n",
                f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n",
                f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n",
                f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n",
            ]
        else:
            lines += ["Error:\n", error_text + "\n"]
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
