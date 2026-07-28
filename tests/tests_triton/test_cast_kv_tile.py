"""
Standalone QAIC validation for the `_cast_kv_tile` @triton.jit device helper.

Source under test:
vllm/v1/attention/ops/triton_unified_attention.py
  - _cast_kv_tile(data, Q, tensor_scale, KV_QUANT_MODE)

`_cast_kv_tile` casts a loaded KV cache tile to the query dtype, optionally
dequantizing when the KV cache is FP8-per-tensor quantized. It is a device
helper (called from inside `kernel_unified_attention`), so it cannot be
launched directly; we wrap it in a minimal `@triton.jit` launcher kernel
(`_cast_launcher`) that loads a tile, calls the helper, and stores the result.

Exact source behaviour:
    KV_QUANT_MODE == 1 (FP8_PER_TENSOR):
        if Q is fp8: return data.to(Q.dtype)                 # scale folded elsewhere
        else:        return (data.to(f32) * load(scale)).to(Q.dtype)
    otherwise (NONE / INT8 / FP8 per-token-head):
        return data.to(Q.dtype)                              # plain cast

Approach (documented per task instructions):
  We exercise the NON-quantized cast path (KV_QUANT_MODE == 0). The FP8
  per-tensor branch requires materialising an fp8 KV tile, and in-kernel FP8
  casts (`tl.float8e4nv` etc.) are not supported by this QAIC/Hexagon Triton
  backend, so we validate the (dominant, always-compiled) plain-cast path.
  With Q and data both float32 this reduces to an identity cast, which is the
  exact semantics of the mode-0 branch.

Reference: pure PyTorch identity cast to the query dtype (float32).
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.triton_utils import tl, triton  # noqa: E402
from vllm.v1.attention.ops.triton_unified_attention import _cast_kv_tile  # noqa: E402

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_cast_kv_tile.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_unified_attention.py"

DEVICE = "qaic"
TILE_SIZE = 16
HEAD_SIZE = 32
N = TILE_SIZE * HEAD_SIZE
KV_QUANT_MODE = 0  # NONE -> plain cast path (see module docstring)

torch.manual_seed(42)
DATA = torch.randn(TILE_SIZE, HEAD_SIZE, dtype=torch.float32, device=DEVICE)
# A single-element query tile only used to establish the target dtype in the
# helper (Q.dtype). One element of the query is enough for the wrapper.
Q_DTYPE_PROBE = torch.zeros(1, dtype=torch.float32, device=DEVICE)
TENSOR_SCALE = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)


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


def pytorch_ref(data):
    """Pure PyTorch reference for the NONE (mode 0) cast path.

    The helper returns ``data.to(Q.dtype)``; with a float32 query this is an
    identity cast, so the reference simply returns the data cast to float32.
    """
    return data.to(torch.float32).clone()


@triton.jit
def _cast_launcher(
    data_ptr,
    q_ptr,
    scale_ptr,
    out_ptr,
    N: tl.constexpr,
    KV_QUANT_MODE: tl.constexpr,
):
    offs = tl.arange(0, N)
    data = tl.load(data_ptr + offs)
    # A dummy scalar load only to give the helper a value whose dtype is the
    # target query dtype (Q.dtype is all the helper reads of Q).
    q = tl.load(q_ptr + 0)
    res = _cast_kv_tile(data, q, scale_ptr, KV_QUANT_MODE)
    tl.store(out_ptr + offs, res)


def kernel_impl(data):
    """Kernel wrapper: launch only."""
    out = torch.empty_like(data)
    _cast_launcher[(1,)](
        data.reshape(-1),
        Q_DTYPE_PROBE,
        TENSOR_SCALE,
        out.reshape(-1),
        N=N,
        KV_QUANT_MODE=KV_QUANT_MODE,
    )
    return out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(DATA)
        kernel_out = kernel_impl(DATA)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        denom = ref_cpu.abs().clamp_min(1e-6)
        stats = {
            "input_shape": tuple(DATA.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(DATA.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(DATA.device),
            "kv_quant_mode": KV_QUANT_MODE,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": (diff / denom).max().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(DATA))
        kern_stats = _bench(lambda: kernel_impl(DATA))
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
            "Kernel: _cast_kv_tile (device helper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Approach notes:\n",
                "- Validated the NONE (KV_QUANT_MODE=0) plain-cast path; the FP8\n",
                "  per-tensor dequant branch needs fp8 casts unsupported on this\n",
                "  QAIC/Hexagon Triton backend.\n\n",
                "Inputs:\n",
                f"- data shape: {stats['input_shape']}\n",
                f"- kv_quant_mode: {stats['kv_quant_mode']}\n",
                f"- in dtype: {stats['in_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- out shape: {stats['output_shape']}\n",
                f"- out dtype: {stats['out_dtype']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
                f"- max_rel_err: {stats['rel_err']}\n",
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
