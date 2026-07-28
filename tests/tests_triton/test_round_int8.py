"""
Standalone QAIC validation for the `round_int8` Triton device helper.

Source under test:
vllm/model_executor/layers/quantization/utils/int8_utils.py
  - round_int8  (@triton.jit device helper)

`round_int8(x)` rounds a float to the nearest integer using the platform
libdevice `round` and casts the result to int8. It is used by the INT8
quantization kernels to convert scaled activations into int8 codes. Because it
is a device-side helper (not a launchable kernel), we wrap it in a tiny
standalone `@triton.jit` kernel that loads a float row, applies `round_int8`,
and stores the int8 result.

Rounding convention: libdevice `round` rounds half AWAY from zero, whereas
`torch.round` rounds half to EVEN. To keep the comparison unambiguous we
generate inputs that avoid exact .5 ties, so both conventions agree, and we
compare int8 codes EXACTLY. Values are also kept within the int8 range so the
cast does not overflow (behaviour on overflow is platform-defined).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_round_int8.txt")
KERNEL_FILE_PATH = (
    "vllm/model_executor/layers/quantization/utils/int8_utils.py"
)

N = 64
DEVICE = "qaic"

_IS_CHILD = os.environ.get("ROUND_INT8_CHILD") == "1"

if _IS_CHILD or __name__ != "__main__":
    import torch

    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm")
    )
    from vllm.model_executor.layers.quantization.utils.int8_utils import (
        round_int8,
    )
    from vllm.triton_utils import tl, triton

    torch.manual_seed(42)
    # Values in [-120, 120], offset by 0.3 to avoid exact .5 ties.
    X = (torch.randint(-120, 121, (N,), dtype=torch.float32, device=DEVICE)
         + 0.3 * torch.sign(torch.randn(N, device=DEVICE)))

    @triton.jit
    def _round_int8_wrapper(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = round_int8(x)
        tl.store(out_ptr + cols, out, mask=mask)


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


def pytorch_ref(x):
    """Pure PyTorch round-to-nearest then cast to int8.

    torch.round is round-half-to-even; inputs avoid ties so the result matches
    the libdevice round-half-away-from-zero convention used by the kernel.
    """
    x = x.cpu()
    return torch.round(x).clamp(-128, 127).to(torch.int8)


def kernel_impl(x):
    out = torch.empty(N, dtype=torch.int8, device=x.device)
    BLOCK = triton.next_power_of_2(N)
    _round_int8_wrapper[(1,)](x, out, N, BLOCK=BLOCK)
    return out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(X)
        kern_out = kernel_impl(X)

        ref_out = ref_out.cpu()
        kern_out = kern_out.cpu()

        exact = bool(torch.equal(ref_out, kern_out))
        assert exact, "int8 rounding mismatch"

        stats = {
            "input_shape": tuple(X.shape),
            "output_shape": tuple(kern_out.shape),
            "in_dtype": str(X.dtype),
            "out_dtype": str(kern_out.dtype),
            "device": str(X.device),
            "exact_match": exact,
            "max_abs_diff": int((ref_out.to(torch.int32)
                                 - kern_out.to(torch.int32)).abs().max().item()),
        }
        pt_stats = _bench(lambda: pytorch_ref(X))
        kern_stats = _bench(lambda: kernel_impl(X))
        speedup = kern_stats["avg_ms"] / pt_stats["avg_ms"] if pt_stats["avg_ms"] > 0 else float("nan")
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
            "Kernel: round_int8\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- x shape: {stats['input_shape']}\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- out shape: {stats['output_shape']}\n")
            lines.append(f"- out dtype: {stats['out_dtype']}\n")
            lines.append(f"- exact int8 match: {stats['exact_match']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
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
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))

    return status


def _run_with_crash_guard():
    import subprocess

    if os.environ.get("ROUND_INT8_CHILD") == "1":
        sys.exit(0 if main() == "SUCCESS" else 1)

    env = dict(os.environ, ROUND_INT8_CHILD="1")
    proc = subprocess.run([sys.executable, __file__], env=env)
    if proc.returncode < 0:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _log(
            f"{timestamp}\n"
            "Kernel: round_int8\n"
            f"Kernel file: {KERNEL_FILE_PATH}\n"
            f"Device target: QAIC (device='{DEVICE}')\n"
            "Status: FAILURE\n\n"
            "Error:\n"
            f"Child killed by signal (exit {proc.returncode}) during "
            "Triton->Hexagon compile/execution.\n"
            "\n------------------------------------\n\n"
        )
    sys.exit(proc.returncode if proc.returncode >= 0 else 1)


if __name__ == "__main__":
    _run_with_crash_guard()
