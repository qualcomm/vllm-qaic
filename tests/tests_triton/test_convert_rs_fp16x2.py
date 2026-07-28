"""
Standalone QAIC validation for the `convert_rs_fp16x2` Triton device helper.

Source under test:
vllm/model_executor/layers/mamba/ops/mamba_ssm.py
  - convert_rs_fp16x2  (device helper)

`convert_rs_fp16x2` performs *stochastic-rounding* conversion of a pair of
fp32 values to fp16 using the inline PTX instruction `cvt.rs.f16x2.f32`
(packed 2-wide). It takes the fp32 tensor `x` plus a tensor of random 32-bit
integers `rand` used to bias the rounding direction, and returns fp16. It is a
`@triton.jit` device helper, so we wrap it in a tiny standalone kernel.

IMPORTANT COMPARISON NOTE:
  1. The `tl.inline_asm_elementwise` PTX (`cvt.rs.f16x2.f32`) is NVIDIA-specific
     and will NOT compile/run on QAIC. This test is COMPILE-ONLY per directive;
     the wrapper is written to be syntactically valid.
  2. Stochastic rounding is *nondeterministic* by design: for a given fp32
     value the result is one of the two nearest fp16 values, chosen with a
     probability proportional to the distance. Therefore a bit-exact reference
     is impossible. The PyTorch reference below is the DETERMINISTIC
     round-to-nearest cast (`.half()`). The comparison validates that the
     kernel output lies within one fp16 ULP of the round-to-nearest result
     (i.e. it is one of the two fp16 values bracketing the fp32 input), NOT
     that it is bit-identical.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import convert_rs_fp16x2
from vllm.triton_utils import tl, triton

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_convert_rs_fp16x2.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/mamba/ops/mamba_ssm.py"

DEVICE = "qaic"
N = 256  # must be even (packed 2-wide)
BLOCK = 256

torch.manual_seed(42)
X = torch.randn(N, dtype=torch.float32, device=DEVICE)
# Random 32-bit integers driving the stochastic-rounding bias.
RAND = torch.randint(
    0, 2**31 - 1, (N,), dtype=torch.int32, device=DEVICE
)


@triton.jit
def _convert_rs_wrap_kernel(x_ptr, rand_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    rand = tl.load(rand_ptr + offs, mask=mask, other=0)
    o = convert_rs_fp16x2(x, rand)
    tl.store(o_ptr + offs, o, mask=mask)


def pytorch_ref(x):
    """Pure PyTorch reference: DETERMINISTIC round-to-nearest fp16 cast.

    This is NOT what the kernel computes bit-for-bit (the kernel uses
    stochastic rounding), but it is the round-to-nearest anchor used to bound
    the stochastically-rounded output to within one fp16 ULP.
    """
    return x.half()


def kernel_impl(x, rand):
    out = torch.empty(N, dtype=torch.float16, device=x.device)
    grid = (triton.cdiv(N, BLOCK),)
    _convert_rs_wrap_kernel[grid](x, rand, out, N, BLOCK=BLOCK)
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
        ref_out = pytorch_ref(X)  # round-to-nearest fp16
        kernel_out = kernel_impl(X, RAND)  # stochastic-rounded fp16

        ref_cpu = ref_out.cpu().to(torch.float32)
        ker_cpu = kernel_out.cpu().to(torch.float32)

        # Stochastic rounding: result must be one of the two fp16 values
        # bracketing X. Bound the deviation from round-to-nearest by one fp16
        # ULP at each value's magnitude (2^-10 relative for normal fp16).
        x_cpu = X.cpu().to(torch.float32)
        ulp = torch.maximum(
            x_cpu.abs() * (2.0**-10), torch.full_like(x_cpu, 6e-8)
        )
        diff = (ker_cpu - x_cpu).abs()
        within_ulp = bool((diff <= (ulp + 1e-6)).all())
        assert within_ulp, "stochastic-rounded value deviates > 1 fp16 ULP"

        nearest_diff = (ker_cpu - ref_cpu).abs()
        stats = {
            "input_shape": tuple(X.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(X.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(X.device),
            "max_abs_diff_vs_fp32": diff.max().item(),
            "mean_abs_diff_vs_fp32": diff.mean().item(),
            "max_abs_diff_vs_round_nearest": nearest_diff.max().item(),
            "within_one_fp16_ulp": within_ulp,
            "comparison": "within-1-ULP (stochastic rounding, not bit-exact)",
        }
        pt_stats = _bench(lambda: pytorch_ref(X))
        kern_stats = _bench(lambda: kernel_impl(X, RAND))
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
            "Kernel: convert_rs_fp16x2 (device helper, inline PTX)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            "Note: inline PTX cvt.rs.f16x2.f32 is NVIDIA-only; compile-only "
            "test. Stochastic rounding compared within 1 fp16 ULP.\n",
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
