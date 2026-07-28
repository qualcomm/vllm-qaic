"""
Standalone QAIC validation for `make_tensor_descriptor` (fla TMA shim).

Source under test:
vllm/model_executor/layers/fla/ops/op.py
  - make_tensor_descriptor  (a compiler shim, NOT a numeric kernel)

The module binds `make_tensor_descriptor` conditionally on Triton's
tensor-descriptor (TMA) API availability:
  * Triton 3.3.x: alias of triton.language._experimental_make_tensor_descriptor
  * Triton 3.4.x+: alias of triton.language.make_tensor_descriptor
  * otherwise (TMA unsupported): a @triton.jit fallback that simply
    `return None` -- a no-op TMA-descriptor constructor whose only purpose is to
    keep the Triton compiler happy on backends without TMA.

SHIM -- NO NUMERICAL COMPARISON. This symbol constructs a tensor-descriptor
*handle* (or None on the fallback path); it performs no arithmetic and has no
numeric output, so there is nothing to compare against a PyTorch reference. We
therefore validate the DOCUMENTED behavior instead:
  * fallback path: it is a callable @triton.jit function; invoking it inside a
    wrapper kernel (which discards the None result) compiles/runs as a no-op.
  * real-API path: it is the bound Triton descriptor constructor (callable).
`kernel_impl` genuinely invokes the shim on the fallback path via a wrapper
kernel. There is intentionally no assert_close / mismatch-count numeric check.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_make_tensor_descriptor.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fla/ops/op.py"

DEVICE = "qaic"

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from vllm.triton_utils import tl, triton
from vllm.model_executor.layers.fla.ops.op import make_tensor_descriptor

torch.manual_seed(42)

# ---- Global shared inputs -------------------------------------------------
# A small backing tensor whose bytes the wrapper kernel copies through, proving
# the kernel body (which also invokes the no-op shim) executes as expected.
N = 16
SRC = torch.arange(N, dtype=torch.int32, device=DEVICE)

# Detect whether the bound symbol is the @triton.jit fallback stub (no-op) or a
# real Triton descriptor constructor. The fallback is a triton JITFunction.
_IS_FALLBACK_STUB = hasattr(make_tensor_descriptor, "fn") or hasattr(
    make_tensor_descriptor, "run"
)


# Wrapper kernel that invokes the fallback shim (result discarded) and performs
# a trivial passthrough copy so we can confirm the kernel body executed.
@triton.jit
def _make_desc_wrapper(src_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    # Invoke the no-op TMA-descriptor shim; it returns None by design.
    _desc = make_tensor_descriptor(
        src_ptr, [N], [1], [BLOCK]
    )
    data = tl.load(src_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, data, mask=mask)


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


def kernel_impl(src):
    """Invoke the shim. On the fallback path we run it inside a wrapper kernel
    (the no-op result is discarded) and return the passthrough copy so we can
    confirm the kernel body ran. On the real-API path we cannot legally build a
    descriptor off-kernel, so we return None and validate callability only."""
    if _IS_FALLBACK_STUB:
        out = torch.empty_like(src)
        BLOCK = 1 << (max(N, 1) - 1).bit_length()
        _make_desc_wrapper[(1,)](src, out, N=N, BLOCK=BLOCK)
        return out
    return None


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # Behavioral validation (this is a compiler shim, not numeric math).
        assert callable(make_tensor_descriptor), (
            "make_tensor_descriptor must be callable"
        )
        out = kernel_impl(SRC)
        if _IS_FALLBACK_STUB:
            variant = "fallback @triton.jit stub (returns None, no-op)"
            # Confirm the wrapper kernel body executed (passthrough copy).
            mism = int((out.cpu() != SRC.cpu()).sum().item())
            assert mism == 0, f"wrapper passthrough mismatch count={mism}"
            body_ran = True
        else:
            variant = "real Triton descriptor constructor (bound API)"
            body_ran = False  # cannot construct a descriptor off-kernel.
        stats = {
            "variant": variant,
            "is_fallback_stub": _IS_FALLBACK_STUB,
            "device": DEVICE,
            "wrapper_body_ran": body_ran,
            "numeric_comparison": "N/A (compiler shim, no numeric output)",
        }
        kern_stats = _bench(lambda: kernel_impl(SRC))
        stats["kernel_latency_ms"] = kern_stats
        status = "SUCCESS"
        print("SUCCESS")
        print(stats)
        print(f"Kernel latency (ms): avg={kern_stats['avg_ms']:.4f}")
    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE")
        print(error_text)
    finally:
        lines = [
            f"{timestamp}\n",
            "Kernel: make_tensor_descriptor (fla TMA shim)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"is_fallback_stub: {_IS_FALLBACK_STUB}\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append(f"variant: {stats['variant']}\n")
            lines.append(f"device: {stats['device']}\n")
            lines.append(f"wrapper_body_ran: {stats['wrapper_body_ran']}\n")
            lines.append(f"numeric_comparison: {stats['numeric_comparison']}\n")
            lines.append(
                "Note: shim with no numerical output; validated documented "
                "no-op/handle behavior rather than numeric math.\n"
            )
            if "kernel_latency_ms" in stats:
                lines.append("Timing:\n")
                lines.append(f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                             f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                             f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                             f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n")
                lines.append("- Speedup: N/A (no numeric PyTorch reference to compare against)\n")
        else:
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    main()
