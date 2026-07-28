"""
Standalone QAIC validation for `_bilinear_pos_embed_kernel`.

Source under test:
vllm/model_executor/models/qwen3_vl.py
  - _bilinear_pos_embed_kernel  (fused bilinear interpolation of a
    (num_grid x num_grid) position-embedding table onto an (h x w) grid, with
    the Qwen spatial-merge block reorder baked into the output index math).
    Launched via the public wrapper `triton_pos_embed_interpolate`.

Exact source semantics (one program per output patch `pid`):
    spatial_idx  = pid % (H*W)
    (spatial-merge de-interleave -> (row, col) in the H x W grid)
    h_frac = row * h_scale;  w_frac = col * w_scale     (scales set by wrapper)
    hf/wf  = floor;  hc/wc = min(floor+1, NUM_GRID-1)
    dh/dw  = frac - floor
    w11 = dh*dw; w10 = dh-w11; w01 = dw-w11; w00 = 1-dh-w01
    out  = w00*E[hf,wf] + w01*E[hf,wc] + w10*E[hc,wf] + w11*E[hc,wc]
The interpolation weights are cast to the output dtype before the MAC (kernel
comment), so we run in fp32 to match precisely.

Config tested: num_grid=8, h=w=8, m_size=2 (spatial merge), t=1,
hidden_dim=32, fp32.
Reference: the eager `pos_embed_interpolate_native` algorithm reimplemented in
pure PyTorch inside this file (no vLLM calls) — same math via torch.linspace +
bilinear gather + spatial-merge permute.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_bilinear_pos_embed_kernel.txt")
KERNEL_FILE_PATH = "vllm/model_executor/models/qwen3_vl.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.model_executor.models.qwen3_vl import (  # noqa: E402
    triton_pos_embed_interpolate,
)

torch.manual_seed(42)

NUM_GRID = 8
H = 8
W = 8
M_SIZE = 2
T = 1
HIDDEN_DIM = 32
DTYPE = torch.float32

# Position-embedding table: (num_grid * num_grid, hidden_dim)
EMBED_WEIGHT = torch.randn(
    NUM_GRID * NUM_GRID, HIDDEN_DIM, dtype=DTYPE, device=DEVICE
)


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


def pytorch_ref(embed_weight, t, h, w, num_grid, m_size, dtype):
    """Pure PyTorch bilinear interpolation + spatial-merge reorder.

    Reimplements the eager `pos_embed_interpolate_native` algorithm; no vLLM
    kernel calls.
    """
    hidden_dim = embed_weight.shape[1]
    device = embed_weight.device

    h_idxs = torch.linspace(0, num_grid - 1, h, dtype=torch.float32, device=device)
    w_idxs = torch.linspace(0, num_grid - 1, w, dtype=torch.float32, device=device)

    h_floor = h_idxs.to(torch.long)
    w_floor = w_idxs.to(torch.long)
    h_ceil = torch.clamp(h_floor + 1, max=num_grid - 1)
    w_ceil = torch.clamp(w_floor + 1, max=num_grid - 1)

    dh = h_idxs - h_floor
    dw = w_idxs - w_floor

    dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
    h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
    h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

    w11 = dh_grid * dw_grid
    w10 = dh_grid - w11
    w01 = dw_grid - w11
    w00 = 1 - dh_grid - w01

    h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
    w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
    h_grid_idx = h_grid * num_grid

    indices = (h_grid_idx + w_grid).reshape(4, -1)
    weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
    weights = weights.to(dtype=dtype)

    embeds = embed_weight[indices]
    embeds = embeds * weights
    combined = embeds.sum(dim=0)

    combined = combined.reshape(h // m_size, m_size, w // m_size, m_size, hidden_dim)
    combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
    repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
    return repeated.to(dtype=dtype)


def kernel_impl(embed_weight, t, h, w, num_grid, m_size, dtype):
    """Kernel launch only."""
    return triton_pos_embed_interpolate(
        embed_weight, t, h, w, num_grid, m_size, dtype
    )


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(EMBED_WEIGHT, T, H, W, NUM_GRID, M_SIZE, DTYPE)
        kernel_out = kernel_impl(EMBED_WEIGHT, T, H, W, NUM_GRID, M_SIZE, DTYPE)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        torch.testing.assert_close(kernel_cpu.float(), ref_cpu.float(),
                                   rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu.float() - ref_cpu.float()).abs()
        stats = {
            "input_shape": tuple(EMBED_WEIGHT.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(EMBED_WEIGHT.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(EMBED_WEIGHT.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }

        pt_stats = _bench(
            lambda: pytorch_ref(EMBED_WEIGHT, T, H, W, NUM_GRID, M_SIZE, DTYPE)
        )
        kern_stats = _bench(
            lambda: kernel_impl(EMBED_WEIGHT, T, H, W, NUM_GRID, M_SIZE, DTYPE)
        )
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
            "Kernel: _bilinear_pos_embed_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- embed_weight shape: {stats['input_shape']}\n",
                f"- in dtype: {stats['in_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- output shape: {stats['output_shape']}\n",
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
