"""
Standalone QAIC validation for `_trtllm_prefill_attn_kvfp8_dequant`.

Source under test:
vllm/v1/attention/backends/flashinfer.py
  - _trtllm_prefill_attn_kvfp8_dequant  (dequantizes an FP8 paged KV cache
    into a contiguous bf16/fp16 "mock" KV cache holding only the pages that a
    TRT-LLM FlashInfer prefill needs. This is a DEQUANT kernel, NOT attention.)

Layout assumptions (read from the launcher `trtllm_prefill_attn_kvfp8_dequant`):
  - kv_cache: FP8, shape [num_pages, 2, num_kv_heads, block_size, head_size];
    dim 1 has size 2 (index 0 = K half, index 1 = V half). It must be
    contiguous over (block_size, head_size) (strides[3]==head_size,
    strides[4]==1).
  - block_tables_prefill: [batch_size, num_of_page_per_token] int32; each entry
    is an original physical page number (>0 means valid, <=0 skipped).
  - Output mock_kv_cache: dequant dtype, shape
    [batch_size*num_of_page_per_token + 1, 2, num_kv_heads, block_size,
    head_size]. Page 0 is unused/scratch; request (b, p) is written to mock
    page (b*num_of_page_per_token + p + 1).
  - Dequant: K half = fp8_val * k_scale, V half = fp8_val * v_scale, both cast
    to the dequant dtype.

We use the smallest faithful example: batch_size=2, one page per request.

Reference: pure PyTorch fp8->float32 read, per-half scale multiply, gather into
the mock page positions matching the launcher's `mock_block_table` (arange+1).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_trtllm_prefill_attn_kvfp8_dequant.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/backends/flashinfer.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.attention.backends.flashinfer import (  # noqa: E402
    trtllm_prefill_attn_kvfp8_dequant,
)

# ---------------------------------------------------------------------------
# Global inputs
# ---------------------------------------------------------------------------
DEVICE = "qaic"
NUM_PAGES = 4           # total physical pages in the source cache
BATCH_SIZE = 2
NUM_PAGE_PER_TOKEN = 1  # pages per request in this prefill
NUM_KV_HEADS = 2
BLOCK_SIZE = 4
HEAD_SIZE = 8
FP8_DTYPE = torch.float8_e4m3fn
DEQUANT_DTYPE = torch.float16

torch.manual_seed(42)

# Source FP8 KV cache: [num_pages, 2(K/V), num_kv_heads, block_size, head_size].
_KV_FLOAT = torch.randn(
    NUM_PAGES, 2, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE,
    dtype=torch.float32, device=DEVICE,
)
KV_CACHE = _KV_FLOAT.to(FP8_DTYPE)

# Block table: which original pages each request needs (all > 0 => all valid).
BLOCK_TABLES_PREFILL = torch.tensor(
    [[1], [2]], dtype=torch.int32, device=DEVICE
)
K_SCALE = torch.tensor(0.5, dtype=torch.float32, device=DEVICE)
V_SCALE = torch.tensor(0.25, dtype=torch.float32, device=DEVICE)


def pytorch_ref(kv_cache, block_tables_prefill, k_scale, v_scale, dequant_dtype):
    """Pure PyTorch dequant + gather into the mock KV cache."""
    kv_f32 = kv_cache.to(torch.float32).cpu()
    bt = block_tables_prefill.cpu()
    ks = float(k_scale.item())
    vs = float(v_scale.item())

    batch_size, npp = bt.shape
    s = kv_cache.shape
    new_s = (batch_size * npp + 1, s[1], s[2], s[3], s[4])
    mock = torch.zeros(new_s, dtype=torch.float32)

    mock_block_table = torch.zeros(batch_size, npp, dtype=torch.int32)
    for b in range(batch_size):
        for p in range(npp):
            orig = int(bt[b, p].item())
            mock_idx = b * npp + p + 1
            mock_block_table[b, p] = mock_idx
            if orig <= 0:
                continue
            # K half scaled by k_scale, V half scaled by v_scale.
            mock[mock_idx, 0] = kv_f32[orig, 0] * ks
            mock[mock_idx, 1] = kv_f32[orig, 1] * vs

    return mock.to(dequant_dtype), mock_block_table


def kernel_impl(kv_cache, block_tables_prefill, k_scale, v_scale, dequant_dtype):
    """Kernel wrapper: launch only."""
    mock_kv_cache, mock_block_table = trtllm_prefill_attn_kvfp8_dequant(
        kv_cache,
        block_tables_prefill,
        k_scale,
        v_scale,
        dequant_dtype,
    )
    return mock_kv_cache, mock_block_table


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


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_mock, ref_bt = pytorch_ref(
            KV_CACHE, BLOCK_TABLES_PREFILL, K_SCALE, V_SCALE, DEQUANT_DTYPE
        )
        kernel_mock, kernel_bt = kernel_impl(
            KV_CACHE, BLOCK_TABLES_PREFILL, K_SCALE, V_SCALE, DEQUANT_DTYPE
        )

        # Compare only the written pages (page 0 is scratch/uninitialized).
        ref_written = ref_mock[1:].to(torch.float32).cpu()
        kernel_written = kernel_mock[1:].to(torch.float32).cpu()

        torch.testing.assert_close(
            kernel_written, ref_written, rtol=1e-3, atol=1e-3
        )
        # Mock block tables must match exactly (integer indices).
        bt_match = bool(torch.equal(kernel_bt.cpu(), ref_bt.cpu()))
        assert bt_match, "mock_block_table mismatch"

        diff = (kernel_written - ref_written).abs()
        stats = {
            "kv_cache_shape": tuple(KV_CACHE.shape),
            "kv_cache_dtype": str(KV_CACHE.dtype),
            "mock_shape": tuple(kernel_mock.shape),
            "dequant_dtype": str(kernel_mock.dtype),
            "device": str(KV_CACHE.device),
            "block_table_match": bt_match,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": (diff.max() / (ref_written.abs().max() + 1e-8)).item(),
        }

        pt_stats = _bench(lambda: pytorch_ref(
            KV_CACHE, BLOCK_TABLES_PREFILL, K_SCALE, V_SCALE, DEQUANT_DTYPE))
        kern_stats = _bench(lambda: kernel_impl(
            KV_CACHE, BLOCK_TABLES_PREFILL, K_SCALE, V_SCALE, DEQUANT_DTYPE))
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
            "Kernel: _trtllm_prefill_attn_kvfp8_dequant\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- kv_cache shape: {stats['kv_cache_shape']}\n")
            lines.append(f"- kv_cache dtype: {stats['kv_cache_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n")
            lines.append(f"- k_scale: {float(K_SCALE.item())}\n")
            lines.append(f"- v_scale: {float(V_SCALE.item())}\n\n")
            lines.append("Output:\n")
            lines.append(f"- mock_kv_cache shape: {stats['mock_shape']}\n")
            lines.append(f"- dequant dtype: {stats['dequant_dtype']}\n")
            lines.append(f"- block_table_match: {stats['block_table_match']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            lines.append(f"- rel_err: {stats['rel_err']}\n")
            if "pytorch_latency_ms" in stats:
                lines.append("Timing:\n")
                lines.append(
                    f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                    f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                    f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n")
                lines.append(
                    f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                    f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                    f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n")
                lines.append(
                    f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n")
        else:
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))

    return status


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result == "SUCCESS" else 1)
