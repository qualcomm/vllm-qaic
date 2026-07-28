"""
Standalone QAIC validation for `_deepseek_v4_stage_mega_moe_inputs_kernel` (XPU).

Source under test:
vllm/models/deepseek_v4/xpu/model.py
  - _deepseek_v4_stage_mega_moe_inputs_kernel   (@triton.jit)
  - launcher: _stage_deepseek_v4_mega_moe_inputs(...)

XPU variant of the MegaMoE input-staging kernel. Located inside the large
model.py by grepping for the kernel name (def at line ~112) and its single
launch site (the `_stage_deepseek_v4_mega_moe_inputs` wrapper at line ~215 that
builds grid=(num_tokens, cdiv(hidden,128)) with GROUP_K=32). The kernel body is
byte-for-byte the same computation as the NVIDIA `_prepare_megamoe_inputs_kernel`:

  (a) per-32-element-group E8M0 (UE8M0 power-of-two) FP8 (float8e4nv) quant of
      hidden states:
          amax  = max(|group|, 1e-4);  scale = amax/448.0
          scale_exp = biased_exp(scale) + (mantissa != 0)  clamped [1,254]
          rounded_scale = 2^(scale_exp-127);  fp8 = hidden / rounded_scale
      The 4 group exponents of each 128-wide K block are packed little-endian
      into one int32 (x_sf).
  (b) top-k repack: topk_ids -> int64 topk_idx_out, topk_weights -> float32
      topk_weights_out (same [T, top_k] layout).

FP8 comparison choice (fp8 represented numerically): reference recomputes the
identical E8M0 rounded_scale, quantizes, and we compare DEQUANTIZED hidden
(x_fp8.float()*rounded_scale) rtol/atol=1e-2, packed E8M0 scales (int32) EXACT,
repacked ids (int64) EXACT, weights (float32) within 1e-3.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.models.deepseek_v4.xpu.model import _stage_deepseek_v4_mega_moe_inputs

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(
    LOG_DIR, "log_deepseek_v4_stage_mega_moe_inputs_kernel.txt"
)
KERNEL_FILE_PATH = "vllm/models/deepseek_v4/xpu/model.py"

DEVICE = "qaic"
torch.manual_seed(42)

NUM_TOKENS = 4
HIDDEN_SIZE = 128
BLOCK_K = 128
GROUP_K = 32
NUM_GROUPS = BLOCK_K // GROUP_K   # 4
NUM_K_BLOCKS = HIDDEN_SIZE // BLOCK_K   # 1
TOP_K = 4
NUM_EXPERTS = 8
FP8_DTYPE = torch.float8_e4m3fn

HIDDEN = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.float32, device=DEVICE)
TOPK_WEIGHTS = torch.rand(NUM_TOKENS, TOP_K, dtype=torch.float32, device=DEVICE)
TOPK_IDS = torch.stack(
    [torch.randperm(NUM_EXPERTS, device=DEVICE)[:TOP_K] for _ in range(NUM_TOKENS)]
).to(torch.int32)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def _e8m0_rounded_scale(amax_group):
    scale = (amax_group / 448.0).to(torch.float32)
    scale_bits = scale.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    exp = (scale_bits >> 23) & 0xFF
    mant = scale_bits & 0x7FFFFF
    scale_exp = exp + (mant != 0).to(torch.int64)
    scale_exp = scale_exp.clamp(1, 254)
    rounded_bits = (scale_exp << 23).to(torch.int32)
    rounded_scale = rounded_bits.view(torch.float32)
    return scale_exp, rounded_scale


def pytorch_ref(hidden, topk_weights, topk_ids):
    h = hidden.cpu().to(torch.float32)
    tw = topk_weights.cpu().to(torch.float32)
    ti = topk_ids.cpu().to(torch.int64)

    x_fp8 = torch.empty(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.float32)
    rounded = torch.empty(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.float32)
    x_sf = torch.empty(NUM_TOKENS, NUM_K_BLOCKS, dtype=torch.int32)

    for t in range(NUM_TOKENS):
        for kb in range(NUM_K_BLOCKS):
            block = h[t, kb * BLOCK_K:(kb + 1) * BLOCK_K].reshape(NUM_GROUPS, GROUP_K)
            amax = block.abs().amax(dim=1).clamp(min=1e-4)
            scale_exp, rounded_scale = _e8m0_rounded_scale(amax)
            scaled = block / rounded_scale[:, None]
            q = scaled.to(FP8_DTYPE).to(torch.float32)
            x_fp8[t, kb * BLOCK_K:(kb + 1) * BLOCK_K] = q.reshape(BLOCK_K)
            rounded[t, kb * BLOCK_K:(kb + 1) * BLOCK_K] = (
                rounded_scale[:, None].expand(NUM_GROUPS, GROUP_K).reshape(BLOCK_K)
            )
            packed = 0
            for g in range(NUM_GROUPS):
                packed |= int(scale_exp[g].item()) << (g * 8)
            x_sf[t, kb] = packed

    return x_fp8, rounded, x_sf, ti.clone(), tw.clone()


def kernel_impl(hidden, topk_weights, topk_ids):
    x_fp8 = torch.empty(NUM_TOKENS, HIDDEN_SIZE, dtype=FP8_DTYPE, device=DEVICE)
    x_sf = torch.empty(NUM_TOKENS, NUM_K_BLOCKS, dtype=torch.int32, device=DEVICE)
    topk_idx_out = torch.empty(NUM_TOKENS, TOP_K, dtype=torch.int64, device=DEVICE)
    topk_weights_out = torch.empty(
        NUM_TOKENS, TOP_K, dtype=torch.float32, device=DEVICE
    )
    _stage_deepseek_v4_mega_moe_inputs(
        hidden,
        topk_weights,
        topk_ids,
        x_fp8,
        x_sf,
        topk_idx_out,
        topk_weights_out,
    )
    return x_fp8, x_sf, topk_idx_out, topk_weights_out


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
        ref_fp8, ref_rounded, ref_sf, ref_idx, ref_w = pytorch_ref(
            HIDDEN, TOPK_WEIGHTS, TOPK_IDS
        )
        k_fp8, k_sf, k_idx, k_w = kernel_impl(HIDDEN, TOPK_WEIGHTS, TOPK_IDS)
        k_fp8 = k_fp8.cpu().to(torch.float32)
        k_sf = k_sf.cpu().to(torch.int32)
        k_idx = k_idx.cpu().to(torch.int64)
        k_w = k_w.cpu().to(torch.float32)

        deq_ref = ref_fp8 * ref_rounded
        deq_k = k_fp8 * ref_rounded
        torch.testing.assert_close(deq_k, deq_ref, rtol=1e-2, atol=1e-2)
        assert torch.equal(k_sf, ref_sf), f"scale mismatch {k_sf} vs {ref_sf}"
        assert torch.equal(k_idx, ref_idx), "topk id repack mismatch"
        torch.testing.assert_close(k_w, ref_w, rtol=1e-3, atol=1e-3)

        diff = (deq_k - deq_ref).abs()
        stats = {
            "hidden_shape": tuple(HIDDEN.shape),
            "x_fp8_shape": tuple(k_fp8.shape),
            "x_sf_shape": tuple(k_sf.shape),
            "topk_idx_shape": tuple(k_idx.shape),
            "fp8_dtype": str(FP8_DTYPE),
            "device": DEVICE,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "scales_exact": True,
            "ids_exact": True,
        }
        pt_stats = _bench(lambda: pytorch_ref(HIDDEN, TOPK_WEIGHTS, TOPK_IDS))
        kern_stats = _bench(lambda: kernel_impl(HIDDEN, TOPK_WEIGHTS, TOPK_IDS))
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
            "Kernel: _deepseek_v4_stage_mega_moe_inputs_kernel (XPU, FP8 E8M0 quant + repack)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- hidden shape: {stats['hidden_shape']} "
                f"(GROUP_K={GROUP_K}, BLOCK_K={BLOCK_K})\n",
                f"- top_k={TOP_K}, num_experts={NUM_EXPERTS}, num_tokens={NUM_TOKENS}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- x_fp8 shape: {stats['x_fp8_shape']} dtype {stats['fp8_dtype']}\n",
                f"- x_sf (packed E8M0) shape: {stats['x_sf_shape']} EXACT match: "
                f"{stats['scales_exact']}\n",
                f"- topk_idx_out shape: {stats['topk_idx_shape']} EXACT match: "
                f"{stats['ids_exact']}\n",
                "- comparison: dequant hidden rtol/atol=1e-2, scales/ids exact, "
                "weights 1e-3\n",
                f"- dequant max_abs_diff: {stats['max_abs_diff']}\n",
                f"- dequant mean_abs_diff: {stats['mean_abs_diff']}\n",
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
