"""
Standalone QAIC validation for `_fused_kv_compress_norm_rope_insert_indexer_attn`.

Source under test:
vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py
  - _fused_kv_compress_norm_rope_insert_indexer_attn  (head=128, all-FP8 path)
  - launcher: compress_norm_rope_store_triton(...) which dispatches to this
    kernel when head_dim != 512 and use_fp4_cache is False.

Per boundary token (position where (pos+1) % COMPRESS_RATIO == 0, slot>=0) the
kernel:
  1. Gathers (1+OVERLAP)*COMPRESS_RATIO state-cache rows (kv || score) for the
     request via block_table.
  2. compressed_kv = sum_t softmax(score, dim=tokens) * kv        (fp32)
  3. RMSNorm(compressed_kv) with rms_norm_weight, eps               (fp32)
  4. GPT-J interleaved RoPE on the last ROPE_HEAD_DIM dims (cos/sin from
     cos_sin_cache at (pos//CR)*CR).
  5. Single-block FP8 UE8M0 quant of the full 128-wide result:
        result_bf16 = result.to(bf16).to(fp32)
        scale = 2^ceil(log2(max(|result_bf16|,1e-4)/448))
        bytes = (result_bf16 / scale) as float8_e4m3fn
     Writes fp8 bytes + one float32 scale into the paged KV cache slot.

Config tested (supported sub-path): HEAD=128, ROPE=64, COMPRESS_RATIO=4,
OVERLAP=False (coff=1 state layout: last dim = 2*HEAD), one boundary token.
Reference: pure PyTorch replicating the above. We DEQUANTIZE the written cache
(fp8 bytes * stored scale) and compare vs the reference dequant (assert_close).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(
    LOG_DIR, "log_fused_kv_compress_norm_rope_insert_indexer_attn.txt"
)
KERNEL_FILE_PATH = (
    "vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py"
)
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (  # noqa: E402
    compress_norm_rope_store_triton,
)

torch.manual_seed(42)

HEAD = 128
ROPE = 64
NOPE = HEAD - ROPE
COMPRESS_RATIO = 4
OVERLAP = False
N_GATHER = (1 + int(OVERLAP)) * COMPRESS_RATIO  # 4
STATE_BLOCK_SIZE = 8
STATE_WIDTH = HEAD  # coff=1 -> state last dim = 2*HEAD
EPS = 1e-6
FP8_MAX = 448.0
QUANT_BLOCK = 128
TOKEN_STRIDE = 128
SCALE_DIM = 4  # one float32
FP8_DTYPE = torch.float8_e4m3fn

# KV cache geometry
KV_BLOCK_SIZE = 4
NUM_KV_BLOCKS = 4
KV_HEAD_BYTES = TOKEN_STRIDE + SCALE_DIM  # per-token bytes (flat layout)

# Boundary token at position 3 -> (3+1)%4==0. start = 3-4+1 = 0, gather pos 0..3.
POSITION = 3
KV_SLOT = 5  # -> block 1, pos_in_block 1


class _KMeta:
    def __init__(self, slot_mapping):
        self.slot_mapping = slot_mapping


# ---- Global shared inputs (used by BOTH implementations) ----
# state cache: [num_blocks, block_size, 2*HEAD] fp32  (kv || score)
STATE_CACHE = torch.randn(
    NUM_KV_BLOCKS, STATE_BLOCK_SIZE, 2 * HEAD, dtype=torch.float32, device=DEVICE
)
TOKEN_TO_REQ = torch.zeros(1, dtype=torch.int64, device=DEVICE)
POSITIONS = torch.tensor([POSITION], dtype=torch.int64, device=DEVICE)
SLOT_MAPPING = torch.zeros(1, dtype=torch.int64, device=DEVICE)  # state slot ok
KV_SLOT_MAPPING = torch.tensor([KV_SLOT], dtype=torch.int64, device=DEVICE)
# block_table[req, blk] = blk  (identity) so gathered rows map to state block idx.
BLOCK_TABLE = (
    torch.arange(NUM_KV_BLOCKS, dtype=torch.int32, device=DEVICE)
    .reshape(1, NUM_KV_BLOCKS)
)
RMS_W = torch.randn(HEAD, dtype=torch.float32, device=DEVICE)
# cos_sin_cache: [max_pos, ROPE] first half cos (ROPE//2) then sin (ROPE//2)
MAX_POS = 8
COS_SIN = torch.randn(MAX_POS, ROPE, dtype=torch.float32, device=DEVICE)


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


def _gptj_rope(normed, cos_sin, compressed_pos):
    """GPT-J interleaved forward RoPE on the last ROPE dims (fp32)."""
    num_pairs = HEAD // 2
    nope_pairs = NOPE // 2
    half_rope = ROPE // 2
    even = normed[0::2].clone()
    odd = normed[1::2].clone()
    cos_row = cos_sin[compressed_pos]
    result = torch.empty(HEAD, dtype=torch.float32)
    for i in range(num_pairs):
        if i >= nope_pairs:
            cs = i - nope_pairs
            c = cos_row[cs]
            s = cos_row[half_rope + cs]
            ne = even[i] * c - odd[i] * s
            no = odd[i] * c + even[i] * s
        else:
            ne = even[i]
            no = odd[i]
        result[2 * i] = ne
        result[2 * i + 1] = no
    return result


def pytorch_ref():
    """Pure PyTorch compress+norm+rope+FP8-quant; returns dequantized row."""
    state = STATE_CACHE.cpu().float()
    cos_sin = COS_SIN.cpu().float()
    rms_w = RMS_W.cpu().float()

    # gather rows: pos 0..N_GATHER-1 (OVERLAP=False so head_offset always 0)
    start = POSITION - (1 + int(OVERLAP)) * COMPRESS_RATIO + 1
    kv_rows = []
    score_rows = []
    for t in range(N_GATHER):
        pos = start + t
        blk = int(BLOCK_TABLE[0, pos // STATE_BLOCK_SIZE].item())
        off = pos % STATE_BLOCK_SIZE
        row = state[blk, off]
        kv_rows.append(row[0:HEAD])
        score_rows.append(row[STATE_WIDTH : STATE_WIDTH + HEAD])
    kv = torch.stack(kv_rows, dim=0)  # [N_GATHER, HEAD]
    score = torch.stack(score_rows, dim=0)

    sm = torch.softmax(score, dim=0)
    comp = (kv * sm).sum(dim=0)  # [HEAD]

    var = comp.pow(2).mean()
    normed = comp * torch.rsqrt(var + EPS) * rms_w

    compressed_pos = (POSITION // COMPRESS_RATIO) * COMPRESS_RATIO
    result = _gptj_rope(normed, cos_sin, compressed_pos)

    result_bf16 = result.to(torch.bfloat16).to(torch.float32)
    absmax = result_bf16.abs().max().clamp_min(1e-4)
    exponent = torch.ceil(torch.log2(absmax / FP8_MAX))
    inv_scale = torch.exp2(-exponent)
    x = torch.clamp(result_bf16 * inv_scale, -FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    scale_val = float(torch.exp2(exponent).item())
    deq = x.to(torch.float32) * scale_val
    return deq, scale_val


def _make_kv_cache():
    flat = torch.zeros(
        NUM_KV_BLOCKS * KV_BLOCK_SIZE * KV_HEAD_BYTES,
        dtype=torch.uint8,
        device=DEVICE,
    )
    return flat


def kernel_impl():
    """Kernel launch only (writes into a fresh KV cache; returns dequant row)."""
    flat = _make_kv_cache()
    kv_cache = flat.view(NUM_KV_BLOCKS, KV_BLOCK_SIZE, KV_HEAD_BYTES)
    compress_norm_rope_store_triton(
        state_cache=STATE_CACHE,
        num_actual=1,
        token_to_req_indices=TOKEN_TO_REQ,
        positions=POSITIONS,
        slot_mapping=SLOT_MAPPING,
        block_table=BLOCK_TABLE,
        block_size=STATE_BLOCK_SIZE,
        state_width=STATE_WIDTH,
        cos_sin_cache=COS_SIN,
        kv_cache=kv_cache,
        k_cache_metadata=_KMeta(KV_SLOT_MAPPING),
        pdl_kwargs={},
        head_dim=HEAD,
        rope_head_dim=ROPE,
        compress_ratio=COMPRESS_RATIO,
        overlap=OVERLAP,
        use_fp4_cache=False,
        rms_norm_weight=RMS_W,
        rms_norm_eps=EPS,
        quant_block=QUANT_BLOCK,
        token_stride=TOKEN_STRIDE,
        scale_dim=SCALE_DIM,
    )
    # read back the written slot
    flat_cpu = flat.cpu()
    block_stride = KV_BLOCK_SIZE * KV_HEAD_BYTES
    blk = KV_SLOT // KV_BLOCK_SIZE
    pos = KV_SLOT % KV_BLOCK_SIZE
    base = blk * block_stride
    fp8_off = base + pos * TOKEN_STRIDE
    fp8_bytes = flat_cpu[fp8_off : fp8_off + HEAD].clone()
    deq_fp8 = fp8_bytes.view(FP8_DTYPE).to(torch.float32)
    scale_off = base + KV_BLOCK_SIZE * TOKEN_STRIDE + pos * SCALE_DIM
    scale = flat_cpu[scale_off : scale_off + 4].clone().view(torch.float32)
    deq = deq_fp8 * float(scale.item())
    return deq, float(scale.item())


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_deq, ref_scale = pytorch_ref()
        kern_deq, kern_scale = kernel_impl()

        ref_cpu = ref_deq.cpu().float()
        kernel_cpu = kern_deq.cpu().float()
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "input_shape": tuple(STATE_CACHE.shape),
            "output_shape": tuple(kern_deq.shape),
            "in_dtype": str(STATE_CACHE.dtype),
            "out_dtype": str(FP8_DTYPE),
            "device": str(STATE_CACHE.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "scale_abs_diff": abs(kern_scale - ref_scale),
        }

        pt_stats = _bench(lambda: pytorch_ref())
        kern_stats = _bench(lambda: kernel_impl())
        speedup = (
            kern_stats["avg_ms"] / pt_stats["avg_ms"]
            if pt_stats["avg_ms"] > 0
            else float("nan")
        )
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
            "Kernel: _fused_kv_compress_norm_rope_insert_indexer_attn\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- state_cache shape: {stats['input_shape']}\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(
                f"- HEAD={HEAD}, ROPE={ROPE}, CR={COMPRESS_RATIO}, "
                f"OVERLAP={OVERLAP} (sub-path)\n"
            )
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output (dequantized FP8 row, rtol/atol=1e-3):\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- out dtype: {stats['out_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            lines.append(f"- scale_abs_diff: {stats['scale_abs_diff']}\n")
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
    sys.exit(0 if main() == "SUCCESS" else 1)
