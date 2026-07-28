"""
Standalone QAIC validation for
`_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn`.

Source under test:
vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py
  - _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn (head=128, MXFP4)
  - launcher: compress_norm_rope_store_triton(...) which dispatches to this
    kernel when head_dim != 512 and use_fp4_cache is True.

Same compress -> RMSNorm -> GPT-J RoPE pipeline as the FP8 indexer kernel, but
the final quant is MXFP4:
  * even/odd RoPE halves are kept (no interleave), bf16-roundtripped.
  * tiled into (N_BLOCKS=HEAD/32, HALF_BLOCK=16); per-block
        amax   = max(|even|, |odd|)  over the block  (>= 6*2^-126)
        scale  = 2^ceil(log2(amax/6))         (E2M1 max magnitude = 6)
        ue8m0  = exp + 127  byte
  * packed = E2M1(even/scale) low nibble | E2M1(odd/scale) high nibble,
    2 values per byte, TOKEN_STRIDE = HEAD//2 = 64 bytes/token.

Config: HEAD=128, ROPE=64, CR=4, OVERLAP=False, one boundary token.

IMPORTANT (documented): MXFP4 packing uses inline PTX
`cvt.rn.satfinite.e2m1x2.f32` (`_fp32x2_to_fp4x2`), which is NVIDIA-PTX only and
may NOT compile on the QAIC/Hexagon Triton backend. We are NOT executing on
device here. The reference implements E2M1 round-to-nearest-even packing and
the identical block-scale math. Comparison (were it to run): DEQUANTIZE the
written packed nibbles * per-block scale and compare vs the reference dequant.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(
    LOG_DIR, "log_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn.txt"
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
N_GATHER = (1 + int(OVERLAP)) * COMPRESS_RATIO
STATE_BLOCK_SIZE = 8
STATE_WIDTH = HEAD
EPS = 1e-6
MXFP4_BLOCK = 32
QUANT_BLOCK = MXFP4_BLOCK
TOKEN_STRIDE = HEAD // 2  # 64 packed bytes/token
SCALE_DIM = HEAD // MXFP4_BLOCK  # 4 ue8m0 bytes/token
N_QUANT_BLOCKS = HEAD // MXFP4_BLOCK  # 4
HALF_BLOCK = MXFP4_BLOCK // 2  # 16

KV_BLOCK_SIZE = 4
NUM_KV_BLOCKS = 4
KV_HEAD_BYTES = TOKEN_STRIDE + SCALE_DIM

POSITION = 3
KV_SLOT = 5

_E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


class _KMeta:
    def __init__(self, slot_mapping):
        self.slot_mapping = slot_mapping


# ---- Global shared inputs (used by BOTH implementations) ----
STATE_CACHE = torch.randn(
    NUM_KV_BLOCKS, STATE_BLOCK_SIZE, 2 * HEAD, dtype=torch.float32, device=DEVICE
)
TOKEN_TO_REQ = torch.zeros(1, dtype=torch.int64, device=DEVICE)
POSITIONS = torch.tensor([POSITION], dtype=torch.int64, device=DEVICE)
SLOT_MAPPING = torch.zeros(1, dtype=torch.int64, device=DEVICE)
KV_SLOT_MAPPING = torch.tensor([KV_SLOT], dtype=torch.int64, device=DEVICE)
BLOCK_TABLE = (
    torch.arange(NUM_KV_BLOCKS, dtype=torch.int32, device=DEVICE)
    .reshape(1, NUM_KV_BLOCKS)
)
RMS_W = torch.randn(HEAD, dtype=torch.float32, device=DEVICE)
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


def _e2m1_encode_rne(v: float) -> int:
    sign = 8 if v < 0 else 0
    a = abs(v)
    if a >= 6.0:
        return sign | 7
    best_code = 0
    best_dist = abs(a - _E2M1_MAG[0])
    for code in range(1, 8):
        d = abs(a - _E2M1_MAG[code])
        if d < best_dist - 1e-12:
            best_dist = d
            best_code = code
        elif abs(d - best_dist) <= 1e-12:
            if (code % 2 == 0) and (best_code % 2 == 1):
                best_code = code
                best_dist = d
    return sign | best_code


def _e2m1_decode(code: int) -> float:
    mag = _E2M1_MAG[code & 7]
    return -mag if (code & 8) else mag


def _rope_even_odd(normed, cos_sin, compressed_pos):
    """GPT-J RoPE returning (new_even, new_odd), each [HEAD//2], bf16-roundtrip."""
    num_pairs = HEAD // 2
    nope_pairs = NOPE // 2
    half_rope = ROPE // 2
    even = normed[0::2].clone()
    odd = normed[1::2].clone()
    cos_row = cos_sin[compressed_pos]
    ne = torch.empty(num_pairs, dtype=torch.float32)
    no = torch.empty(num_pairs, dtype=torch.float32)
    for i in range(num_pairs):
        if i >= nope_pairs:
            cs = i - nope_pairs
            c = cos_row[cs]
            s = cos_row[half_rope + cs]
            ne[i] = even[i] * c - odd[i] * s
            no[i] = odd[i] * c + even[i] * s
        else:
            ne[i] = even[i]
            no[i] = odd[i]
    ne = ne.to(torch.bfloat16).to(torch.float32)
    no = no.to(torch.bfloat16).to(torch.float32)
    return ne, no


def pytorch_ref():
    """Pure PyTorch compress+norm+rope+MXFP4-quant; returns dequant even/odd."""
    state = STATE_CACHE.cpu().float()
    cos_sin = COS_SIN.cpu().float()
    rms_w = RMS_W.cpu().float()

    start = POSITION - (1 + int(OVERLAP)) * COMPRESS_RATIO + 1
    kv_rows, score_rows = [], []
    for t in range(N_GATHER):
        pos = start + t
        blk = int(BLOCK_TABLE[0, pos // STATE_BLOCK_SIZE].item())
        off = pos % STATE_BLOCK_SIZE
        row = state[blk, off]
        kv_rows.append(row[0:HEAD])
        score_rows.append(row[STATE_WIDTH : STATE_WIDTH + HEAD])
    kv = torch.stack(kv_rows, dim=0)
    score = torch.stack(score_rows, dim=0)
    sm = torch.softmax(score, dim=0)
    comp = (kv * sm).sum(dim=0)

    var = comp.pow(2).mean()
    normed = comp * torch.rsqrt(var + EPS) * rms_w

    compressed_pos = (POSITION // COMPRESS_RATIO) * COMPRESS_RATIO
    ne, no = _rope_even_odd(normed, cos_sin, compressed_pos)

    ne2 = ne.reshape(N_QUANT_BLOCKS, HALF_BLOCK)
    no2 = no.reshape(N_QUANT_BLOCKS, HALF_BLOCK)
    deq_even = torch.empty_like(ne2)
    deq_odd = torch.empty_like(no2)
    for b in range(N_QUANT_BLOCKS):
        amax = max(ne2[b].abs().max().item(), no2[b].abs().max().item())
        amax = max(amax, 6.0 * (2**-126))
        log2_ratio = float(torch.ceil(torch.log2(torch.tensor(amax / 6.0))).item())
        log2_ratio = min(max(log2_ratio, -127.0), 127.0)
        inv_scale = 2.0 ** (-log2_ratio)
        scale = 2.0 ** log2_ratio
        for j in range(HALF_BLOCK):
            ce = _e2m1_encode_rne(float(ne2[b, j].item()) * inv_scale)
            co = _e2m1_encode_rne(float(no2[b, j].item()) * inv_scale)
            deq_even[b, j] = _e2m1_decode(ce) * scale
            deq_odd[b, j] = _e2m1_decode(co) * scale
    return deq_even.reshape(-1), deq_odd.reshape(-1)


def _make_kv_cache():
    return torch.zeros(
        NUM_KV_BLOCKS * KV_BLOCK_SIZE * KV_HEAD_BYTES,
        dtype=torch.uint8,
        device=DEVICE,
    )


def kernel_impl():
    """Kernel launch only; returns dequantized even/odd halves from cache."""
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
        use_fp4_cache=True,
        rms_norm_weight=RMS_W,
        rms_norm_eps=EPS,
        quant_block=QUANT_BLOCK,
        token_stride=TOKEN_STRIDE,
        scale_dim=SCALE_DIM,
    )
    flat_cpu = flat.cpu()
    block_stride = KV_BLOCK_SIZE * KV_HEAD_BYTES
    blk = KV_SLOT // KV_BLOCK_SIZE
    pos = KV_SLOT % KV_BLOCK_SIZE
    base = blk * block_stride
    val_off = base + pos * TOKEN_STRIDE
    packed = flat_cpu[val_off : val_off + TOKEN_STRIDE]  # [64] uint8
    scale_off = base + KV_BLOCK_SIZE * TOKEN_STRIDE + pos * SCALE_DIM
    ue8m0 = flat_cpu[scale_off : scale_off + SCALE_DIM]  # [4] uint8

    packed_2d = packed.reshape(N_QUANT_BLOCKS, HALF_BLOCK)
    deq_even = torch.empty(N_QUANT_BLOCKS, HALF_BLOCK, dtype=torch.float32)
    deq_odd = torch.empty(N_QUANT_BLOCKS, HALF_BLOCK, dtype=torch.float32)
    for b in range(N_QUANT_BLOCKS):
        scale = 2.0 ** (int(ue8m0[b].item()) - 127)
        for j in range(HALF_BLOCK):
            byte = int(packed_2d[b, j].item())
            lo = byte & 0xF  # even
            hi = (byte >> 4) & 0xF  # odd
            deq_even[b, j] = _e2m1_decode(lo) * scale
            deq_odd[b, j] = _e2m1_decode(hi) * scale
    return deq_even.reshape(-1), deq_odd.reshape(-1)


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_e, ref_o = pytorch_ref()
        kern_e, kern_o = kernel_impl()

        re = torch.cat([ref_e, ref_o]).float()
        ke = torch.cat([kern_e, kern_o]).float()
        torch.testing.assert_close(ke, re, rtol=1e-3, atol=1e-3)

        diff = (ke - re).abs()
        stats = {
            "input_shape": tuple(STATE_CACHE.shape),
            "output_shape": (HEAD,),
            "in_dtype": str(STATE_CACHE.dtype),
            "out_dtype": "mxfp4 (packed uint8 nibbles)",
            "device": str(STATE_CACHE.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
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
            "Kernel: _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            "Note: MXFP4 pack uses inline PTX (cvt.rn.satfinite.e2m1x2.f32),\n"
            "      NVIDIA-only; may not compile on QAIC/Hexagon. Reference is\n"
            "      pure-PyTorch E2M1 packing; comparison is dequantized values.\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- state_cache shape: {stats['input_shape']}\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(
                f"- HEAD={HEAD}, ROPE={ROPE}, CR={COMPRESS_RATIO}, "
                f"OVERLAP={OVERLAP} (MXFP4 sub-path)\n"
            )
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output (dequantized MXFP4, rtol/atol=1e-3):\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- out dtype: {stats['out_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
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
