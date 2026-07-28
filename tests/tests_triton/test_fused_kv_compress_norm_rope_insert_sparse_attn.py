"""
Standalone QAIC validation for
`_fused_kv_compress_norm_rope_insert_sparse_attn`.

Source under test:
vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py
  - _fused_kv_compress_norm_rope_insert_sparse_attn (head=512 DeepseekV4 attn)
  - launcher: compress_norm_rope_store_triton(...) which dispatches to this
    kernel when head_dim == 512.

Per boundary token this kernel:
  1. Gathers (1+OVERLAP)*COMPRESS_RATIO state-cache rows (kv || score). With
     OVERLAP=True (coff=2) the state last dim packs two heads: kv occupies
     [0:2*HEAD] (state_width) and score [state_width:2*state_width]; tokens
     0..CR-1 read head 0, tokens CR..2CR-1 read head 1 (head_offset = HEAD).
  2. compressed_kv = sum_t softmax(score, dim=tokens) * kv                (fp32)
  3. RMSNorm(compressed_kv) with rms_norm_weight, eps                     (fp32)
  4. FP8 UE8M0 quant of the NoPE portion (first 448 dims), 7 blocks of 64:
        qin        = normed.to(bf16).to(fp32)
        absmax_b   = max(|qin_block|, 1e-4)
        exp_b      = ceil(log2(absmax_b / 448))
        bytes      = clamp(qin_block * 2^-exp_b, +-448) as float8_e4m3fn
        ue8m0_b    = clamp(exp_b + 127, 0, 255)     (+ 1 pad byte)
  5. GPT-J interleaved RoPE on the last ROPE=64 dims, stored bf16 after the
     448 fp8 bytes.
Cache token layout: 448 fp8 + 128 bf16 (=576) then 8 ue8m0 scale bytes/token.

Config: HEAD=512, ROPE=64, CR=4, OVERLAP=True, one boundary token.
Reference: pure PyTorch replicating the above. We DEQUANTIZE the written NoPE
FP8 (bytes * 2^(ue8m0-127)) and read back the RoPE bf16; both compared with
assert_close (fp8/bf16 tolerance).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(
    LOG_DIR, "log_fused_kv_compress_norm_rope_insert_sparse_attn.txt"
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

HEAD = 512
ROPE = 64
NOPE = HEAD - ROPE  # 448
COMPRESS_RATIO = 4
OVERLAP = True
COFF = 1 + int(OVERLAP)  # 2
N_GATHER = COFF * COMPRESS_RATIO  # 8
STATE_BLOCK_SIZE = 8
STATE_WIDTH = COFF * HEAD  # 1024
EPS = 1e-6
FP8_MAX = 448.0
QUANT_BLOCK = 64
TOKEN_STRIDE = NOPE + ROPE * 2  # 576
SCALE_DIM = NOPE // 64 + 1  # 8 (7 real + 1 pad)
N_NOPE_BLOCKS = NOPE // QUANT_BLOCK  # 7
FP8_DTYPE = torch.float8_e4m3fn

KV_BLOCK_SIZE = 4
NUM_KV_BLOCKS = 4
KV_HEAD_BYTES = TOKEN_STRIDE + SCALE_DIM  # 584

POSITION = 7  # (7+1)%4==0; start = 7 - 2*4 + 1 = 0 -> gather pos 0..7
KV_SLOT = 5


class _KMeta:
    def __init__(self, slot_mapping):
        self.slot_mapping = slot_mapping


# ---- Global shared inputs (used by BOTH implementations) ----
# state cache: [num_blocks, block_size, 2*state_width=2048] fp32 (kv || score)
STATE_CACHE = torch.randn(
    NUM_KV_BLOCKS, STATE_BLOCK_SIZE, 2 * STATE_WIDTH, dtype=torch.float32,
    device=DEVICE,
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
MAX_POS = 16
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


def _compressed_normed():
    """Shared: gather -> softmax weighted sum -> RMSNorm -> normed[HEAD]."""
    state = STATE_CACHE.cpu().float()
    rms_w = RMS_W.cpu().float()
    start = POSITION - COFF * COMPRESS_RATIO + 1
    kv_rows, score_rows = [], []
    for t in range(N_GATHER):
        pos = start + t
        blk = int(BLOCK_TABLE[0, pos // STATE_BLOCK_SIZE].item())
        off = pos % STATE_BLOCK_SIZE
        head_offset = HEAD if t >= COMPRESS_RATIO else 0
        row = state[blk, off]
        kv_rows.append(row[head_offset : head_offset + HEAD])
        score_rows.append(
            row[STATE_WIDTH + head_offset : STATE_WIDTH + head_offset + HEAD]
        )
    kv = torch.stack(kv_rows, dim=0)
    score = torch.stack(score_rows, dim=0)
    sm = torch.softmax(score, dim=0)
    comp = (kv * sm).sum(dim=0)
    var = comp.pow(2).mean()
    normed = comp * torch.rsqrt(var + EPS) * rms_w
    return normed


def pytorch_ref():
    """Returns (dequant_nope[448], rope_bf16[64])."""
    normed = _compressed_normed()
    cos_sin = COS_SIN.cpu().float()

    # FP8 quant of NoPE portion (7 blocks of 64)
    qin = normed.to(torch.bfloat16).to(torch.float32)
    deq_nope = torch.empty(NOPE, dtype=torch.float32)
    for b in range(N_NOPE_BLOCKS):
        lo, hi = b * QUANT_BLOCK, (b + 1) * QUANT_BLOCK
        blk = qin[lo:hi]
        absmax = blk.abs().max().clamp_min(1e-4)
        exp = float(torch.ceil(torch.log2(absmax / FP8_MAX)).item())
        inv = 2.0 ** (-exp)
        x = torch.clamp(blk * inv, -FP8_MAX, FP8_MAX).to(FP8_DTYPE)
        encoded = min(max(exp + 127.0, 0.0), 255.0)
        scale = 2.0 ** (int(encoded) - 127)  # dequant via stored ue8m0 byte
        deq_nope[lo:hi] = x.to(torch.float32) * scale

    # GPT-J RoPE on last ROPE dims -> store result[NOPE:HEAD] as bf16
    num_pairs = HEAD // 2
    nope_pairs = NOPE // 2  # 224
    half_rope = ROPE // 2
    even = normed[0::2].clone()
    odd = normed[1::2].clone()
    compressed_pos = (POSITION // COMPRESS_RATIO) * COMPRESS_RATIO
    cos_row = cos_sin[compressed_pos]
    result = torch.empty(HEAD, dtype=torch.float32)
    for i in range(num_pairs):
        if i >= nope_pairs:
            cs = i - nope_pairs
            c = cos_row[cs]
            s = cos_row[half_rope + cs]
            result[2 * i] = even[i] * c - odd[i] * s
            result[2 * i + 1] = odd[i] * c + even[i] * s
        else:
            result[2 * i] = even[i]
            result[2 * i + 1] = odd[i]
    rope_bf16 = result[NOPE:HEAD].to(torch.bfloat16).to(torch.float32)
    return deq_nope, rope_bf16


def _make_kv_cache():
    return torch.zeros(
        NUM_KV_BLOCKS * KV_BLOCK_SIZE * KV_HEAD_BYTES,
        dtype=torch.uint8,
        device=DEVICE,
    )


def kernel_impl():
    """Kernel launch only; returns (dequant_nope[448], rope_bf16[64])."""
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
    flat_cpu = flat.cpu()
    block_stride = KV_BLOCK_SIZE * KV_HEAD_BYTES
    blk = KV_SLOT // KV_BLOCK_SIZE
    pos = KV_SLOT % KV_BLOCK_SIZE
    base = blk * block_stride
    tok = base + pos * TOKEN_STRIDE
    fp8_bytes = flat_cpu[tok : tok + NOPE].clone()  # 448 uint8
    deq_bytes = fp8_bytes.view(FP8_DTYPE).to(torch.float32)
    scale_off = base + KV_BLOCK_SIZE * TOKEN_STRIDE + pos * SCALE_DIM
    ue8m0 = flat_cpu[scale_off : scale_off + SCALE_DIM]  # 8 bytes

    deq_nope = torch.empty(NOPE, dtype=torch.float32)
    for b in range(N_NOPE_BLOCKS):
        scale = 2.0 ** (int(ue8m0[b].item()) - 127)
        lo, hi = b * QUANT_BLOCK, (b + 1) * QUANT_BLOCK
        deq_nope[lo:hi] = deq_bytes[lo:hi] * scale

    # rope bf16 region: 64 bf16 values right after the 448 fp8 bytes
    rope_byte_off = tok + NOPE
    rope_bytes = flat_cpu[rope_byte_off : rope_byte_off + ROPE * 2].clone()
    rope_bf16 = rope_bytes.view(torch.bfloat16).to(torch.float32)
    return deq_nope, rope_bf16


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_nope, ref_rope = pytorch_ref()
        kern_nope, kern_rope = kernel_impl()

        re = torch.cat([ref_nope, ref_rope]).float()
        ke = torch.cat([kern_nope, kern_rope]).float()
        torch.testing.assert_close(ke, re, rtol=1e-2, atol=1e-2)

        diff = (ke - re).abs()
        stats = {
            "input_shape": tuple(STATE_CACHE.shape),
            "output_shape": (HEAD,),
            "in_dtype": str(STATE_CACHE.dtype),
            "out_dtype": "fp8(nope 448) + bf16(rope 64)",
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
            "Kernel: _fused_kv_compress_norm_rope_insert_sparse_attn\n",
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
                f"OVERLAP={OVERLAP} (coff={COFF})\n"
            )
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output (dequant nope + rope bf16, rtol/atol=1e-2):\n")
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
