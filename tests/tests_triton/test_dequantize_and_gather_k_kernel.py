"""
Standalone QAIC validation for `_dequantize_and_gather_k_kernel`.

Source under test:
vllm/models/deepseek_v4/common/ops/cache_utils.py
  - _dequantize_and_gather_k_kernel  (gather the last `gather_len` tokens of
    each sequence from the DeepseekV4 paged K-cache and dequantize UE8M0 FP8 K
    back to bf16, writing into a dense [num_reqs, max_tokens, 512] output).
    Launched via the public wrapper `dequantize_and_gather_k_cache_triton`
    (the CUDA-only cutedsl fast path in `dequantize_and_gather_k_cache` is
    bypassed by calling the triton wrapper directly).

Per token the kernel:
    block_in_seq = pos // block_size;  pos_in_block = pos % block_size
    physical_block = block_table[req, block_in_seq]
    for each 64-elem quant block g in [0,7):
        exp   = scale_byte[g] - 127;  scale = 2^exp
        out[..0:448] = fp8_val * scale      (fp8 -> fp32 -> bf16)
    out[..448:512] = raw bf16 passthrough
The output token index is (offset + i) where i in [0, gather_len).

Validation: build the packed uint8 cache in PURE PyTorch (UE8M0 encode), run the
kernel, and compare the kernel's dense output against a pure-PyTorch
decode+gather of the same cache. This exercises the real dequant path
numerically.

Config tested: num_reqs=2, block_size=64, 1 block/seq, seq_lens=[40, 24],
gather all tokens, offset=0, 512 dims. Output dtype bf16.
Reference: pure PyTorch UE8M0 dequant + gather.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_dequantize_and_gather_k_kernel.txt")
KERNEL_FILE_PATH = "vllm/models/deepseek_v4/common/ops/cache_utils.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.models.deepseek_v4.common.ops.cache_utils import (  # noqa: E402
    dequantize_and_gather_k_cache_triton,
)

torch.manual_seed(42)

FP8_DIM = 448
BF16_DIM = 64
INPUT_DIM = 512
SCALE_DIM = 8
QUANT_BLOCK = 64
N_QUANT_BLOCKS = 7
FP8_MAX = 448.0
TOKEN_DATA_SIZE = FP8_DIM + BF16_DIM * 2  # 576
BLOCK_SIZE = 64
NUM_BLOCKS = 4
BLOCK_BYTES = (
    (BLOCK_SIZE * TOKEN_DATA_SIZE + BLOCK_SIZE * SCALE_DIM + TOKEN_DATA_SIZE - 1)
    // TOKEN_DATA_SIZE
) * TOKEN_DATA_SIZE

FP8_DTYPE = torch.float8_e4m3fn

NUM_REQS = 2
SEQ_LENS = torch.tensor([40, 24], dtype=torch.int32, device=DEVICE)
MAX_TOKENS = int(SEQ_LENS.max().item())
MAX_BLOCKS = 1
# Distinct physical blocks per request.
BLOCK_TABLE = torch.tensor(
    [[1], [2]], dtype=torch.int32, device=DEVICE
)
OFFSET = 0

# Raw K values (fp32) we will encode into the cache. [num_reqs, max_tokens, 512]
K_VALS = torch.randn(
    NUM_REQS, MAX_TOKENS, INPUT_DIM, dtype=torch.float32, device=DEVICE
) * 3.0


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


def _encode_token(x_row):
    """UE8M0 encode one 512-dim K row into (fp8_uint8[448], scale_u8[8], bf16[64]).

    Pure PyTorch. Mirrors quantize_and_insert_k_kernel.
    """
    x = x_row.float()
    fp8_bytes = torch.zeros(FP8_DIM, dtype=torch.uint8)
    scale_bytes = torch.zeros(SCALE_DIM, dtype=torch.uint8)
    for g in range(N_QUANT_BLOCKS):
        blk = x[g * QUANT_BLOCK:(g + 1) * QUANT_BLOCK]
        block_max = torch.clamp(blk.abs().max(), min=1e-4)
        exp = torch.ceil(torch.log2(block_max / FP8_MAX))
        scale = torch.exp2(exp)
        q = torch.clamp(blk / scale, -FP8_MAX, FP8_MAX).to(FP8_DTYPE)
        fp8_bytes[g * QUANT_BLOCK:(g + 1) * QUANT_BLOCK] = q.view(torch.uint8)
        enc = torch.clamp(exp + 127.0, 0.0, 255.0)
        scale_bytes[g] = int(enc.item())
    bf16_vals = x_row[FP8_DIM:INPUT_DIM].to(torch.bfloat16)
    bf16_bytes = bf16_vals.view(torch.uint8)
    return fp8_bytes, scale_bytes, bf16_bytes


def build_cache():
    """Build the packed uint8 K-cache from K_VALS in pure PyTorch (on CPU)."""
    cache = torch.zeros(NUM_BLOCKS, BLOCK_BYTES, dtype=torch.uint8)
    kv = K_VALS.cpu()
    bt = BLOCK_TABLE.cpu()
    seq = SEQ_LENS.cpu()
    for req in range(NUM_REQS):
        for pos in range(int(seq[req].item())):
            block_in_seq = pos // BLOCK_SIZE
            phys = int(bt[req, block_in_seq].item())
            pib = pos % BLOCK_SIZE
            fp8_bytes, scale_bytes, bf16_bytes = _encode_token(kv[req, pos])
            base = pib * TOKEN_DATA_SIZE
            cache[phys, base:base + FP8_DIM] = fp8_bytes
            cache[phys, base + FP8_DIM:base + TOKEN_DATA_SIZE] = bf16_bytes
            sbase = BLOCK_SIZE * TOKEN_DATA_SIZE + pib * SCALE_DIM
            cache[phys, sbase:sbase + SCALE_DIM] = scale_bytes
    return cache.to(DEVICE)


K_CACHE = build_cache()


def pytorch_ref():
    """Pure PyTorch decode+gather of the cache -> [num_reqs, max_tokens, 512]."""
    cache = K_CACHE.cpu()
    bt = BLOCK_TABLE.cpu()
    seq = SEQ_LENS.cpu()
    out = torch.zeros(NUM_REQS, MAX_TOKENS, INPUT_DIM, dtype=torch.bfloat16)
    for req in range(NUM_REQS):
        gather_len = int(seq[req].item())
        start_pos = int(seq[req].item()) - gather_len
        for i in range(gather_len):
            pos = start_pos + i
            block_in_seq = pos // BLOCK_SIZE
            phys = int(bt[req, block_in_seq].item())
            pib = pos % BLOCK_SIZE
            base = pib * TOKEN_DATA_SIZE
            tok = cache[phys, base:base + TOKEN_DATA_SIZE]
            fp8_vals = tok[0:FP8_DIM].contiguous().view(FP8_DTYPE).float()
            bf16_vals = tok[FP8_DIM:TOKEN_DATA_SIZE].contiguous().view(torch.bfloat16)
            sbase = BLOCK_SIZE * TOKEN_DATA_SIZE + pib * SCALE_DIM
            scale_bytes = cache[phys, sbase:sbase + SCALE_DIM]
            row = torch.zeros(INPUT_DIM, dtype=torch.float32)
            for g in range(N_QUANT_BLOCKS):
                exp = float(scale_bytes[g].item()) - 127.0
                scale = 2.0 ** exp
                row[g * QUANT_BLOCK:(g + 1) * QUANT_BLOCK] = (
                    fp8_vals[g * QUANT_BLOCK:(g + 1) * QUANT_BLOCK] * scale
                )
            out[req, OFFSET + i, 0:FP8_DIM] = row[0:FP8_DIM].to(torch.bfloat16)
            out[req, OFFSET + i, FP8_DIM:INPUT_DIM] = bf16_vals
    return out


def kernel_impl():
    """Kernel launch only. Returns the dense gathered/dequantized output."""
    out = torch.zeros(
        NUM_REQS, MAX_TOKENS, INPUT_DIM, dtype=torch.bfloat16, device=DEVICE
    )
    dequantize_and_gather_k_cache_triton(
        out,
        K_CACHE,
        SEQ_LENS,
        None,  # gather_lens=None -> gather all seq tokens
        BLOCK_TABLE,
        BLOCK_SIZE,
        OFFSET,
    )
    return out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref()
        kernel_out = kernel_impl()

        ref_cpu = ref_out.cpu().float()
        kernel_cpu = kernel_out.cpu().float()
        # Only compare rows that were actually written (valid seq positions).
        mask = torch.zeros(NUM_REQS, MAX_TOKENS, dtype=torch.bool)
        seq = SEQ_LENS.cpu()
        for req in range(NUM_REQS):
            mask[req, OFFSET:OFFSET + int(seq[req].item())] = True
        rr = ref_cpu[mask]
        kr = kernel_cpu[mask]
        torch.testing.assert_close(kr, rr, rtol=1e-2, atol=1e-2)

        diff = (kr - rr).abs()
        stats = {
            "input_shape": tuple(K_CACHE.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(K_CACHE.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(K_CACHE.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }

        pt_stats = _bench(pytorch_ref)
        kern_stats = _bench(kernel_impl)
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
            "Kernel: _dequantize_and_gather_k_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- cache shape: {stats['input_shape']}\n",
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
