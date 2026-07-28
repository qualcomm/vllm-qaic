"""
Standalone QAIC validation for `copy_and_expand_dflash_inputs_kernel`.

Source under test:
vllm/v1/spec_decode/utils.py
  - copy_and_expand_dflash_inputs_kernel  (DFlash first-pass input setup)

Per request (2D grid: axis0=req, axis1=token block) the kernel:
  1. Copies context positions from target_positions -> out_context_positions.
  2. Computes query positions (last_ctx_pos + 1 + query_off) -> out_query_positions.
  3. Writes query input_ids: [bonus_token, mask, mask, ...].
  4. Computes slot_mapping for context and query positions via block_table lookup
     (slot = block_id * block_size + pos % block_size).
  5. Writes token_indices_to_sample for the mask (speculative) tokens.

Config tested: 3 requests, context lens [4, 2, 3], num_speculative_tokens=2
(num_query_per_req=3), HAS_NUM_REJECTED=False, single token block.
There is no python launcher in the source, so we launch the @triton.jit kernel
directly. All outputs are integer indices -> EXACT (torch.equal) comparison.
Reference: pure PyTorch replication of the per-request loop.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_copy_and_expand_dflash_inputs_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/spec_decode/utils.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.spec_decode.utils import (  # noqa: E402
    copy_and_expand_dflash_inputs_kernel,
)

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
NUM_REQS = 3
NUM_SPECULATIVE_TOKENS = 2
NUM_QUERY_PER_REQ = NUM_SPECULATIVE_TOKENS + 1  # bonus + spec tokens
BLOCK_SIZE_KV = 4
MAX_REQS = 3
MAX_BLOCKS = 8
PARALLEL_DRAFTING_TOKEN_ID = 999
BLOCK = 16  # >= max per-request total tokens (num_ctx + num_query_per_req)

CTX_LENS = [4, 2, 3]
_qsl = [0]
for c in CTX_LENS:
    _qsl.append(_qsl[-1] + c)
QUERY_START_LOC = torch.tensor(_qsl, dtype=torch.int32, device=DEVICE)
NUM_CONTEXT = _qsl[-1]  # 9
NUM_QUERY_TOTAL = NUM_REQS * NUM_QUERY_PER_REQ  # 9
TOTAL_INPUT_TOKENS = NUM_CONTEXT

TARGET_POSITIONS = torch.tensor(
    [0, 1, 2, 3, 10, 11, 20, 21, 22], dtype=torch.int32, device=DEVICE
)
NEXT_TOKEN_IDS = torch.tensor([5001, 5002, 5003], dtype=torch.int32, device=DEVICE)
BLOCK_TABLE = torch.randint(
    0, 50, (MAX_REQS, MAX_BLOCKS), dtype=torch.int32, device=DEVICE
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


def pytorch_ref():
    qsl = QUERY_START_LOC.cpu().tolist()
    tpos = TARGET_POSITIONS.cpu().tolist()
    nti = NEXT_TOKEN_IDS.cpu().tolist()
    bt = BLOCK_TABLE.cpu()

    out_input_ids = torch.zeros(NUM_QUERY_TOTAL, dtype=torch.int32)
    out_ctx_pos = torch.zeros(NUM_CONTEXT, dtype=torch.int32)
    out_query_pos = torch.zeros(NUM_QUERY_TOTAL, dtype=torch.int32)
    out_ctx_slot = torch.zeros(NUM_CONTEXT, dtype=torch.int64)
    out_query_slot = torch.zeros(NUM_QUERY_TOTAL, dtype=torch.int64)
    out_token_idx = torch.zeros(
        NUM_REQS * NUM_SPECULATIVE_TOKENS, dtype=torch.int32
    )

    for req in range(NUM_REQS):
        ctx_start = qsl[req]
        ctx_end = qsl[req + 1]
        num_ctx = ctx_end - ctx_start
        valid_ctx_end = ctx_end  # HAS_NUM_REJECTED=False
        last_pos = tpos[valid_ctx_end - 1]

        # Context positions & slots
        for k in range(num_ctx):
            pos = tpos[ctx_start + k]
            out_ctx_pos[ctx_start + k] = pos
            block_num = min(pos // BLOCK_SIZE_KV, MAX_BLOCKS - 1)
            block_id = int(bt[req, block_num].item())
            out_ctx_slot[ctx_start + k] = block_id * BLOCK_SIZE_KV + pos % BLOCK_SIZE_KV

        # Query positions, slots, input ids, token indices
        for off in range(NUM_QUERY_PER_REQ):
            query_out = req * NUM_QUERY_PER_REQ + off
            qpos = last_pos + 1 + off
            out_query_pos[query_out] = qpos
            block_num = min(qpos // BLOCK_SIZE_KV, MAX_BLOCKS - 1)
            block_id = int(bt[req, block_num].item())
            out_query_slot[query_out] = block_id * BLOCK_SIZE_KV + qpos % BLOCK_SIZE_KV
            out_input_ids[query_out] = (
                nti[req] if off == 0 else PARALLEL_DRAFTING_TOKEN_ID
            )
            if off > 0:
                sample_idx = req * NUM_SPECULATIVE_TOKENS + (off - 1)
                out_token_idx[sample_idx] = query_out

    return (
        out_input_ids,
        out_ctx_pos,
        out_query_pos,
        out_ctx_slot,
        out_query_slot,
        out_token_idx,
    )


def kernel_impl():
    out_input_ids = torch.zeros(NUM_QUERY_TOTAL, dtype=torch.int32, device=DEVICE)
    out_ctx_pos = torch.zeros(NUM_CONTEXT, dtype=torch.int32, device=DEVICE)
    out_query_pos = torch.zeros(NUM_QUERY_TOTAL, dtype=torch.int32, device=DEVICE)
    out_ctx_slot = torch.zeros(NUM_CONTEXT, dtype=torch.int64, device=DEVICE)
    out_query_slot = torch.zeros(NUM_QUERY_TOTAL, dtype=torch.int64, device=DEVICE)
    out_token_idx = torch.zeros(
        NUM_REQS * NUM_SPECULATIVE_TOKENS, dtype=torch.int32, device=DEVICE
    )

    grid = (NUM_REQS, 1)
    copy_and_expand_dflash_inputs_kernel[grid](
        NEXT_TOKEN_IDS,
        TARGET_POSITIONS,
        out_input_ids,
        out_ctx_pos,
        out_query_pos,
        out_ctx_slot,
        out_query_slot,
        out_token_idx,
        BLOCK_TABLE,
        BLOCK_TABLE.stride(0),
        QUERY_START_LOC,
        None,
        PARALLEL_DRAFTING_TOKEN_ID,
        BLOCK_SIZE_KV,
        NUM_QUERY_PER_REQ,
        NUM_SPECULATIVE_TOKENS,
        TOTAL_INPUT_TOKENS,
        BLOCK_SIZE=BLOCK,
        HAS_NUM_REJECTED=False,
    )
    return (
        out_input_ids,
        out_ctx_pos,
        out_query_pos,
        out_ctx_slot,
        out_query_slot,
        out_token_idx,
    )


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref()
        kern = kernel_impl()
        names = [
            "input_ids",
            "context_positions",
            "query_positions",
            "context_slot_mapping",
            "query_slot_mapping",
            "token_indices",
        ]
        for name, r, k in zip(names, ref, kern):
            kc = k.cpu()
            assert torch.equal(kc, r), (
                f"{name} mismatch: ref={r.tolist()} kern={kc.tolist()}"
            )

        stats = {
            "input_shape": tuple(TARGET_POSITIONS.shape),
            "output_shape": tuple(kern[0].shape),
            "in_dtype": str(TARGET_POSITIONS.dtype),
            "out_dtype": str(kern[0].dtype),
            "device": str(TARGET_POSITIONS.device),
            "max_abs_diff": 0,
            "mean_abs_diff": 0,
        }
        pt_stats = _bench(pytorch_ref)
        kern_stats = _bench(kernel_impl)
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
            "Kernel: copy_and_expand_dflash_inputs_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- input shape: {stats['input_shape']}\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- out dtype: {stats['out_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            lines.append("- relative_error: 0.0 (exact integer match)\n")
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
