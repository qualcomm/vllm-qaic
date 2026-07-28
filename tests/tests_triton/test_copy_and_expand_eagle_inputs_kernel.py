"""
Standalone QAIC validation for `copy_and_expand_eagle_inputs_kernel`.

Source under test:
vllm/v1/spec_decode/utils.py
  - copy_and_expand_eagle_inputs_kernel  (copy/expand target model inputs into
    the Eagle drafting buffers)

Grid is (num_reqs, num_token_blocks). Per request the output region for the
draft buffers is laid out as:
  [0, num_valid_tokens)                      -> copied target token ids
  [num_valid_tokens]                         -> bonus token (from next_token_ids)
  (num_valid_tokens, +num_padding_slots)     -> parallel drafting slots (masked)
  [num_valid_tokens+num_padding_slots, end)  -> rejected slots (padding id)
Positions increment from the request's first target position. It also records
new-token sampling indices for the bonus + parallel-draft slots.

Config tested: 2 requests, shift_input_ids=False, num_padding_slots_per_request=2.
No python launcher exists, so we launch the @triton.jit kernel directly with a
single token block. All outputs are integer indices / boolean masks -> EXACT
(torch.equal) comparison. Reference: pure PyTorch replication of the region
logic. The hidden_state_mapping output is only written on the shift path and is
not exercised here.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_copy_and_expand_eagle_inputs_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/spec_decode/utils.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.spec_decode.utils import (  # noqa: E402
    copy_and_expand_eagle_inputs_kernel,
)

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
NUM_REQS = 2
NUM_PADDING_SLOTS = 2
PADDING_TOKEN_ID = -1
PARALLEL_DRAFTING_TOKEN_ID = 888
SHIFT_INPUT_IDS = False
BLOCK = 16

# query_start_loc has num_reqs + 1 entries.
QUERY_START_LOC = torch.tensor([0, 4, 6], dtype=torch.int32, device=DEVICE)
# query_end_loc[req]: index of last VALID token for the request.
QUERY_END_LOC = torch.tensor([2, 5], dtype=torch.int32, device=DEVICE)
TOTAL_INPUT_TOKENS = 6

TARGET_TOKEN_IDS = torch.tensor(
    [100, 101, 102, 103, 104, 105], dtype=torch.int32, device=DEVICE
)
TARGET_POSITIONS = torch.tensor(
    [10, 11, 12, 13, 20, 21], dtype=torch.int32, device=DEVICE
)
NEXT_TOKEN_IDS = torch.tensor([7000, 7001], dtype=torch.int32, device=DEVICE)

# Output buffer sizes (derived from the config layout above).
OUT_TOTAL = 10  # req0 slots [0,6), req1 slots [6,10)
NEW_TOKEN_TOTAL = NUM_PADDING_SLOTS * NUM_REQS  # 4


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
    qel = QUERY_END_LOC.cpu().tolist()
    tok = TARGET_TOKEN_IDS.cpu().tolist()
    pos = TARGET_POSITIONS.cpu().tolist()
    nti = NEXT_TOKEN_IDS.cpu().tolist()

    out_input_ids = torch.zeros(OUT_TOTAL, dtype=torch.int32)
    out_positions = torch.zeros(OUT_TOTAL, dtype=torch.int32)
    out_rejected = torch.zeros(OUT_TOTAL, dtype=torch.bool)
    out_masked = torch.zeros(OUT_TOTAL, dtype=torch.bool)
    out_new_token_indices = torch.zeros(NEW_TOKEN_TOTAL, dtype=torch.int32)

    for req in range(NUM_REQS):
        query_start = qsl[req]
        next_query_start = qsl[req + 1]
        query_end = qel[req]

        # shift_input_ids == False
        num_valid = query_end - query_start + 1
        input_offset = 0
        output_start = query_start + req * NUM_PADDING_SLOTS

        num_rejected = next_query_start - query_end - 1
        total_output = num_valid + NUM_PADDING_SLOTS + num_rejected
        start_pos = pos[query_start]
        bonus_token = nti[req]

        for j in range(total_output):
            out_idx = output_start + j
            is_valid = j < num_valid
            is_bonus = j == num_valid
            is_parallel = (j > num_valid) and (j < num_valid + NUM_PADDING_SLOTS)
            is_rejected = j >= num_valid + NUM_PADDING_SLOTS

            if is_valid:
                token = tok[query_start + input_offset + j]
            elif is_bonus:
                token = bonus_token
            elif is_parallel:
                token = PARALLEL_DRAFTING_TOKEN_ID
            else:  # rejected
                token = PADDING_TOKEN_ID
            out_input_ids[out_idx] = token

            out_positions[out_idx] = 0 if is_rejected else start_pos + j
            out_rejected[out_idx] = is_rejected
            out_masked[out_idx] = is_parallel

            is_new_token = (j >= num_valid) and (j < num_valid + NUM_PADDING_SLOTS)
            if is_new_token:
                local = j - num_valid
                out_new_token_indices[req * NUM_PADDING_SLOTS + local] = out_idx

    return out_input_ids, out_positions, out_rejected, out_masked, out_new_token_indices


def kernel_impl():
    out_input_ids = torch.zeros(OUT_TOTAL, dtype=torch.int32, device=DEVICE)
    out_positions = torch.zeros(OUT_TOTAL, dtype=torch.int32, device=DEVICE)
    out_rejected = torch.zeros(OUT_TOTAL, dtype=torch.bool, device=DEVICE)
    out_masked = torch.zeros(OUT_TOTAL, dtype=torch.bool, device=DEVICE)
    out_new_token_indices = torch.zeros(
        NEW_TOKEN_TOTAL, dtype=torch.int32, device=DEVICE
    )
    out_hidden_state_mapping = torch.zeros(
        TOTAL_INPUT_TOKENS, dtype=torch.int32, device=DEVICE
    )

    grid = (NUM_REQS, 1)
    copy_and_expand_eagle_inputs_kernel[grid](
        TARGET_TOKEN_IDS,
        TARGET_POSITIONS,
        NEXT_TOKEN_IDS,
        out_input_ids,
        out_positions,
        out_rejected,
        out_masked,
        out_new_token_indices,
        out_hidden_state_mapping,
        QUERY_START_LOC,
        QUERY_END_LOC,
        PADDING_TOKEN_ID,
        PARALLEL_DRAFTING_TOKEN_ID,
        TOTAL_INPUT_TOKENS,
        NUM_PADDING_SLOTS,
        SHIFT_INPUT_IDS,
        BLOCK_SIZE_TOKENS=BLOCK,
    )
    return out_input_ids, out_positions, out_rejected, out_masked, out_new_token_indices


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
            "positions",
            "is_rejected_mask",
            "is_masked_mask",
            "new_token_indices",
        ]
        for name, r, k in zip(names, ref, kern):
            kc = k.cpu()
            assert torch.equal(kc, r), (
                f"{name} mismatch: ref={r.tolist()} kern={kc.tolist()}"
            )

        stats = {
            "input_shape": tuple(TARGET_TOKEN_IDS.shape),
            "output_shape": tuple(kern[0].shape),
            "in_dtype": str(TARGET_TOKEN_IDS.dtype),
            "out_dtype": str(kern[0].dtype),
            "device": str(TARGET_TOKEN_IDS.device),
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
            "Kernel: copy_and_expand_eagle_inputs_kernel\n",
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
            lines.append("- relative_error: 0.0 (exact integer/mask match)\n")
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
