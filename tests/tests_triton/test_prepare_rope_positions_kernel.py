"""
Standalone QAIC validation for `_prepare_rope_positions_kernel`.

Source under test:
vllm/v1/worker/gpu/mm/rope.py
  - _prepare_rope_positions_kernel  (device kernel launched from
    RopeState.prepare_positions)

The real launcher (`RopeState.prepare_positions`) needs a fully constructed
model + config, so we replicate its exact launch here with a thin
`kernel_impl` wrapper that calls `_prepare_rope_positions_kernel[(num_reqs,)]`
with the same argument order / strides.

Per request (program batch_idx):
    req_state_idx = idx_mapping[batch_idx]
    is_prefill    = num_computed_tokens[req_state_idx] < prefill_lens[req_state_idx]
    for pos in query range:
        orig_pos = num_computed + block
        for j in NUM_DIMS:
            pos_j = prefill_positions[req_state_idx, j, orig_pos] if is_prefill
                    else orig_pos + delta[req_state_idx]
            positions[j, query_start + block] = pos_j

Config tested: NUM_DIMS=3 (M-RoPE), 3 requests mixing prefill and decode rows,
so both the table-lookup and the (orig_pos + delta) branches are exercised.
Output is integer positions -> EXACT integer equality.
Reference: pure-PyTorch replication.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_prepare_rope_positions_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/mm/rope.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.worker.gpu.mm.rope import _prepare_rope_positions_kernel  # noqa: E402

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
NUM_REQS = 3
NUM_DIMS = 3
MAX_MODEL_LEN = 32
BLOCK_SIZE = 1024

# batch_idx -> req_state_idx
IDX_MAPPING = torch.arange(NUM_REQS, dtype=torch.int32, device=DEVICE)
# Per req_state: prefill lengths / computed tokens decide the prefill branch.
PREFILL_LENS = torch.tensor([10, 4, 8], dtype=torch.int32, device=DEVICE)
NUM_COMPUTED = torch.tensor([0, 6, 2], dtype=torch.int32, device=DEVICE)
# -> req0: prefill (0<10), req1: decode (6>=4), req2: prefill (2<8)
# Query lens [4, 1, 3] -> query_start_loc [0,4,5,8]; total tokens = 8
QUERY_START_LOC = torch.tensor([0, 4, 5, 8], dtype=torch.int32, device=DEVICE)
TOTAL_TOKENS = 8
DELTA = torch.tensor([0, 100, 0], dtype=torch.int32, device=DEVICE)

# prefill_positions flattened as [NUM_REQS * NUM_DIMS, MAX_MODEL_LEN], int32,
# with distinct known values: value = req*10000 + dim*100 + pos.
PREFILL_POS = torch.empty(
    NUM_REQS * NUM_DIMS, MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE
)
for _req in range(NUM_REQS):
    for _dim in range(NUM_DIMS):
        _row = _req * NUM_DIMS + _dim
        PREFILL_POS[_row] = (
            _req * 10000
            + _dim * 100
            + torch.arange(MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE)
        )

POSITIONS_SHAPE = (NUM_DIMS, TOTAL_TOKENS + 1)


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
    """Pure PyTorch replication of the position preparation."""
    positions = torch.zeros(POSITIONS_SHAPE, dtype=torch.int64)
    idx_mapping = IDX_MAPPING.cpu()
    prefill_lens = PREFILL_LENS.cpu()
    num_computed = NUM_COMPUTED.cpu()
    qsl = QUERY_START_LOC.cpu()
    delta = DELTA.cpu()
    prefill_pos = PREFILL_POS.cpu()
    for b in range(NUM_REQS):
        rsi = int(idx_mapping[b])
        is_prefill = int(num_computed[rsi]) < int(prefill_lens[rsi])
        qstart = int(qsl[b])
        qend = int(qsl[b + 1])
        query_len = qend - qstart
        d = int(delta[rsi])
        nc = int(num_computed[rsi])
        for blk in range(query_len):
            orig_pos = nc + blk
            for j in range(NUM_DIMS):
                if is_prefill:
                    pos = int(prefill_pos[rsi * NUM_DIMS + j, orig_pos])
                else:
                    pos = orig_pos + d
                positions[j, qstart + blk] = pos
    return positions


def kernel_impl():
    """Kernel launch only, mirroring RopeState.prepare_positions."""
    positions = torch.zeros(POSITIONS_SHAPE, dtype=torch.int64, device=DEVICE)
    _prepare_rope_positions_kernel[(NUM_REQS,)](
        positions,
        positions.stride(0),
        PREFILL_POS,
        NUM_DIMS * MAX_MODEL_LEN,
        MAX_MODEL_LEN,
        DELTA,
        IDX_MAPPING,
        QUERY_START_LOC,
        PREFILL_LENS,
        NUM_COMPUTED,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_DIMS=NUM_DIMS,
    )
    return positions


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref()
        kernel_out = kernel_impl()

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        assert torch.equal(kernel_cpu, ref_cpu), (
            f"ref={ref_cpu.tolist()} k={kernel_cpu.tolist()}"
        )

        stats = {
            "input_shape": tuple(PREFILL_POS.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(PREFILL_POS.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(PREFILL_POS.device),
            "max_abs_diff": 0,
            "mean_abs_diff": 0.0,
            "positions": kernel_cpu.tolist(),
        }

        pt_stats = _bench(pytorch_ref)
        kern_stats = _bench(kernel_impl)
        speedup = (kern_stats["avg_ms"] / pt_stats["avg_ms"]
                   if pt_stats["avg_ms"] > 0 else float("nan"))
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
            "Kernel: _prepare_rope_positions_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- input shape (prefill_positions): {stats['input_shape']}\n",
                f"- in dtype: {stats['in_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- output shape: {stats['output_shape']}\n",
                f"- out dtype: {stats['out_dtype']}\n",
                f"- positions: {stats['positions']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']} (exact-match comparison)\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
            ]
            if "pytorch_latency_ms" in stats:
                lines += [
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
