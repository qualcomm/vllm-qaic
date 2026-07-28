"""
Standalone QAIC validation for `postprocess_mamba_fused_kernel`.

Source under test:
vllm/v1/worker/mamba_utils.py
  - postprocess_mamba_fused_kernel  (launched from
    MambaSpecDecodeGPUContext.run_fused_postprocess)

This fused kernel computes the mamba spec-decode "align" decision per request
and, when a copy is needed, moves one mamba state block to its destination
block via raw byte-address memcpy. The real launcher
(`run_fused_postprocess`) requires a fully-initialized
`MambaSpecDecodeGPUContext` bound to live forward-context state pointers and
per-group block tables, so we replicate its exact launch inline in
`kernel_impl` with hand-built metadata (rule 2: no cheaply-callable launcher).

We exercise the TEMPORAL-state path (conv_width == 0), which is the clean copy
case: for each request needing a copy,
    running          = num_computed + num_scheduled - num_draft
    new_num_computed = running + num_accepted - 1
    aligned          = (new_num_computed // block_size) * block_size
    needs_copy       = aligned >= running
    accept_bias      = aligned - running
    dest_block_idx   = aligned // block_size - 1
    actual_src_idx   = mamba_state_idx + accept_bias
    state[block_table[r, actual_src_idx]] <- state[block_table[r, dest_block_idx]]  (dst<-src)
and num_accepted_out[r] is set to 1 only when src_block_idx == dest_block_idx.

Config: block_size=4, 2 requests (one triggers a full-block copy, one hits the
needs_copy=False early return), 1 mamba group / 1 layer / 1 temporal state
(total_states=1). State tensor [16, 8] float32 with per-block distinct values.
Outputs (mutated state tensor + num_accepted_out) are validated with EXACT
integer/float equality against a pure-PyTorch replication.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_postprocess_mamba_fused_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/mamba_utils.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.worker.mamba_utils import postprocess_mamba_fused_kernel  # noqa: E402

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
BLOCK_SIZE = 4
COPY_BLOCK_SIZE = 1024
NUM_REQS = 2
TOTAL_STATES = 1  # 1 layer * 1 (temporal) state type

NUM_PHYS_BLOCKS = 16
INNER = 8  # elements per block
ELEM_SIZE = 4  # float32

# Per-request decision inputs.
NUM_ACCEPTED = torch.tensor([3, 1], dtype=torch.int32, device=DEVICE)
MAMBA_STATE_IDX = torch.tensor([0, 0], dtype=torch.int32, device=DEVICE)
NUM_SCHEDULED = torch.tensor([6, 3], dtype=torch.int32, device=DEVICE)
NUM_COMPUTED = torch.tensor([0, 0], dtype=torch.int32, device=DEVICE)
NUM_DRAFT = torch.tensor([0, 0], dtype=torch.int32, device=DEVICE)

# block_table[req, block_idx] -> physical block id in the state tensor.
BLOCK_TABLE = torch.tensor(
    [[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int32, device=DEVICE
)
BLOCK_TABLE_STRIDE_REQ = BLOCK_TABLE.stride(0)

# State tensor: distinct values per physical block so a copy is observable.
STATE_BASE = torch.empty(NUM_PHYS_BLOCKS, INNER, dtype=torch.float32, device=DEVICE)
for _b in range(NUM_PHYS_BLOCKS):
    STATE_BASE[_b] = _b * 100 + torch.arange(INNER, dtype=torch.float32, device=DEVICE)


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
    """Pure PyTorch replication. Returns (state, num_accepted_out)."""
    state = STATE_BASE.clone().cpu()
    num_accepted_out = NUM_ACCEPTED.clone().cpu()
    bt = BLOCK_TABLE.cpu()
    na = NUM_ACCEPTED.cpu()
    msi = MAMBA_STATE_IDX.cpu()
    ns = NUM_SCHEDULED.cpu()
    nc = NUM_COMPUTED.cpu()
    nd = NUM_DRAFT.cpu()
    for r in range(NUM_REQS):
        running = int(nc[r]) + int(ns[r]) - int(nd[r])
        new_num_computed = running + int(na[r]) - 1
        aligned = (new_num_computed // BLOCK_SIZE) * BLOCK_SIZE
        if not (aligned >= running):
            continue
        accept_bias = aligned - running
        src_block_idx = int(msi[r])
        dest_block_idx = aligned // BLOCK_SIZE - 1
        # Temporal state copy.
        actual_src_idx = src_block_idx + accept_bias
        actual_src_id = int(bt[r, actual_src_idx])
        dest_id = int(bt[r, dest_block_idx])
        if src_block_idx == dest_block_idx:
            num_accepted_out[r] = 1
        if src_block_idx == dest_block_idx and accept_bias == 0:
            continue
        # Kernel copies src_addr -> dst_addr where
        #   src_addr = state[actual_src_id], dst_addr = state[dest_id]
        state[dest_id] = state[actual_src_id].clone()
    return state, num_accepted_out


def kernel_impl():
    """Kernel launch only, replicating run_fused_postprocess."""
    state = STATE_BASE.clone()

    state_base_addrs = torch.tensor(
        [state.data_ptr()], dtype=torch.int64, device=DEVICE
    )
    state_block_strides = torch.tensor(
        [state.stride(0) * ELEM_SIZE], dtype=torch.int64, device=DEVICE
    )
    state_elem_sizes = torch.tensor([ELEM_SIZE], dtype=torch.int32, device=DEVICE)
    state_inner_sizes = torch.tensor([INNER], dtype=torch.int64, device=DEVICE)
    state_conv_widths = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    state_group_indices = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    block_table_ptrs = torch.tensor(
        [BLOCK_TABLE.data_ptr()], dtype=torch.int64, device=DEVICE
    )
    num_accepted_out = NUM_ACCEPTED.clone()

    grid = (NUM_REQS, TOTAL_STATES)
    postprocess_mamba_fused_kernel[grid](
        NUM_ACCEPTED,
        MAMBA_STATE_IDX,
        NUM_SCHEDULED,
        NUM_COMPUTED,
        NUM_DRAFT,
        block_table_ptrs,
        BLOCK_TABLE_STRIDE_REQ,
        state_base_addrs,
        state_block_strides,
        state_elem_sizes,
        state_inner_sizes,
        state_conv_widths,
        state_group_indices,
        num_accepted_out,
        NUM_REQS,
        block_size=BLOCK_SIZE,
        COPY_BLOCK_SIZE=COPY_BLOCK_SIZE,
    )
    return state, num_accepted_out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_state, ref_na = pytorch_ref()
        kern_state, kern_na = kernel_impl()

        ref_state_cpu = ref_state.cpu()
        kern_state_cpu = kern_state.cpu()
        ref_na_cpu = ref_na.cpu()
        kern_na_cpu = kern_na.cpu()

        assert torch.equal(kern_state_cpu, ref_state_cpu), "state copy mismatch"
        assert torch.equal(kern_na_cpu, ref_na_cpu), (
            f"num_accepted_out mismatch ref={ref_na_cpu.tolist()} "
            f"k={kern_na_cpu.tolist()}"
        )

        diff = (kern_state_cpu - ref_state_cpu).abs()
        stats = {
            "input_shape": tuple(STATE_BASE.shape),
            "output_shape": tuple(kern_state.shape),
            "in_dtype": str(STATE_BASE.dtype),
            "out_dtype": str(kern_state.dtype),
            "device": str(STATE_BASE.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "num_accepted_out": kern_na_cpu.tolist(),
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
            "Kernel: postprocess_mamba_fused_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- input shape (state): {stats['input_shape']}\n",
                f"- in dtype: {stats['in_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- output shape: {stats['output_shape']}\n",
                f"- out dtype: {stats['out_dtype']}\n",
                f"- num_accepted_out: {stats['num_accepted_out']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
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
