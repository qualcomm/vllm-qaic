"""
Standalone QAIC validation for `_eplb_map_and_record_i32_kernel`.

Source under test:
vllm/model_executor/layers/fused_moe/router/base_router.py
  - _eplb_map_and_record_i32_kernel (map logical expert ids -> physical replica
    ids via a logical->physical map with a hashed replica selection, and
    atomically record per-physical-expert load counts).

NOTE: in the source the kernel (and its launcher) are defined only inside
`if current_platform.is_cuda_alike():`, so they are not importable on the QAIC
path. We therefore embed a byte-for-byte copy of the kernel here (the object
under test) and drive it directly.

Kernel logic (per flattened topk entry `offs`):
  expert_id     = topk_ids[offs]
  valid         = 0 <= expert_id < num_logical_experts
  replica_count = max(logical_replica_count[expert_id], 1)
  token_idx     = offs // num_active_experts
  hashed        = (token_idx * 2654435769) & 0xFFFFFFFF   # Knuth mult. hash
  replica_idx   = hashed % replica_count
  physical_id   = logical_to_physical[expert_id * map_slots + replica_idx]
  out_ids[offs] = physical_id
  if record_enabled and 0 <= physical_id < out_size:
      atomic_add(out[physical_id], 1)

Determinism: integer atomic_add is commutative, so the recorded counts are
independent of program/thread interleaving. We use a single program (BLOCK_SIZE
>= numel) for clarity. Integer kernel -> exact equality on both the mapped ids
and the recorded per-expert counts.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_eplb_map_and_record_i32_kernel.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fused_moe/router/base_router.py"
KERNEL_NAME = "_eplb_map_and_record_i32_kernel"

KNUTH_MULTIPLIER = 2654435769


# Byte-for-byte copy of the source kernel (see module docstring).
@triton.jit
def _eplb_map_and_record_i32_kernel(
    topk_ids_ptr,
    logical_replica_count_ptr,
    logical_to_physical_ptr,
    out_ids_ptr,
    out_ptr,
    record_enabled_ptr,
    num_logical_experts,
    map_slots,
    out_size,
    numel,
    num_active_experts,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    expert_id = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int64)
    valid_expert = (expert_id >= 0) & (expert_id < num_logical_experts)
    safe_expert_id = tl.where(valid_expert, expert_id, 0)

    replica_count = tl.load(
        logical_replica_count_ptr + safe_expert_id,
        mask=mask & valid_expert,
        other=1,
    )
    replica_count = tl.maximum(replica_count, 1)
    KNUTH_MULTIPLIER = 2654435769
    token_idx = (offs // num_active_experts).to(tl.int64)
    hashed = (token_idx * KNUTH_MULTIPLIER) & 0xFFFFFFFF
    replica_idx = hashed % replica_count

    map_index = safe_expert_id * map_slots + replica_idx
    physical_id = tl.load(
        logical_to_physical_ptr + map_index,
        mask=mask & valid_expert,
        other=-1,
    )
    tl.store(out_ids_ptr + offs, physical_id, mask=mask)

    record_enabled = tl.load(record_enabled_ptr) != 0
    valid = mask & record_enabled & (physical_id >= 0) & (physical_id < out_size)
    safe_physical_id = tl.where(physical_id >= 0, physical_id, 0)
    tl.atomic_add(out_ptr + safe_physical_id, 1, mask=valid)


DEVICE = "qaic"
NUM_TOKENS = 8
NUM_ACTIVE_EXPERTS = 4  # topk
NUM_LOGICAL_EXPERTS = 6
MAP_SLOTS = 3  # max replicas per logical expert
NUM_PHYSICAL_EXPERTS = 12  # out_size
NUMEL = NUM_TOKENS * NUM_ACTIVE_EXPERTS
BLOCK_SIZE = triton.next_power_of_2(NUMEL)  # single program

torch.manual_seed(42)
TOPK_IDS = torch.randint(
    0, NUM_LOGICAL_EXPERTS, (NUM_TOKENS, NUM_ACTIVE_EXPERTS), dtype=torch.int32,
    device=DEVICE,
)
# replicas per logical expert in [1, MAP_SLOTS].
LOGICAL_REPLICA_COUNT = torch.randint(
    1, MAP_SLOTS + 1, (NUM_LOGICAL_EXPERTS,), dtype=torch.int32, device=DEVICE
)
# logical->physical map: [num_logical_experts, map_slots], values in physical range.
LOGICAL_TO_PHYSICAL = torch.randint(
    0, NUM_PHYSICAL_EXPERTS, (NUM_LOGICAL_EXPERTS, MAP_SLOTS), dtype=torch.int32,
    device=DEVICE,
)
RECORD_ENABLED = torch.tensor([1], dtype=torch.int32, device=DEVICE)


def _log(status, stats, error_text, ts):
    os.makedirs(LOG_DIR, exist_ok=True)
    lines = [
        f"{ts}\n",
        f"Kernel: {KERNEL_NAME}\n",
        f"Kernel file: {KERNEL_FILE_PATH}\n",
        f"Device target: QAIC (device='{DEVICE}')\n",
        f"Status: {status}\n\n",
    ]
    if status == "SUCCESS":
        for k, v in stats.items():
            lines.append(f"- {k}: {v}\n")
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
        lines.append("Error:\n" + error_text + "\n")
    lines.append("\n------------------------------------\n\n")
    with open(LOG_FILE, "a") as f:
        f.write("".join(lines))


def pytorch_ref():
    topk = TOPK_IDS.cpu().reshape(-1).to(torch.int64)
    replica_count = LOGICAL_REPLICA_COUNT.cpu().to(torch.int64)
    l2p = LOGICAL_TO_PHYSICAL.cpu().reshape(-1).to(torch.int64)
    record_enabled = int(RECORD_ENABLED.cpu().item()) != 0

    out_ids = torch.empty(NUMEL, dtype=torch.int64)
    counts = torch.zeros(NUM_PHYSICAL_EXPERTS, dtype=torch.int64)
    for offs in range(NUMEL):
        expert_id = int(topk[offs].item())
        valid_expert = 0 <= expert_id < NUM_LOGICAL_EXPERTS
        safe_expert_id = expert_id if valid_expert else 0
        rc = int(replica_count[safe_expert_id].item()) if valid_expert else 1
        rc = max(rc, 1)
        token_idx = offs // NUM_ACTIVE_EXPERTS
        hashed = (token_idx * KNUTH_MULTIPLIER) & 0xFFFFFFFF
        replica_idx = hashed % rc
        if valid_expert:
            map_index = safe_expert_id * MAP_SLOTS + replica_idx
            physical_id = int(l2p[map_index].item())
        else:
            physical_id = -1
        out_ids[offs] = physical_id
        if record_enabled and 0 <= physical_id < NUM_PHYSICAL_EXPERTS:
            counts[physical_id] += 1
    return out_ids, counts


def kernel_impl():
    topk_in = TOPK_IDS.contiguous().to(torch.int32)
    out_ids = torch.empty(NUMEL, dtype=torch.int32, device=DEVICE)
    out_counts = torch.zeros(NUM_PHYSICAL_EXPERTS, dtype=torch.int32, device=DEVICE)
    grid = (triton.cdiv(NUMEL, BLOCK_SIZE),)
    _eplb_map_and_record_i32_kernel[grid](
        topk_in,
        LOGICAL_REPLICA_COUNT.contiguous(),
        LOGICAL_TO_PHYSICAL.contiguous(),
        out_ids,
        out_counts,
        RECORD_ENABLED,
        NUM_LOGICAL_EXPERTS,
        MAP_SLOTS,
        NUM_PHYSICAL_EXPERTS,
        NUMEL,
        NUM_ACTIVE_EXPERTS,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out_ids, out_counts


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
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_ids, ref_counts = pytorch_ref()
        out_ids, out_counts = kernel_impl()
        ids_cpu = out_ids.cpu().to(torch.int64)
        counts_cpu = out_counts.cpu().to(torch.int64)

        ids_mismatch = int((ids_cpu != ref_ids).sum().item())
        counts_mismatch = int((counts_cpu != ref_counts).sum().item())
        assert ids_mismatch == 0, f"mapped id mismatch count={ids_mismatch}"
        assert counts_mismatch == 0, f"recorded count mismatch={counts_mismatch}"

        stats = {
            "topk_ids_shape": tuple(TOPK_IDS.shape),
            "out_ids_shape": tuple(out_ids.shape),
            "out_counts_shape": tuple(out_counts.shape),
            "dtype": str(out_ids.dtype),
            "device": str(out_ids.device),
            "num_logical_experts": NUM_LOGICAL_EXPERTS,
            "map_slots": MAP_SLOTS,
            "num_physical_experts": NUM_PHYSICAL_EXPERTS,
            "ids_mismatch_count": ids_mismatch,
            "counts_mismatch_count": counts_mismatch,
            "max_abs_diff": 0,
            "comparison": "exact integer equality (mapped ids + recorded counts)",
            "determinism": "integer atomic_add is commutative; single program used",
            "kernel_file": KERNEL_FILE_PATH,
            "timestamp": ts,
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
        _log(status, stats, error_text, ts)
    return status


if __name__ == "__main__":
    main()
