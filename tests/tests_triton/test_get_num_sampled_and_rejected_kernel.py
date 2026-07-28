"""
Standalone QAIC validation for `_get_num_sampled_and_rejected_kernel`.

Source under test:
vllm/v1/worker/gpu/input_batch.py
  - _get_num_sampled_and_rejected_kernel  (via launcher
    `get_num_sampled_and_rejected`)

For each request (batch_idx):
    is_chunked_prefilling = seq_len < prefill_len
    num_sampled = 0 if is_chunked_prefilling else num_sampled  (mutated in place)
    num_logits  = cu_num_logits[b+1] - cu_num_logits[b]
    num_rejected = 0 if is_chunked_prefilling else (num_logits - num_sampled)

Config tested: 4 requests mixing decode rows and one chunked-prefill row
(seq_len < prefill_len) to exercise the zeroing branch. Both output tensors
(num_sampled mutated + num_rejected) are validated with EXACT integer equality.
Reference: pure-PyTorch replication.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_get_num_sampled_and_rejected_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/input_batch.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected  # noqa: E402

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
NUM_REQS = 4
IDX_MAPPING = torch.arange(NUM_REQS, dtype=torch.int32, device=DEVICE)
# num_logits per req = [3, 3, 2, 1]
CU_NUM_LOGITS = torch.tensor([0, 3, 6, 8, 9], dtype=torch.int32, device=DEVICE)
# req 2 is chunked prefill: seq_len < prefill_len -> zeroed.
SEQ_LENS = torch.tensor([10, 12, 3, 8], dtype=torch.int32, device=DEVICE)
PREFILL_LEN = torch.tensor([5, 5, 20, 5], dtype=torch.int32, device=DEVICE)
NUM_SAMPLED_BASE = torch.tensor([2, 3, 1, 1], dtype=torch.int32, device=DEVICE)


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
    """Pure PyTorch replication. Returns (num_sampled, num_rejected)."""
    idx_mapping = IDX_MAPPING.cpu()
    cnl = CU_NUM_LOGITS.cpu()
    seq_lens = SEQ_LENS.cpu()
    prefill_len = PREFILL_LEN.cpu()
    num_sampled = NUM_SAMPLED_BASE.clone().cpu()
    num_rejected = torch.empty_like(num_sampled)
    for b in range(NUM_REQS):
        req_state_idx = int(idx_mapping[b])
        is_chunked = int(seq_lens[b]) < int(prefill_len[req_state_idx])
        ns = 0 if is_chunked else int(num_sampled[b])
        num_sampled[b] = ns
        num_logits = int(cnl[b + 1]) - int(cnl[b])
        nr = num_logits - ns
        num_rejected[b] = 0 if is_chunked else nr
    return num_sampled, num_rejected


def kernel_impl():
    """Kernel launch only. Returns (num_sampled, num_rejected)."""
    num_sampled = NUM_SAMPLED_BASE.clone()
    ns, nr = get_num_sampled_and_rejected(
        num_sampled,
        SEQ_LENS,
        CU_NUM_LOGITS,
        IDX_MAPPING,
        PREFILL_LEN,
    )
    return ns, nr


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_ns, ref_nr = pytorch_ref()
        kern_ns, kern_nr = kernel_impl()

        ref_ns_cpu, ref_nr_cpu = ref_ns.cpu(), ref_nr.cpu()
        kern_ns_cpu, kern_nr_cpu = kern_ns.cpu(), kern_nr.cpu()

        assert torch.equal(kern_ns_cpu, ref_ns_cpu), (
            f"num_sampled mismatch ref={ref_ns_cpu.tolist()} "
            f"k={kern_ns_cpu.tolist()}"
        )
        assert torch.equal(kern_nr_cpu, ref_nr_cpu), (
            f"num_rejected mismatch ref={ref_nr_cpu.tolist()} "
            f"k={kern_nr_cpu.tolist()}"
        )

        stats = {
            "input_shape": tuple(NUM_SAMPLED_BASE.shape),
            "output_shape": tuple(kern_nr.shape),
            "in_dtype": str(NUM_SAMPLED_BASE.dtype),
            "out_dtype": str(kern_nr.dtype),
            "device": str(NUM_SAMPLED_BASE.device),
            "max_abs_diff": 0,
            "mean_abs_diff": 0.0,
            "num_sampled": kern_ns_cpu.tolist(),
            "num_rejected": kern_nr_cpu.tolist(),
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
            "Kernel: _get_num_sampled_and_rejected_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- input shape: {stats['input_shape']}\n",
                f"- in dtype: {stats['in_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- output shape: {stats['output_shape']}\n",
                f"- out dtype: {stats['out_dtype']}\n",
                f"- num_sampled: {stats['num_sampled']}\n",
                f"- num_rejected: {stats['num_rejected']}\n",
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
