"""
Standalone QAIC validation for `_bf16_mla_sparse_kernel`.

Source under test:
vllm/v1/attention/ops/xpu_mla_sparse.py
  - _bf16_mla_sparse_kernel  (BF16 sparse MLA *prefill* attention over the
    top-k selected KV indices, with the query/key head split into a
    no-position-embedding (NoPE) part and a rotary (RoPE) part.)

For every query token / query head, the kernel gathers its `index_topk`
selected KV rows (`indices[token, kv_head, :]`), computes attention scores as
    score = sm_scale * (q_nope . k_nope + q_rope . k_rope)
does an online (base-2) softmax over just those selected rows, and produces
three outputs:
  * out        = softmax(score) @ v          (v = kv[..., :d_v])
  * max_logits = max_j score[j]              (natural-log domain)
  * lse        = logsumexp_j score[j]        (natural-log domain)

NoPE/RoPE score combination:
  The kernel loads q_nope over dims [0, BLOCK_DMODEL) and q_rope over dims
  [BLOCK_DMODEL, dim_qk), and likewise for k, then *sums* the two dot
  products (`qk = q_nope@k_nope; qk += q_rope@k_rope`). Because the two dim
  ranges are disjoint and concatenated, this sum is exactly the full
  q . k dot product over all `dim_qk` dims, so the reference simply performs
  the full-width scaled dot product (still gathering the NoPE and RoPE halves
  and summing their contributions explicitly for faithfulness).

The kernel scales by `sm_scale * LOG2E` and uses exp2, which is
mathematically equivalent to natural-exp softmax with the original
`sm_scale`; `max_logits`/`lse` are converted back to the natural-log domain
inside the kernel, so the reference uses plain natural-log softmax.

Reference: pure PyTorch gather + full-width scaled dot product + softmax.

Note on precision: the kernel operates in bfloat16 (q/k/v are bf16 and its
matmuls accumulate through bf16), and `out` is emitted as bf16. We cast both
sides to float32 before comparing. We attempt rtol/atol=1e-3; bf16's limited
mantissa may require loosening to ~1e-2 on real hardware (documented here).
"""

import datetime
import math
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_bf16_mla_sparse_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/xpu_mla_sparse.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.attention.ops.xpu_mla_sparse import (  # noqa: E402
    triton_bf16_mla_sparse_interface,
)

# ---------------------------------------------------------------------------
# Global inputs (shared by both implementations)
# ---------------------------------------------------------------------------
DEVICE = "qaic"

NUM_TOKENS = 2  # query tokens
NUM_HEADS_Q = 2  # query heads
NUM_HEADS_KV = 1  # kernel only supports kv head == 1
SEQ_KV = 32  # number of KV rows available to select from
D_V = 512  # value dim (kernel asserts d_v == 512)
BLOCK_DPE = 64  # RoPE portion of dim_qk
NOPE_DIM = D_V  # NoPE portion (== BLOCK_DMODEL)
DIM_QK = NOPE_DIM + BLOCK_DPE  # 576
TOPK = 16  # index_topk (must be a multiple of BLOCK_N == 16)
SM_SCALE = 1.0 / math.sqrt(DIM_QK)

torch.manual_seed(42)

# q: [num_tokens, num_heads_q, dim_qk]; kv: [seq_kv, num_heads_kv, dim_qk].
Q = (0.1 * torch.randn(NUM_TOKENS, NUM_HEADS_Q, DIM_QK, dtype=torch.float32)).to(
    dtype=torch.bfloat16, device=DEVICE
)
KV = (0.1 * torch.randn(SEQ_KV, NUM_HEADS_KV, DIM_QK, dtype=torch.float32)).to(
    dtype=torch.bfloat16, device=DEVICE
)
# indices: [num_tokens, num_heads_kv, topk]; distinct valid rows in [0, seq_kv).
_perm = torch.stack(
    [torch.randperm(SEQ_KV)[:TOPK] for _ in range(NUM_TOKENS * NUM_HEADS_KV)]
)
INDICES = _perm.reshape(NUM_TOKENS, NUM_HEADS_KV, TOPK).to(
    dtype=torch.int32, device=DEVICE
)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


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


def pytorch_ref(q, kv, indices, sm_scale, d_v, block_dpe):
    """Pure PyTorch reference for BF16 sparse MLA prefill attention.

    Returns (out, max_logits, lse) matching the kernel's three outputs.
    Pure PyTorch only -- no custom / Triton / vLLM / QAIC kernel calls.
    """
    q = q.float().cpu()
    kv = kv.float().cpu()
    indices = indices.cpu()

    num_tokens, num_heads_q, dim_qk = q.shape
    nope_dim = dim_qk - block_dpe

    out = torch.zeros(num_tokens, num_heads_q, d_v, dtype=torch.float32)
    max_logits = torch.zeros(num_tokens, num_heads_q, dtype=torch.float32)
    lse = torch.zeros(num_tokens, num_heads_q, dtype=torch.float32)

    kv_group_num = num_heads_q // num_heads_kv_of(kv)

    for t in range(num_tokens):
        for h in range(num_heads_q):
            kv_head = h // kv_group_num  # always 0 here (single kv head)
            sel = indices[t, kv_head]  # [topk]
            valid = (sel >= 0) & (sel < kv.shape[0])
            sel_valid = sel[valid].long()

            k_sel = kv[sel_valid, kv_head]  # [n_sel, dim_qk]
            k_nope = k_sel[:, :nope_dim]  # [n_sel, nope_dim]
            k_rope = k_sel[:, nope_dim:]  # [n_sel, block_dpe]
            v_sel = k_sel[:, :d_v]  # v = kv[..., :d_v]

            q_nope = q[t, h, :nope_dim]  # [nope_dim]
            q_rope = q[t, h, nope_dim:]  # [block_dpe]

            # NoPE + RoPE contributions summed (== full-width q.k).
            score = sm_scale * (
                q_nope @ k_nope.transpose(0, 1) + q_rope @ k_rope.transpose(0, 1)
            )  # [n_sel]

            m = score.max()
            p = torch.exp(score - m)
            denom = p.sum()
            out[t, h] = (p / denom) @ v_sel
            max_logits[t, h] = m
            lse[t, h] = m + torch.log(denom)

    return out, max_logits, lse


def num_heads_kv_of(kv):
    return kv.shape[1]


def kernel_impl(q, kv, indices, sm_scale, d_v, block_dpe):
    """Kernel wrapper: launch only."""
    out, max_logits, lse = triton_bf16_mla_sparse_interface(
        q,
        kv,
        indices,
        sm_scale,
        d_v=d_v,
        block_dpe=block_dpe,
    )
    return out, max_logits, lse


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out, ref_ml, ref_lse = pytorch_ref(
            Q, KV, INDICES, SM_SCALE, D_V, BLOCK_DPE
        )
        k_out, k_ml, k_lse = kernel_impl(Q, KV, INDICES, SM_SCALE, D_V, BLOCK_DPE)

        k_out = k_out.float().cpu()
        k_ml = k_ml.float().cpu()
        k_lse = k_lse.float().cpu()

        torch.testing.assert_close(k_out, ref_out, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(k_ml, ref_ml, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(k_lse, ref_lse, rtol=1e-3, atol=1e-3)

        out_diff = (k_out - ref_out).abs()
        ml_diff = (k_ml - ref_ml).abs()
        lse_diff = (k_lse - ref_lse).abs()

        stats = {
            "q_shape": tuple(Q.shape),
            "kv_shape": tuple(KV.shape),
            "indices_shape": tuple(INDICES.shape),
            "out_shape": tuple(k_out.shape),
            "max_logits_shape": tuple(k_ml.shape),
            "lse_shape": tuple(k_lse.shape),
            "in_dtype": str(Q.dtype),
            "out_dtype": "torch.float32 (cast from bf16)",
            "device": str(Q.device),
            "sm_scale": SM_SCALE,
            "topk": TOPK,
            "out_max_abs_diff": out_diff.max().item(),
            "out_mean_abs_diff": out_diff.mean().item(),
            "out_rel_err": (out_diff.max() / (ref_out.abs().max() + 1e-8)).item(),
            "max_logits_max_abs_diff": ml_diff.max().item(),
            "lse_max_abs_diff": lse_diff.max().item(),
            "grid": (f"({NUM_TOKENS},{NUM_HEADS_Q})", "BLOCK_H=16,BLOCK_N=16"),
        }

        pt_stats = _bench(lambda: pytorch_ref(Q, KV, INDICES, SM_SCALE, D_V, BLOCK_DPE))
        kern_stats = _bench(
            lambda: kernel_impl(Q, KV, INDICES, SM_SCALE, D_V, BLOCK_DPE)
        )
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
            "Kernel: _bf16_mla_sparse_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- q shape: {stats['q_shape']}\n")
            lines.append(f"- kv shape: {stats['kv_shape']}\n")
            lines.append(f"- indices shape: {stats['indices_shape']}\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n")
            lines.append(f"- sm_scale: {stats['sm_scale']}\n")
            lines.append(f"- topk: {stats['topk']}\n\n")
            lines.append("Grid Configuration:\n")
            lines.append(f"- grid: {stats['grid'][0]}\n")
            lines.append(f"- blocks: {stats['grid'][1]}\n\n")
            lines.append("Outputs:\n")
            lines.append(f"- out shape: {stats['out_shape']}\n")
            lines.append(f"- max_logits shape: {stats['max_logits_shape']}\n")
            lines.append(f"- lse shape: {stats['lse_shape']}\n")
            lines.append(f"- out dtype: {stats['out_dtype']}\n")
            lines.append(f"- out max_abs_diff: {stats['out_max_abs_diff']}\n")
            lines.append(f"- out mean_abs_diff: {stats['out_mean_abs_diff']}\n")
            lines.append(f"- out rel_err: {stats['out_rel_err']}\n")
            lines.append(
                f"- max_logits max_abs_diff: {stats['max_logits_max_abs_diff']}\n"
            )
            lines.append(f"- lse max_abs_diff: {stats['lse_max_abs_diff']}\n")
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
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))

    return status


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result == "SUCCESS" else 1)
