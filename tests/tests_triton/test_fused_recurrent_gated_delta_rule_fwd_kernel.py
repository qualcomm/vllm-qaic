"""
Standalone QAIC validation for `fused_recurrent_gated_delta_rule_fwd_kernel`.

Source under test:
vllm/model_executor/layers/fla/ops/fused_recurrent.py
  - fused_recurrent_gated_delta_rule_fwd_kernel  (sequential gated delta-rule
    recurrent linear-attention forward pass)
  - fused_recurrent_gated_delta_rule_fwd  (Python launcher)

Per (batch n, value-head hv), for t = 0..T-1:
    q_t = q[n, t, h] * scale               (h = hv // (HV // H), GQA repeat)
    h_state *= exp(g[n, t, hv])            (state shape [V, K])
    v_t = v[n, t, hv] - h_state @ k_t
    v_t *= beta[n, t, hv]                  (headwise beta, elementwise)
    h_state += outer(v_t, k_t)
    o_t = h_state @ q_t

Reference: pure PyTorch literal for-loop replicating this recurrence exactly.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(
    LOG_DIR, "log_fused_recurrent_gated_delta_rule_fwd_kernel.txt"
)
KERNEL_FILE_PATH = "vllm/model_executor/layers/fla/ops/fused_recurrent.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.model_executor.layers.fla.ops.fused_recurrent import (  # noqa: E402
    fused_recurrent_gated_delta_rule_fwd,
)

# ---------------------------------------------------------------------------
# Global inputs (shared by pytorch_ref and kernel_impl)
# ---------------------------------------------------------------------------
DEVICE = "qaic"
DTYPE = torch.float32

B = 1
T = 16
H = 1
HV = 1
K = 16
V = 16
SCALE = K**-0.5
USE_QK_L2NORM = False

torch.manual_seed(42)
Q = torch.randn(B, T, H, K, dtype=DTYPE, device=DEVICE)
Kt = torch.randn(B, T, H, K, dtype=DTYPE, device=DEVICE)
Vt = torch.randn(B, T, HV, V, dtype=DTYPE, device=DEVICE)
G = torch.randn(B, T, HV, dtype=DTYPE, device=DEVICE) * 0.1 - 0.5  # keep decay in a sane range
BETA = torch.rand(B, T, HV, V, dtype=DTYPE, device=DEVICE)  # headwise beta [B,T,HV,V]

# NOTE: when `inplace_final_state=True` the kernel unconditionally indexes
# `ssm_state_indices` for the per-timestep final-state store (regardless of
# whether continuous-batching / initial-state loading is otherwise in use),
# so a real (non-null, i.e. > 0) index tensor is required. We use a 2-slot
# state bank [num_states=2, HV, V, K] with slot 0 reserved as the unused
# NULL_BLOCK_ID sentinel and slot 1 as the actual state, selected by every
# (batch, timestep) via SSM_STATE_INDICES.
NUM_STATES = 2
STATE_SLOT = 1
INITIAL_STATE = torch.zeros(NUM_STATES, HV, V, K, dtype=DTYPE, device=DEVICE)
SSM_STATE_INDICES = torch.full((B, T), STATE_SLOT, dtype=torch.int32, device=DEVICE)


def pytorch_ref(q, k, v, g, beta, scale, initial_state, ssm_state_indices, state_slot):
    """Pure PyTorch literal recurrence matching the kernel step-by-step.

    q, k: [B, T, H, K]
    v, beta: [B, T, HV, V] (headwise beta)
    g: [B, T, HV]
    initial_state: [num_states, HV, V, K] state bank, indexed via
        ssm_state_indices (here constant == state_slot for every token).
    Returns (o [B, T, HV, V], final_state [num_states, HV, V, K]).
    """
    q = q.cpu().clone()
    k = k.cpu().clone()
    v = v.cpu().clone()
    g = g.cpu().clone()
    beta = beta.cpu().clone()
    h_bank = initial_state.cpu().clone().to(torch.float32)

    Bd, Td, Hd, Kd = q.shape
    HVd, Vd = v.shape[2], v.shape[3]
    o = torch.zeros(Bd, Td, HVd, Vd, dtype=torch.float32)

    for n in range(Bd):
        for hv in range(HVd):
            h = hv // (HVd // Hd)
            # state is loaded once, from the slot selected at t=0.
            h_state = h_bank[state_slot, hv].clone()  # [V, K]
            for t in range(Td):
                q_t = q[n, t, h].to(torch.float32) * scale  # [K]
                k_t = k[n, t, h].to(torch.float32)  # [K]
                v_t = v[n, t, hv].to(torch.float32)  # [V]
                g_t = g[n, t, hv].to(torch.float32)  # scalar

                h_state = h_state * torch.exp(g_t)
                v_t = v_t - (h_state * k_t.unsqueeze(0)).sum(dim=1)
                beta_t = beta[n, t, hv].to(torch.float32)  # [V]
                v_t = v_t * beta_t
                h_state = h_state + v_t.unsqueeze(1) * k_t.unsqueeze(0)
                o_t = (h_state * q_t.unsqueeze(0)).sum(dim=1)

                o[n, t, hv] = o_t
                # store into the slot selected for this (n, t)
                h_bank[state_slot, hv] = h_state
    return o, h_bank


def kernel_impl(q, k, v, g, beta, scale, initial_state, ssm_state_indices):
    state = initial_state.clone()
    o, final_state = fused_recurrent_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=None,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=USE_QK_L2NORM,
    )
    return o, final_state


def _bench(fn, warmup=3, iters=10):
    """Device-synced wall-clock benchmark. Returns dict of latency stats (ms)."""
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


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_o, ref_state = pytorch_ref(
            Q, Kt, Vt, G, BETA, SCALE, INITIAL_STATE, SSM_STATE_INDICES, STATE_SLOT
        )
        kernel_o, kernel_state = kernel_impl(
            Q, Kt, Vt, G, BETA, SCALE, INITIAL_STATE, SSM_STATE_INDICES
        )

        ref_o_cpu = ref_o.cpu()
        kernel_o_cpu = kernel_o.cpu()
        ref_state_cpu = ref_state.cpu()
        kernel_state_cpu = kernel_state.cpu()

        torch.testing.assert_close(kernel_o_cpu, ref_o_cpu, rtol=1e-3, atol=1e-3)

        diff_o = (kernel_o_cpu - ref_o_cpu).abs()
        rel_err_o = (diff_o / (ref_o_cpu.abs() + 1e-8)).mean().item()
        diff_state = (kernel_state_cpu - ref_state_cpu).abs()

        stats = {
            "q_shape": tuple(Q.shape),
            "k_shape": tuple(Kt.shape),
            "v_shape": tuple(Vt.shape),
            "g_shape": tuple(G.shape),
            "beta_shape": tuple(BETA.shape),
            "input_dtype": str(Q.dtype),
            "output_shape": tuple(kernel_o.shape),
            "output_dtype": str(kernel_o.dtype),
            "device": str(Q.device),
            "max_abs_diff_o": diff_o.max().item(),
            "mean_abs_diff_o": diff_o.mean().item(),
            "rel_error_o": rel_err_o,
            "max_abs_diff_state": diff_state.max().item(),
            "mean_abs_diff_state": diff_state.mean().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(
            Q, Kt, Vt, G, BETA, SCALE, INITIAL_STATE, SSM_STATE_INDICES, STATE_SLOT))
        kern_stats = _bench(lambda: kernel_impl(
            Q, Kt, Vt, G, BETA, SCALE, INITIAL_STATE, SSM_STATE_INDICES))
        speedup = kern_stats["avg_ms"] / pt_stats["avg_ms"] if pt_stats["avg_ms"] > 0 else float("nan")
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
            "Kernel: fused_recurrent_gated_delta_rule_fwd_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- q shape: {stats['q_shape']}\n")
            lines.append(f"- k shape: {stats['k_shape']}\n")
            lines.append(f"- v shape: {stats['v_shape']}\n")
            lines.append(f"- g shape: {stats['g_shape']}\n")
            lines.append(f"- beta shape: {stats['beta_shape']}\n")
            lines.append(f"- input dtype: {stats['input_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- o shape: {stats['output_shape']}\n")
            lines.append(f"- o dtype: {stats['output_dtype']}\n")
            lines.append(f"- max_abs_diff (o): {stats['max_abs_diff_o']}\n")
            lines.append(f"- mean_abs_diff (o): {stats['mean_abs_diff_o']}\n")
            lines.append(f"- rel_error (o): {stats['rel_error_o']}\n")
            lines.append(f"- max_abs_diff (final_state): {stats['max_abs_diff_state']}\n")
            lines.append(f"- mean_abs_diff (final_state): {stats['mean_abs_diff_state']}\n")
            if "pytorch_latency_ms" in stats:
                lines.append("Timing:\n")
                lines.append(f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                             f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                             f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                             f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n")
                lines.append(f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                             f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                             f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                             f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n")
                lines.append(f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n")
        else:
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))

    return status


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result == "SUCCESS" else 1)
