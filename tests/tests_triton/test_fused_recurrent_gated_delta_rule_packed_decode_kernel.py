"""
Standalone QAIC validation for `fused_recurrent_gated_delta_rule_packed_decode_kernel`.

Source under test:
vllm/model_executor/layers/fla/ops/fused_recurrent.py
  - fused_recurrent_gated_delta_rule_packed_decode_kernel  (single decode-step
    gated delta-rule update reading packed conv'd QKV, one step per batch
    item, indexed into a shared state bank via `ssm_state_indices`)
  - fused_recurrent_gated_delta_rule_packed_decode  (Python launcher)

`mixed_qkv` is [B, qkv_dim] with q/k/v packed as:
    q = mixed_qkv[:, h*K : h*K+K]
    k = mixed_qkv[:, H*K + h*K : H*K + h*K+K]
    v = mixed_qkv[:, 2*H*K + hv*V : 2*H*K + hv*V+V]
`a`, `b` are [B, HV] gating inputs; `A_log`, `dt_bias` are [HV].
`ssm_state_indices` [B] selects the state slot in `initial_state`
([num_states, HV, V, K]) that each batch item reads/writes (NULL_BLOCK_ID=0
sentinel skips the item -- avoided here by using indices >= 1).

Per batch item n, for hv in range(HV):
    state_idx = ssm_state_indices[n]
    h_state = initial_state[state_idx, hv]                      # [V, K]
    x = a[n, hv] + dt_bias[hv]
    softplus_x = log(1 + exp(x)) if x <= threshold(20.0) else x
    g = -exp(A_log[hv]) * softplus_x
    beta = sigmoid(b[n, hv])
    q_t = q[n, h] * scale ; k_t = k[n, h] ; v_t = v[n, hv]
    h_state *= exp(g)
    v_t -= h_state @ k_t
    v_t *= beta
    h_state += outer(v_t, k_t)
    o[n, hv] = h_state @ q_t
    initial_state[state_idx, hv] = h_state  (in-place)

Reference: pure PyTorch literal single-step recurrence matching this exactly.
"""

import datetime
import math
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(
    LOG_DIR, "log_fused_recurrent_gated_delta_rule_packed_decode_kernel.txt"
)
KERNEL_FILE_PATH = "vllm/model_executor/layers/fla/ops/fused_recurrent.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.model_executor.layers.fla.ops.fused_recurrent import (  # noqa: E402
    fused_recurrent_gated_delta_rule_packed_decode,
)

# ---------------------------------------------------------------------------
# Global inputs (shared by pytorch_ref and kernel_impl)
# ---------------------------------------------------------------------------
DEVICE = "qaic"
DTYPE = torch.float32

B = 2
H = 1
HV = 1
K = 16
V = 16
SCALE = K**-0.5
THRESHOLD = 20.0
USE_QK_L2NORM = False

QKV_DIM = 2 * H * K + HV * V  # q + k + v packed

torch.manual_seed(42)
MIXED_QKV = torch.randn(B, QKV_DIM, dtype=DTYPE, device=DEVICE)
A_GATE = torch.randn(B, HV, dtype=DTYPE, device=DEVICE)
B_GATE = torch.randn(B, HV, dtype=DTYPE, device=DEVICE)
A_LOG = torch.randn(HV, dtype=DTYPE, device=DEVICE)
DT_BIAS = torch.randn(HV, dtype=DTYPE, device=DEVICE)

NUM_STATES = 3
# batch item 0 -> state slot 1, batch item 1 -> state slot 2 (both non-null,
# non-overlapping, to exercise per-batch state indexing).
SSM_STATE_INDICES = torch.tensor([1, 2], dtype=torch.int32, device=DEVICE)
INITIAL_STATE = torch.randn(NUM_STATES, HV, V, K, dtype=DTYPE, device=DEVICE)
OUT = torch.zeros(B, 1, HV, V, dtype=DTYPE, device=DEVICE)


def pytorch_ref(mixed_qkv, a, b, a_log, dt_bias, scale, threshold, initial_state,
                ssm_state_indices, out_shape):
    """Pure PyTorch literal single decode-step recurrence, operating on a
    clone of `initial_state` so the shared global is left untouched (the
    kernel path mutates its own clone in-place separately).

    mixed_qkv: [B, qkv_dim] packed q/k/v.
    a, b: [B, HV] gating inputs. a_log, dt_bias: [HV].
    initial_state: [num_states, HV, V, K] state bank.
    Returns (out [B, 1, HV, V], updated_state_bank [num_states, HV, V, K]).
    """
    mixed_qkv = mixed_qkv.cpu().clone()
    a = a.cpu().clone()
    b = b.cpu().clone()
    a_log = a_log.cpu().clone()
    dt_bias = dt_bias.cpu().clone()
    ssm_state_indices = ssm_state_indices.cpu().clone()
    h_bank = initial_state.cpu().clone().to(torch.float32)

    Bd = mixed_qkv.shape[0]
    HVd = h_bank.shape[1]
    Hd = H

    out = torch.zeros(Bd, 1, HVd, V, dtype=torch.float32)

    for n in range(Bd):
        state_idx = int(ssm_state_indices[n].item())
        for hv in range(HVd):
            h = hv // (HVd // Hd)
            q_t = mixed_qkv[n, h * K : h * K + K].to(torch.float32)
            k_t = mixed_qkv[n, Hd * K + h * K : Hd * K + h * K + K].to(torch.float32)
            v_t = mixed_qkv[
                n, 2 * Hd * K + hv * V : 2 * Hd * K + hv * V + V
            ].to(torch.float32)

            x = a[n, hv].item() + dt_bias[hv].item()
            if x <= threshold:
                softplus_x = math.log(1.0 + math.exp(x))
            else:
                softplus_x = x
            g = -math.exp(a_log[hv].item()) * softplus_x
            beta = torch.sigmoid(b[n, hv]).item()

            q_t = q_t * scale

            h_state = h_bank[state_idx, hv].clone()  # [V, K]
            h_state = h_state * math.exp(g)
            v_t = v_t - (h_state * k_t.unsqueeze(0)).sum(dim=1)
            v_t = v_t * beta
            h_state = h_state + v_t.unsqueeze(1) * k_t.unsqueeze(0)
            o_t = (h_state * q_t.unsqueeze(0)).sum(dim=1)

            out[n, 0, hv] = o_t
            h_bank[state_idx, hv] = h_state

    return out, h_bank


def kernel_impl(mixed_qkv, a, b, a_log, dt_bias, scale, initial_state,
                ssm_state_indices):
    state = initial_state.clone()
    out = torch.zeros(mixed_qkv.shape[0], 1, state.shape[1], V, dtype=mixed_qkv.dtype,
                       device=mixed_qkv.device)
    out_result, final_state = fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=state,
        out=out,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=USE_QK_L2NORM,
    )
    return out_result, final_state


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
        ref_out, ref_state = pytorch_ref(
            MIXED_QKV, A_GATE, B_GATE, A_LOG, DT_BIAS, SCALE, THRESHOLD,
            INITIAL_STATE, SSM_STATE_INDICES, OUT.shape,
        )
        kernel_out, kernel_state = kernel_impl(
            MIXED_QKV, A_GATE, B_GATE, A_LOG, DT_BIAS, SCALE, INITIAL_STATE,
            SSM_STATE_INDICES,
        )

        ref_out_cpu = ref_out.cpu()
        kernel_out_cpu = kernel_out.cpu()
        ref_state_cpu = ref_state.cpu()
        kernel_state_cpu = kernel_state.cpu()

        torch.testing.assert_close(kernel_out_cpu, ref_out_cpu, rtol=1e-3, atol=1e-3)

        diff_o = (kernel_out_cpu - ref_out_cpu).abs()
        rel_err_o = (diff_o / (ref_out_cpu.abs() + 1e-8)).mean().item()
        diff_state = (kernel_state_cpu - ref_state_cpu).abs()

        stats = {
            "mixed_qkv_shape": tuple(MIXED_QKV.shape),
            "a_shape": tuple(A_GATE.shape),
            "b_shape": tuple(B_GATE.shape),
            "A_log_shape": tuple(A_LOG.shape),
            "dt_bias_shape": tuple(DT_BIAS.shape),
            "ssm_state_indices": SSM_STATE_INDICES.cpu().tolist(),
            "input_dtype": str(MIXED_QKV.dtype),
            "output_shape": tuple(kernel_out.shape),
            "output_dtype": str(kernel_out.dtype),
            "device": str(MIXED_QKV.device),
            "max_abs_diff_o": diff_o.max().item(),
            "mean_abs_diff_o": diff_o.mean().item(),
            "rel_error_o": rel_err_o,
            "max_abs_diff_state": diff_state.max().item(),
            "mean_abs_diff_state": diff_state.mean().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(
            MIXED_QKV, A_GATE, B_GATE, A_LOG, DT_BIAS, SCALE, THRESHOLD,
            INITIAL_STATE, SSM_STATE_INDICES, OUT.shape))
        kern_stats = _bench(lambda: kernel_impl(
            MIXED_QKV, A_GATE, B_GATE, A_LOG, DT_BIAS, SCALE, INITIAL_STATE,
            SSM_STATE_INDICES))
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
            "Kernel: fused_recurrent_gated_delta_rule_packed_decode_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- mixed_qkv shape: {stats['mixed_qkv_shape']}\n")
            lines.append(f"- a shape: {stats['a_shape']}\n")
            lines.append(f"- b shape: {stats['b_shape']}\n")
            lines.append(f"- A_log shape: {stats['A_log_shape']}\n")
            lines.append(f"- dt_bias shape: {stats['dt_bias_shape']}\n")
            lines.append(f"- ssm_state_indices: {stats['ssm_state_indices']}\n")
            lines.append(f"- input dtype: {stats['input_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- out shape: {stats['output_shape']}\n")
            lines.append(f"- out dtype: {stats['output_dtype']}\n")
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
