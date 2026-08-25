"""Compare QAIC eager rejection-sampler kernels with upstream Triton.

Run this harness in a fresh process so ``QAIC_VISIBLE_DEVICES`` is set before
PyTorch imports the QAIC backend, for example:

    QAIC_VISIBLE_DEVICES=0 \
    HEXAGON_TOOLS=/prj/crd/austin/validation/scratch/users/eplatero/pytorch/pytorch_build_tools/hexagon_tools-21.0.02/Tools \
    HEXAGON_SDK_ROOT=/prj/crd/austin/validation/scratch/users/eplatero/pytorch/pytorch_build_tools/hexagon_sdk-6.5.0 \
    HEXAGON_ARCH_VERSION=68 QAIC_NUM_CORES=16 QAIC_NUM_THREADS=4 \
    .venv_eager/bin/python tests/test_qaic/spec_decode/benchmark_rejection_sampler_pyt_triton.py

The upstream module cannot be imported as the Triton reference implementation:
vLLM's Triton feature detection disables its normal launch path for QAIC.  The
harness therefore extracts the four JIT function definitions directly from the
specified upstream source file and binds them to Qualcomm Triton in this
process.  It never modifies the imported vLLM module or production shim.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import statistics
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch_qaic  # noqa: F401

import triton
import triton.language as tl

from vllm_qaic.v1.sample.rejection_sampler_shim import (
    _sample_recovered_tokens_kernel_pyt,
)
from vllm_qaic.v1.sample.rejection_sampler_triton import get_qaic_triton_kernels


_KERNEL_NAMES = (
    "expand_kernel",
    "rejection_greedy_sample_kernel",
    "rejection_random_sample_kernel",
    "sample_recovered_tokens_kernel",
)
_HYBRID_HANDOFF_CASE = "hybrid-handoff"
_HYBRID_PROFILES_CASE = "hybrid-profiles"


@triton.jit
def _qaic_two_reduce_recovered_tokens_kernel(
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    q_ptr,
    vocab_size,
    PADDED_VOCAB_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    pos = tl.program_id(1)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx
    if pos >= num_draft_tokens:
        return

    vocab_offset = tl.arange(0, PADDED_VOCAB_SIZE)
    token_idx = start_idx + pos
    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)
        prob = tl.load(
            target_probs_ptr + token_idx * vocab_size + vocab_offset,
            mask=vocab_offset < vocab_size,
            other=0,
        )
        prob = tl.where(vocab_offset == draft_token_id, 0.0, prob)
    else:
        draft_prob = tl.load(
            draft_probs_ptr + token_idx * vocab_size + vocab_offset,
            mask=vocab_offset < vocab_size,
            other=0,
        )
        target_prob = tl.load(
            target_probs_ptr + token_idx * vocab_size + vocab_offset,
            mask=vocab_offset < vocab_size,
            other=0,
        )
        prob = tl.maximum(target_prob - draft_prob, 0)

    q = tl.load(
        q_ptr + req_idx * vocab_size + vocab_offset,
        mask=vocab_offset < vocab_size,
        other=float("-inf"),
    )
    score = prob / q
    max_score = tl.max(score, axis=0)
    first_max_neg_offset = tl.max(
        tl.where(score == max_score, -vocab_offset, -PADDED_VOCAB_SIZE), axis=0
    )
    tl.store(output_token_ids_ptr + token_idx, -first_max_neg_offset)


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_size: int
    max_spec_len: int
    vocab_size: int
    warmup: int
    iterations: int


@dataclass
class KernelCase:
    name: str
    config: BenchmarkConfig
    inputs: dict[str, Any]
    expected: torch.Tensor
    pyt_output: torch.Tensor
    triton_output: torch.Tensor
    reset_pyt: Callable[[], None]
    reset_triton: Callable[[], None]
    pyt_launch: Callable[[], None]
    triton_launch: Callable[[], None]


def _summary(samples_ns: list[int]) -> dict[str, float]:
    ordered = sorted(samples_ns)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "median_us": statistics.median(samples_ns) / 1_000,
        "p95_us": ordered[p95_index] / 1_000,
        "samples_us": [sample / 1_000 for sample in samples_ns],
    }


def _synchronize() -> None:
    torch.qaic.synchronize()


def _measure_once(reset: Callable[[], None], operation: Callable[[], None]) -> float:
    reset()
    _synchronize()
    started = time.perf_counter_ns()
    operation()
    _synchronize()
    return (time.perf_counter_ns() - started) / 1_000


def _warm_up(
    iterations: int, reset: Callable[[], None], operation: Callable[[], None]
) -> None:
    for _ in range(iterations):
        reset()
        operation()
    _synchronize()


def _measure_steady_state(
    iterations: int, reset: Callable[[], None], operation: Callable[[], None]
) -> dict[str, float]:
    samples_ns: list[int] = []
    for _ in range(iterations):
        reset()
        _synchronize()
        started = time.perf_counter_ns()
        operation()
        _synchronize()
        samples_ns.append(time.perf_counter_ns() - started)
    return _summary(samples_ns)


def _default_upstream_source() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "vllm_0_15_0"
        / "vllm"
        / "v1"
        / "sample"
        / "rejection_sampler.py"
    )


def _load_upstream_kernels(source_path: Path) -> tuple[dict[str, Any], str]:
    """Load only the four upstream JIT definitions with real QAIC Triton."""
    source = source_path.read_text()
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    module = ast.parse(source, filename=str(source_path))
    definitions = {
        node.name: node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in _KERNEL_NAMES
    }
    missing = sorted(set(_KERNEL_NAMES) - definitions.keys())
    if missing:
        raise RuntimeError(
            f"Upstream source {source_path} does not define: {', '.join(missing)}"
        )

    namespace: dict[str, Any] = {
        "triton": triton,
        "tl": tl,
        "__name__": "__qaic_upstream_kernels__",
    }
    for name in _KERNEL_NAMES:
        definition = definitions[name]
        extracted = ast.Module(body=[definition], type_ignores=[])
        ast.fix_missing_locations(extracted)
        exec(compile(extracted, str(source_path), "exec"), namespace)

    return {name: namespace[name] for name in _KERNEL_NAMES}, source_hash


def _qaic_tensor(values: Any, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(values, dtype=dtype).to(device="qaic")


def _lengths(batch_size: int, max_spec_len: int, include_empty: bool) -> list[int]:
    if max_spec_len < 1:
        raise ValueError("max_spec_len must be at least one")
    lengths = [1 + index % max_spec_len for index in range(batch_size)]
    if include_empty and batch_size > 1:
        lengths[0] = 0
    return lengths


def _cumulative(lengths: Iterable[int]) -> list[int]:
    total = 0
    result: list[int] = []
    for length in lengths:
        total += length
        result.append(total)
    return result


def _new_output(batch_size: int, max_spec_len: int) -> torch.Tensor:
    return torch.full(
        (batch_size, max_spec_len + 1), -1, dtype=torch.int32, device="qaic"
    )


def _new_token_output(num_tokens: int) -> torch.Tensor:
    return torch.full((num_tokens,), -1, dtype=torch.int32, device="qaic")


def _reference_expand_kernel_pytorch(
    output: torch.Tensor,
    inputs: torch.Tensor,
    cumulative_tokens: torch.Tensor,
    replace_from: int,
    replace_to: int,
    MAX_NUM_TOKENS: int = 128,
) -> None:
    del MAX_NUM_TOKENS
    batch_size = inputs.shape[0]
    if batch_size == 0 or int(cumulative_tokens[-1].item()) == 0:
        return
    source = inputs.clone()
    source[source == replace_from] = replace_to
    previous = torch.zeros(
        batch_size,
        dtype=cumulative_tokens.dtype,
        device=cumulative_tokens.device,
    )
    previous[1:] = cumulative_tokens[:-1]
    counts = (cumulative_tokens - previous).to(torch.int64)
    request_indices = torch.repeat_interleave(
        torch.arange(batch_size, device=inputs.device), counts
    )
    output.copy_(source[request_indices])


def _reference_rejection_greedy_sample_kernel_pytorch(
    output: torch.Tensor,
    cumulative_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    target_argmax: torch.Tensor,
    bonus_tokens: torch.Tensor,
    is_greedy: torch.Tensor | None,
    max_spec_len: int,
    uniform_probs: torch.Tensor | None,
    synthetic_conditional_rates: torch.Tensor | None,
    SYNTHETIC_MODE: bool = False,
) -> None:
    del max_spec_len
    batch_size = int(cumulative_tokens.shape[0])
    previous = torch.zeros(
        batch_size,
        dtype=cumulative_tokens.dtype,
        device=cumulative_tokens.device,
    )
    previous[1:] = cumulative_tokens[:-1]

    for request_index in range(batch_size):
        if is_greedy is not None and not bool(is_greedy[request_index].item()):
            continue
        start_idx = int(previous[request_index].item())
        end_idx = int(cumulative_tokens[request_index].item())
        rejected = False
        for position in range(end_idx - start_idx):
            token_index = start_idx + position
            draft_token = draft_tokens[token_index]
            target_token = target_argmax[token_index].to(torch.int32)
            if SYNTHETIC_MODE:
                assert uniform_probs is not None
                assert synthetic_conditional_rates is not None
                accepted = bool(
                    (
                        uniform_probs[token_index]
                        < synthetic_conditional_rates[position]
                    ).item()
                )
                token = draft_token if accepted else target_token
            else:
                accepted = bool((draft_token == target_token).item())
                token = target_token
            output[request_index, position] = token
            if not accepted:
                rejected = True
                break
        if not rejected:
            output[request_index, end_idx - start_idx] = bonus_tokens[request_index]


def _reference_rejection_random_sample_kernel_pytorch(
    output: torch.Tensor,
    cumulative_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,
    bonus_tokens: torch.Tensor,
    recovered_tokens: torch.Tensor,
    uniform_probs: torch.Tensor,
    is_greedy: torch.Tensor,
    max_spec_len: int,
    vocab_size: int,
    synthetic_conditional_rates: torch.Tensor | None,
    NO_DRAFT_PROBS: bool = True,
    SYNTHETIC_MODE: bool = False,
) -> None:
    del max_spec_len, vocab_size
    batch_size = int(cumulative_tokens.shape[0])
    previous = torch.zeros(
        batch_size,
        dtype=cumulative_tokens.dtype,
        device=cumulative_tokens.device,
    )
    previous[1:] = cumulative_tokens[:-1]

    for request_index in range(batch_size):
        if bool(is_greedy[request_index].item()):
            continue
        start_idx = int(previous[request_index].item())
        end_idx = int(cumulative_tokens[request_index].item())
        rejected = False
        for position in range(end_idx - start_idx):
            token_index = start_idx + position
            draft_token = draft_tokens[token_index]
            if SYNTHETIC_MODE:
                assert synthetic_conditional_rates is not None
                accepted = bool(
                    (
                        uniform_probs[token_index]
                        < synthetic_conditional_rates[position]
                    ).item()
                )
            else:
                if NO_DRAFT_PROBS:
                    draft_probability = 1.0
                else:
                    assert draft_probs is not None
                    draft_probability = draft_probs[token_index, draft_token]
                target_probability = target_probs[token_index, draft_token]
                accepted = bool(
                    (
                        uniform_probs[token_index] * draft_probability
                        < target_probability
                    ).item()
                )
            token = draft_token if accepted else recovered_tokens[token_index]
            output[request_index, position] = token
            if not accepted:
                rejected = True
                break
        if not rejected:
            output[request_index, end_idx - start_idx] = bonus_tokens[request_index]


def _clone_for_validation(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone()


def _expand_case(kernels: dict[str, Any], config: BenchmarkConfig) -> KernelCase:
    lengths = _lengths(config.batch_size, config.max_spec_len, include_empty=True)
    cumulative = _cumulative(lengths)
    inputs = _qaic_tensor(
        [7 if index == 1 else 100 + index for index in range(config.batch_size)],
        dtype=torch.int32,
    )
    cu_tokens = _qaic_tensor(cumulative, dtype=torch.int32)
    pyt_output = _new_token_output(cumulative[-1])
    triton_output = _new_token_output(cumulative[-1])

    def pyt_launch() -> None:
        _reference_expand_kernel_pytorch(
            pyt_output,
            inputs,
            cu_tokens,
            7,
            70,
            MAX_NUM_TOKENS=config.max_spec_len,
        )

    def triton_launch() -> None:
        kernels["expand_kernel"][(config.batch_size,)](
            triton_output,
            inputs,
            cu_tokens,
            7,
            70,
            MAX_NUM_TOKENS=config.max_spec_len,
        )

    def reset_pyt() -> None:
        pyt_output.fill_(-1)

    def reset_triton() -> None:
        triton_output.fill_(-1)

    reset_pyt()
    pyt_launch()
    _synchronize()
    return KernelCase(
        name="expand_kernel",
        config=config,
        inputs={"lengths": lengths, "replace_from": 7, "replace_to": 70},
        expected=_clone_for_validation(pyt_output),
        pyt_output=pyt_output,
        triton_output=triton_output,
        reset_pyt=reset_pyt,
        reset_triton=reset_triton,
        pyt_launch=pyt_launch,
        triton_launch=triton_launch,
    )


def _greedy_case(kernels: dict[str, Any], config: BenchmarkConfig) -> KernelCase:
    lengths = _lengths(config.batch_size, config.max_spec_len, include_empty=True)
    cumulative = _cumulative(lengths)
    num_tokens = cumulative[-1]
    draft = _qaic_tensor(
        [index % config.vocab_size for index in range(num_tokens)], dtype=torch.int32
    )
    target = _qaic_tensor(
        [
            value if index % 3 else (value + 1) % config.vocab_size
            for index, value in enumerate(range(num_tokens))
        ],
        dtype=torch.int32,
    )
    bonus = _qaic_tensor(
        [config.vocab_size - 1 - index for index in range(config.batch_size)],
        dtype=torch.int32,
    )
    cu_tokens = _qaic_tensor(cumulative, dtype=torch.int32)
    is_greedy = _qaic_tensor(
        [index % 2 == 0 for index in range(config.batch_size)], dtype=torch.bool
    )
    pyt_output = _new_output(config.batch_size, config.max_spec_len)
    triton_output = _new_output(config.batch_size, config.max_spec_len)

    def pyt_launch() -> None:
        _reference_rejection_greedy_sample_kernel_pytorch(
            pyt_output,
            cu_tokens,
            draft,
            target,
            bonus,
            is_greedy,
            config.max_spec_len,
            None,
            None,
            SYNTHETIC_MODE=False,
        )

    def triton_launch() -> None:
        kernels["rejection_greedy_sample_kernel"][(config.batch_size,)](
            triton_output,
            cu_tokens,
            draft,
            target,
            bonus,
            is_greedy,
            config.max_spec_len,
            None,
            None,
            SYNTHETIC_MODE=False,
        )

    def reset_pyt() -> None:
        pyt_output.fill_(-1)

    def reset_triton() -> None:
        triton_output.fill_(-1)

    reset_pyt()
    pyt_launch()
    _synchronize()
    return KernelCase(
        name="rejection_greedy_sample_kernel",
        config=config,
        inputs={
            "lengths": lengths,
            "is_greedy": _clone_for_validation(is_greedy).tolist(),
        },
        expected=_clone_for_validation(pyt_output),
        pyt_output=pyt_output,
        triton_output=triton_output,
        reset_pyt=reset_pyt,
        reset_triton=reset_triton,
        pyt_launch=pyt_launch,
        triton_launch=triton_launch,
    )


def _random_case(kernels: dict[str, Any], config: BenchmarkConfig) -> KernelCase:
    lengths = _lengths(config.batch_size, config.max_spec_len, include_empty=True)
    cumulative = _cumulative(lengths)
    num_tokens = cumulative[-1]
    draft_values = [index % config.vocab_size for index in range(num_tokens)]
    draft = _qaic_tensor(draft_values, dtype=torch.int32)
    target_probs = torch.full(
        (num_tokens, config.vocab_size), 0.01, dtype=torch.float32
    )
    for index, token_id in enumerate(draft_values):
        target_probs[index, token_id] = 0.9 if index % 3 else 0.1
    target_probs = target_probs.to(device="qaic")
    bonus = _qaic_tensor(
        [config.vocab_size - 1 - index for index in range(config.batch_size)],
        dtype=torch.int32,
    )
    recovered = _qaic_tensor(
        [(value + 2) % config.vocab_size for value in draft_values], dtype=torch.int32
    )
    uniform_probs = _qaic_tensor(
        [0.5 if index % 3 else 0.8 for index in range(num_tokens)], dtype=torch.float32
    )
    cu_tokens = _qaic_tensor(cumulative, dtype=torch.int32)
    is_greedy = _qaic_tensor(
        [index % 2 == 0 for index in range(config.batch_size)], dtype=torch.bool
    )
    pyt_output = _new_output(config.batch_size, config.max_spec_len)
    triton_output = _new_output(config.batch_size, config.max_spec_len)

    def pyt_launch() -> None:
        _reference_rejection_random_sample_kernel_pytorch(
            pyt_output,
            cu_tokens,
            draft,
            None,
            target_probs,
            bonus,
            recovered,
            uniform_probs,
            is_greedy,
            config.max_spec_len,
            config.vocab_size,
            None,
            NO_DRAFT_PROBS=True,
            SYNTHETIC_MODE=False,
        )

    def triton_launch() -> None:
        kernels["rejection_random_sample_kernel"][(config.batch_size,)](
            triton_output,
            cu_tokens,
            draft,
            None,
            target_probs,
            bonus,
            recovered,
            uniform_probs,
            is_greedy,
            config.max_spec_len,
            config.vocab_size,
            None,
            NO_DRAFT_PROBS=True,
            SYNTHETIC_MODE=False,
        )

    def reset_pyt() -> None:
        pyt_output.fill_(-1)

    def reset_triton() -> None:
        triton_output.fill_(-1)

    reset_pyt()
    pyt_launch()
    _synchronize()
    return KernelCase(
        name="rejection_random_sample_kernel",
        config=config,
        inputs={"lengths": lengths, "no_draft_probs": True, "mixed_requests": True},
        expected=_clone_for_validation(pyt_output),
        pyt_output=pyt_output,
        triton_output=triton_output,
        reset_pyt=reset_pyt,
        reset_triton=reset_triton,
        pyt_launch=pyt_launch,
        triton_launch=triton_launch,
    )


def _recovered_case(kernels: dict[str, Any], config: BenchmarkConfig) -> KernelCase:
    lengths = _lengths(config.batch_size, config.max_spec_len, include_empty=True)
    cumulative = _cumulative(lengths)
    num_tokens = cumulative[-1]
    draft_values = [index % config.vocab_size for index in range(num_tokens)]
    draft = _qaic_tensor(draft_values, dtype=torch.int32)
    target_probs = (
        torch.linspace(
            0.001,
            1.0,
            num_tokens * config.vocab_size,
            dtype=torch.float32,
        )
        .reshape(num_tokens, config.vocab_size)
        .to(device="qaic")
    )
    q = (
        torch.linspace(
            0.5,
            1.5,
            config.batch_size * config.vocab_size,
            dtype=torch.float32,
        )
        .reshape(config.batch_size, config.vocab_size)
        .to(device="qaic")
    )
    inv_q = q.reciprocal()
    cu_tokens = _qaic_tensor(cumulative, dtype=torch.int32)
    padded_vocab_size = triton.next_power_of_2(config.vocab_size)
    pyt_output = _new_token_output(num_tokens)
    triton_output = _new_token_output(num_tokens)

    def pyt_launch() -> None:
        _sample_recovered_tokens_kernel_pyt(
            pyt_output,
            cu_tokens,
            draft,
            None,
            target_probs,
            inv_q,
            config.vocab_size,
            BLOCK_SIZE=padded_vocab_size,
            NO_DRAFT_PROBS=True,
            USE_FP64_GUMBEL=False,
        )

    def triton_launch() -> None:
        kernels["sample_recovered_tokens_kernel"][
            (
                config.batch_size,
                config.max_spec_len,
            )
        ](
            triton_output,
            cu_tokens,
            draft,
            None,
            target_probs,
            q,
            config.vocab_size,
            PADDED_VOCAB_SIZE=padded_vocab_size,
            NO_DRAFT_PROBS=True,
        )

    def reset_pyt() -> None:
        pyt_output.fill_(-1)

    def reset_triton() -> None:
        triton_output.fill_(-1)

    reset_pyt()
    pyt_launch()
    _synchronize()
    return KernelCase(
        name="sample_recovered_tokens_kernel",
        config=config,
        inputs={
            "lengths": lengths,
            "no_draft_probs": True,
            "padded_vocab_size": padded_vocab_size,
        },
        expected=_clone_for_validation(pyt_output),
        pyt_output=pyt_output,
        triton_output=triton_output,
        reset_pyt=reset_pyt,
        reset_triton=reset_triton,
        pyt_launch=pyt_launch,
        triton_launch=triton_launch,
    )


def _handoff_lengths(batch_size: int, max_spec_len: int) -> list[int]:
    if batch_size < 2:
        raise ValueError("hybrid-handoff requires --batch-size of at least two")
    return [0, *[1 + index % max_spec_len for index in range(batch_size - 1)]]


def _handoff_target_probs(
    num_tokens: int,
    vocab_size: int,
    draft_values: list[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    target_probs = torch.full((num_tokens, vocab_size), 0.01, dtype=dtype)
    for token_index, token_id in enumerate(draft_values):
        target_probs[token_index, token_id] = 0.75
        target_probs[token_index, (token_id + 1) % vocab_size] = 0.9
    return target_probs.to(device="qaic")


def _run_hybrid_handoff_matrix(
    random_kernel: Any, config: BenchmarkConfig
) -> dict[str, Any]:
    """Validate the production QAIC PyTorch-to-Triton random boundary.

    The actual eager sampler creates contiguous QAIC int32 draft IDs, runs the
    retained PyTorch recovered-token producer, and then passes its int32 QAIC
    output to the random rejection kernel.  Keep that exact boundary here:
    neither output is copied to CPU before the real Qualcomm-Triton launch.
    """
    lengths = _handoff_lengths(config.batch_size, config.max_spec_len)
    cumulative = _cumulative(lengths)
    num_tokens = cumulative[-1]
    draft_values = [index % config.vocab_size for index in range(num_tokens)]
    draft = _qaic_tensor(draft_values, dtype=torch.int32).contiguous()
    cu_tokens = _qaic_tensor(cumulative, dtype=torch.int32)
    bonus = _qaic_tensor(
        [config.vocab_size - 1 - index for index in range(config.batch_size)],
        dtype=torch.int32,
    )
    q = (
        torch.linspace(
            0.5,
            1.5,
            config.batch_size * config.vocab_size,
            dtype=torch.float32,
        )
        .reshape(config.batch_size, config.vocab_size)
        .to(device="qaic")
    )
    inv_q = q.reciprocal()
    block_size = triton.next_power_of_2(config.vocab_size)
    matrix: list[dict[str, Any]] = []
    scenarios = (
        ("all-random-all-accepted", False, 0.0),
        ("all-random-first-rejection", False, 1.0),
        ("mixed-zero-draft", True, 0.8),
    )

    if (
        draft.dtype != torch.int32
        or draft.device.type != "qaic"
        or not draft.is_contiguous()
    ):
        raise AssertionError(
            "hybrid-handoff requires contiguous QAIC torch.int32 draft IDs"
        )

    for dtype in (torch.float32, torch.float16):
        target_probs = _handoff_target_probs(
            num_tokens, config.vocab_size, draft_values, dtype
        )
        for scenario_name, mixed_requests, uniform_value in scenarios:
            if mixed_requests:
                is_greedy_values = [
                    index % 2 == 0 for index in range(config.batch_size)
                ]
            else:
                is_greedy_values = [False] * config.batch_size
            is_greedy = _qaic_tensor(is_greedy_values, dtype=torch.bool)
            uniform_probs = _qaic_tensor(
                [uniform_value] * num_tokens, dtype=torch.float32
            )
            pyt_recovered = _new_token_output(num_tokens)
            hybrid_recovered = _new_token_output(num_tokens)
            pyt_output = _new_output(config.batch_size, config.max_spec_len)
            hybrid_output = _new_output(config.batch_size, config.max_spec_len)

            def produce_recovered(output: torch.Tensor) -> None:
                _sample_recovered_tokens_kernel_pyt(
                    output,
                    cu_tokens,
                    draft,
                    None,
                    target_probs,
                    inv_q,
                    config.vocab_size,
                    BLOCK_SIZE=block_size,
                    NO_DRAFT_PROBS=True,
                    USE_FP64_GUMBEL=False,
                )

            def pyt_launch() -> None:
                produce_recovered(pyt_recovered)
                _reference_rejection_random_sample_kernel_pytorch(
                    pyt_output,
                    cu_tokens,
                    draft,
                    None,
                    target_probs,
                    bonus,
                    pyt_recovered,
                    uniform_probs,
                    is_greedy,
                    config.max_spec_len,
                    config.vocab_size,
                    None,
                    NO_DRAFT_PROBS=True,
                    SYNTHETIC_MODE=False,
                )

            def hybrid_launch() -> None:
                produce_recovered(hybrid_recovered)
                random_kernel[(config.batch_size,)](
                    hybrid_output,
                    cu_tokens,
                    draft,
                    None,
                    target_probs,
                    bonus,
                    hybrid_recovered,
                    uniform_probs,
                    is_greedy,
                    config.max_spec_len,
                    config.vocab_size,
                    None,
                    NO_DRAFT_PROBS=True,
                    SYNTHETIC_MODE=False,
                )

            def reset_pyt() -> None:
                pyt_recovered.fill_(-1)
                pyt_output.fill_(-1)

            def reset_hybrid() -> None:
                hybrid_recovered.fill_(-1)
                hybrid_output.fill_(-1)

            reset_pyt()
            pyt_launch()
            _synchronize()
            expected_output = _clone_for_validation(pyt_output)
            expected_recovered = _clone_for_validation(pyt_recovered)

            compile_first_launch_us = _measure_once(reset_hybrid, hybrid_launch)
            _validate(
                f"{_HYBRID_HANDOFF_CASE}:{scenario_name}:{dtype}",
                hybrid_output,
                expected_output,
            )
            _validate(
                f"{_HYBRID_HANDOFF_CASE}:recovered:{scenario_name}:{dtype}",
                hybrid_recovered,
                expected_recovered,
            )
            if (
                hybrid_recovered.dtype != torch.int32
                or hybrid_recovered.device.type != "qaic"
                or not hybrid_recovered.is_contiguous()
            ):
                raise AssertionError(
                    "PyTorch recovered-token output must remain contiguous QAIC "
                    "torch.int32"
                )

            _warm_up(config.warmup, reset_pyt, pyt_launch)
            _warm_up(config.warmup, reset_hybrid, hybrid_launch)
            _validate(
                f"{_HYBRID_HANDOFF_CASE}:{scenario_name}:{dtype}",
                hybrid_output,
                expected_output,
            )
            matrix.append(
                {
                    "scenario": scenario_name,
                    "probability_dtype": str(dtype),
                    "correctness": "exact",
                    "pytorch": _measure_steady_state(
                        config.iterations, reset_pyt, pyt_launch
                    ),
                    "hybrid": {
                        "compile_first_launch_us": compile_first_launch_us,
                        "steady_state": _measure_steady_state(
                            config.iterations, reset_hybrid, hybrid_launch
                        ),
                    },
                }
            )

    return {
        "kernel": _HYBRID_HANDOFF_CASE,
        "shape": {
            "batch_size": config.batch_size,
            "max_spec_len": config.max_spec_len,
            "vocab_size": config.vocab_size,
        },
        "production_contract": {
            "draft_token_ids": {
                "dtype": str(draft.dtype),
                "device": str(draft.device),
                "contiguous": draft.is_contiguous(),
            },
            "recovered_token_ids": {
                "dtype": "torch.int32",
                "device": "qaic",
                "contiguous": True,
            },
            "lengths": lengths,
        },
        "matrix": matrix,
    }


def _all_greedy_profile_case(greedy_kernel: Any, config: BenchmarkConfig) -> KernelCase:
    lengths = _handoff_lengths(config.batch_size, config.max_spec_len)
    cumulative = _cumulative(lengths)
    num_tokens = cumulative[-1]
    draft = _qaic_tensor(
        [index % config.vocab_size for index in range(num_tokens)],
        dtype=torch.int32,
    ).contiguous()
    target_argmax = _qaic_tensor(
        [
            token_id if index % 3 else (token_id + 1) % config.vocab_size
            for index, token_id in enumerate(_clone_for_validation(draft).tolist())
        ],
        dtype=torch.int32,
    )
    bonus = _qaic_tensor(
        [config.vocab_size - 1 - index for index in range(config.batch_size)],
        dtype=torch.int32,
    )
    cu_tokens = _qaic_tensor(cumulative, dtype=torch.int32)
    pyt_output = _new_output(config.batch_size, config.max_spec_len)
    hybrid_output = _new_output(config.batch_size, config.max_spec_len)

    def pyt_launch() -> None:
        _reference_rejection_greedy_sample_kernel_pytorch(
            pyt_output,
            cu_tokens,
            draft,
            target_argmax,
            bonus,
            None,
            config.max_spec_len,
            None,
            None,
            SYNTHETIC_MODE=False,
        )

    def hybrid_launch() -> None:
        greedy_kernel[(config.batch_size,)](
            hybrid_output,
            cu_tokens,
            draft,
            target_argmax,
            bonus,
            None,
            config.max_spec_len,
            None,
            None,
            SYNTHETIC_MODE=False,
        )

    def reset_pyt() -> None:
        pyt_output.fill_(-1)

    def reset_hybrid() -> None:
        hybrid_output.fill_(-1)

    reset_pyt()
    pyt_launch()
    _synchronize()
    return KernelCase(
        name="all-greedy",
        config=config,
        inputs={"lengths": lengths, "all_greedy": True},
        expected=_clone_for_validation(pyt_output),
        pyt_output=pyt_output,
        triton_output=hybrid_output,
        reset_pyt=reset_pyt,
        reset_triton=reset_hybrid,
        pyt_launch=pyt_launch,
        triton_launch=hybrid_launch,
    )


def _promotion_check(
    pytorch: dict[str, float], hybrid: dict[str, float]
) -> dict[str, Any]:
    median_improvement_percent = (
        (pytorch["median_us"] - hybrid["median_us"]) / pytorch["median_us"] * 100
    )
    return {
        "median_improvement_percent": median_improvement_percent,
        "p95_regression": hybrid["p95_us"] > pytorch["p95_us"],
        "passes": (
            median_improvement_percent >= 10 and hybrid["p95_us"] <= pytorch["p95_us"]
        ),
    }


def _run_hybrid_profiles(config: BenchmarkConfig) -> dict[str, Any]:
    """Benchmark the production hybrid objects on sampler-level profiles."""
    hybrid_kernels = get_qaic_triton_kernels()
    all_greedy = _run_case(
        _all_greedy_profile_case(hybrid_kernels.rejection_greedy_sample_kernel, config)
    )
    handoff = _run_hybrid_handoff_matrix(
        hybrid_kernels.rejection_random_sample_kernel, config
    )
    profiles = [
        {
            "profile": "all-greedy",
            "pytorch": all_greedy["pyt"],
            "hybrid": all_greedy["triton"]["steady_state"],
        }
    ]
    for result in handoff["matrix"]:
        profile = (
            "mixed-greedy-random"
            if result["scenario"] == "mixed-zero-draft"
            else "all-random"
        )
        profiles.append(
            {
                "profile": profile,
                "scenario": result["scenario"],
                "probability_dtype": result["probability_dtype"],
                "pytorch": result["pytorch"],
                "hybrid": result["hybrid"]["steady_state"],
            }
        )
    for profile in profiles:
        profile["promotion_gate"] = _promotion_check(
            profile["pytorch"], profile["hybrid"]
        )
    return {
        "kernel": _HYBRID_PROFILES_CASE,
        "production_contract": handoff["production_contract"],
        "profiles": profiles,
        "promotion_gate_passes_all_profiles": all(
            profile["promotion_gate"]["passes"] for profile in profiles
        ),
    }


def _validate(name: str, output: torch.Tensor, expected: torch.Tensor) -> None:
    actual = _clone_for_validation(output)
    if not torch.equal(actual, expected):
        raise AssertionError(
            f"{name} output mismatch:\nexpected={expected.tolist()}\nactual={actual.tolist()}"
        )


def _run_case(case: KernelCase) -> dict[str, Any]:
    triton_compile_us = _measure_once(case.reset_triton, case.triton_launch)
    _validate(case.name, case.triton_output, case.expected)
    _warm_up(case.config.warmup, case.reset_pyt, case.pyt_launch)
    _warm_up(case.config.warmup, case.reset_triton, case.triton_launch)
    _validate(case.name, case.triton_output, case.expected)
    return {
        "kernel": case.name,
        "shape": {
            "batch_size": case.config.batch_size,
            "max_spec_len": case.config.max_spec_len,
            "vocab_size": case.config.vocab_size,
        },
        "inputs": case.inputs,
        "correctness": "exact",
        "pyt": _measure_steady_state(
            case.config.iterations, case.reset_pyt, case.pyt_launch
        ),
        "triton": {
            "compile_first_launch_us": triton_compile_us,
            "steady_state": _measure_steady_state(
                case.config.iterations, case.reset_triton, case.triton_launch
            ),
        },
    }


def _environment_metadata(source_path: Path, source_hash: str) -> dict[str, Any]:
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "qaic_visible_devices": os.environ.get("QAIC_VISIBLE_DEVICES"),
        "qaic_num_cores": os.environ.get("QAIC_NUM_CORES"),
        "qaic_num_threads": os.environ.get("QAIC_NUM_THREADS"),
        "hexagon_arch_version": os.environ.get("HEXAGON_ARCH_VERSION"),
        "upstream_source": str(source_path),
        "upstream_source_sha256": source_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-spec-len", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=32_768)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument(
        "--case",
        choices=(*_KERNEL_NAMES, _HYBRID_HANDOFF_CASE, _HYBRID_PROFILES_CASE, "all"),
        default="all",
    )
    parser.add_argument(
        "--recovered-variant",
        choices=("upstream", "qaic-two-reduce"),
        default="upstream",
    )
    parser.add_argument(
        "--upstream-source", type=Path, default=_default_upstream_source()
    )
    args = parser.parse_args()

    if torch.qaic.device_count() != 1:
        raise RuntimeError(
            "Run with one visible QAIC device, for example QAIC_VISIBLE_DEVICES=0."
        )
    if args.max_spec_len > 128:
        raise ValueError("max_spec_len must not exceed upstream MAX_SPEC_LEN=128")
    if args.vocab_size < 2:
        raise ValueError("vocab_size must be at least two")

    config = BenchmarkConfig(
        batch_size=args.batch_size,
        max_spec_len=args.max_spec_len,
        vocab_size=args.vocab_size,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    kernels, source_hash = _load_upstream_kernels(args.upstream_source)
    if args.recovered_variant == "qaic-two-reduce":
        kernels[
            "sample_recovered_tokens_kernel"
        ] = _qaic_two_reduce_recovered_tokens_kernel
    factories = {
        "expand_kernel": _expand_case,
        "rejection_greedy_sample_kernel": _greedy_case,
        "rejection_random_sample_kernel": _random_case,
        "sample_recovered_tokens_kernel": _recovered_case,
    }
    selected = (
        (*_KERNEL_NAMES, _HYBRID_HANDOFF_CASE, _HYBRID_PROFILES_CASE)
        if args.case == "all"
        else (args.case,)
    )
    results = []
    for name in selected:
        if name == _HYBRID_HANDOFF_CASE:
            hybrid_kernels = get_qaic_triton_kernels()
            results.append(
                _run_hybrid_handoff_matrix(
                    hybrid_kernels.rejection_random_sample_kernel, config
                )
            )
            continue
        if name == _HYBRID_PROFILES_CASE:
            results.append(_run_hybrid_profiles(config))
            continue
        case = factories[name](kernels, config)
        if name == "sample_recovered_tokens_kernel":
            case.inputs["variant"] = args.recovered_variant
        results.append(_run_case(case))
    print(
        json.dumps(
            {
                "environment": _environment_metadata(args.upstream_source, source_hash),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
