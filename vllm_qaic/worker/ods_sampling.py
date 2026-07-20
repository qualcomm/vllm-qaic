"""On-device sampling array construction helpers.

This module converts vLLM per-request sampling metadata into per-slot NumPy
arrays consumed by QAIC on-device sampling (ODS) bindings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from vllm.v1.sample.metadata import SamplingMetadata


@dataclass(frozen=True)
class ODSSamplingArrays:
    """Per-slot ODS sampling-control arrays.

    Notes:
    - Greedy requests are represented by vLLM with ``temperature == 0``. ODS
      preserves ``temperature == 0`` unchanged. The on-device sampler uses its
      dedicated greedy selection path for those slots (argmax via
      ``torch.where(temperatures == 0, greedy_samples, random_samples)``),
      independent of ``top_ks``, ``top_ps``, ``min_ps``, and random draws.
    - Seeded random draws are reproducible for a given generator seed on the
      ODS path, but are not guaranteed to match host-side vLLM sampler outputs
      bit-for-bit for the same seed.
    """

    temperatures: np.ndarray
    top_ks: np.ndarray
    top_ps: np.ndarray
    min_ps: np.ndarray
    repetition_penalties: np.ndarray
    presence_penalties: np.ndarray
    random_numbers: np.ndarray


def build_ods_sampling_arrays(
    sampling_metadata: SamplingMetadata,
    min_p_by_slot: dict[int, float],
    max_top_k_ids: int,
    random_number_count: int,
    temperature_fallback: np.ndarray | None = None,
    top_k_fallback: np.ndarray | None = None,
    top_p_fallback: np.ndarray | None = None,
) -> ODSSamplingArrays:
    """Build per-slot ODS sampling-control arrays from ``SamplingMetadata``.

    ``SamplingMetadata`` in this vLLM version does not carry ``min_p`` as a
    tensor field. ``min_p`` comes from per-request ``SamplingParams`` and is
    therefore passed in separately via ``min_p_by_slot``.

    This function only performs array construction and defensive normalization.
    Request-level validation (for example, rejecting top-k above deployment
    limits) is enforced earlier by guardrails; clamping here is a defensive
    fallback for runtime safety.

    vLLM batch-level fast-path optimization may set ``temperature`` to ``None``
    for all-greedy batches, ``top_k`` to ``None`` for no-top-k batches, and
    ``top_p`` to ``None`` for no-top-p batches. Callers may provide per-slot
    fallback arrays derived from each request's ``SamplingParams`` via
    ``temperature_fallback``, ``top_k_fallback``, and ``top_p_fallback``.
    """

    if sampling_metadata.temperature is None and temperature_fallback is None:
        raise ValueError(
            "sampling_metadata.temperature is None and no temperature_fallback "
            "was provided"
        )
    if sampling_metadata.top_k is None and top_k_fallback is None:
        raise ValueError(
            "sampling_metadata.top_k is None and no top_k_fallback was "
            "provided"
        )
    if sampling_metadata.top_p is None and top_p_fallback is None:
        raise ValueError(
            "sampling_metadata.top_p is None and no top_p_fallback was "
            "provided"
        )
    if max_top_k_ids <= 0:
        raise ValueError("max_top_k_ids must be positive")
    if random_number_count <= 0:
        raise ValueError("random_number_count must be positive")

    if sampling_metadata.temperature is None:
        assert temperature_fallback is not None
        num_slots = int(temperature_fallback.shape[0])
    else:
        num_slots = int(sampling_metadata.temperature.shape[0])

    if sampling_metadata.top_k is None:
        assert top_k_fallback is not None
        top_k_slots = int(top_k_fallback.shape[0])
    else:
        top_k_slots = int(sampling_metadata.top_k.shape[0])

    if sampling_metadata.top_p is None:
        assert top_p_fallback is not None
        top_p_slots = int(top_p_fallback.shape[0])
    else:
        top_p_slots = int(sampling_metadata.top_p.shape[0])

    if top_k_slots != num_slots or top_p_slots != num_slots:
        raise ValueError(
            "ODS sampling control slot-count mismatch: "
            f"temperature implies {num_slots} slots, "
            f"top_k implies {top_k_slots} slots, "
            f"top_p implies {top_p_slots} slots. "
            "All must agree."
        )

    missing_min_p_slots = [slot for slot in range(num_slots) if slot not in min_p_by_slot]
    if missing_min_p_slots:
        raise ValueError(
            "min_p_by_slot is missing required slots: "
            f"{missing_min_p_slots}"
        )

    if sampling_metadata.temperature is None:
        assert temperature_fallback is not None
        temperatures = np.asarray(temperature_fallback, dtype=np.float32)
    else:
        temperatures = (
            sampling_metadata.temperature.detach().cpu().numpy().astype(np.float32, copy=True)
        )
    if sampling_metadata.top_k is None:
        assert top_k_fallback is not None
        top_ks = np.asarray(top_k_fallback, dtype=np.int32)
    else:
        top_ks = sampling_metadata.top_k.detach().cpu().numpy().astype(np.int32, copy=True)
    if sampling_metadata.top_p is None:
        assert top_p_fallback is not None
        top_ps = np.asarray(top_p_fallback, dtype=np.float32)
    else:
        top_ps = sampling_metadata.top_p.detach().cpu().numpy().astype(np.float32, copy=True)
    if sampling_metadata.no_penalties:
        repetition_penalties = np.full((num_slots,), 1.0, dtype=np.float32)
        presence_penalties = np.full((num_slots,), 0.0, dtype=np.float32)
    else:
        repetition_penalties = (
            sampling_metadata.repetition_penalties.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=True)
        )
        presence_penalties = (
            sampling_metadata.presence_penalties.detach().cpu().numpy().astype(np.float32, copy=True)
        )
    min_ps = np.asarray(
        [float(min_p_by_slot[slot]) for slot in range(num_slots)],
        dtype=np.float32,
    )

    # vLLM uses top_k <= 0 (default sentinel is 0) to mean "disabled". For ODS,
    # map disabled to the deployment's widest allowed device range.
    top_ks = np.where(top_ks <= 0, max_top_k_ids, top_ks)

    # Defensive runtime clamp: request-level over-limit rejection is handled
    # earlier by guardrails, but keep runtime arrays in-bounds regardless.
    top_ks = np.minimum(top_ks, max_top_k_ids).astype(np.int32, copy=False)

    random_numbers = np.empty((num_slots, random_number_count), dtype=np.float32)
    for slot_index in range(num_slots):
        generator = sampling_metadata.generators.get(slot_index)
        if generator is None:
            random_values = torch.rand(
                random_number_count,
                dtype=torch.float32,
            )
        else:
            random_values = torch.rand(
                random_number_count,
                generator=generator,
                dtype=torch.float32,
            )
        random_numbers[slot_index] = (
            random_values.detach().cpu().numpy().astype(np.float32, copy=False)
        )

    return ODSSamplingArrays(
        temperatures=temperatures,
        top_ks=top_ks,
        top_ps=top_ps,
        min_ps=min_ps,
        repetition_penalties=repetition_penalties,
        presence_penalties=presence_penalties,
        random_numbers=random_numbers,
    )


def detect_nondefault_frequency_penalty(
    sampling_metadata: SamplingMetadata,
) -> list[int]:
    """Return slot indices with non-default (non-zero) frequency penalty."""

    penalties = sampling_metadata.frequency_penalties
    nondefault_slots = torch.nonzero(penalties != 0.0, as_tuple=False).flatten()
    return [int(slot) for slot in nondefault_slots.tolist()]
