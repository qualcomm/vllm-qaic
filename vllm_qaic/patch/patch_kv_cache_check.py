# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Skip vLLM's physical-memory KV-cache check for QAIC AoT.

In AoT mode, vLLM runs on the host CPU and the compiled QPC owns the
accelerator-side KV-cache allocation.  Therefore the ``available_memory``
value passed to vLLM is a logical cache capacity reported by the QAIC worker,
not memory that vLLM can allocate directly.  The upstream check can reject a
valid QPC configuration before the engine starts.
"""

from collections.abc import Callable

import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.platforms import current_platform

from vllm_qaic.logger import init_logger

logger = init_logger(__name__)

_PATCH_ATTR = "_qaic_aot_kv_cache_check_patch_applied"


if not getattr(kv_cache_utils._check_enough_kv_cache_memory, _PATCH_ATTR, False):
    _check_enough_kv_cache_memory = (
        kv_cache_utils._check_enough_kv_cache_memory
    )

    def qaic_check_enough_kv_cache_memory(
        available_memory: int,
        get_needed_memory: Callable[[], int],
        max_model_len: int,
        estimate_max_model_len: Callable[[int], int],
    ) -> None:
        # AoT's accelerator/QPC owns the actual KV-cache allocation.  Keep the
        # zero-memory guard: bypassing it would only turn a useful startup
        # error into a later, less actionable failure.
        if current_platform.is_aot_inference() and available_memory > 0:
            logger.info_once(
                "Skipping vLLM KV-cache memory validation for QAIC AoT; "
                "KV-cache capacity is managed by the compiled QPC."
            )
            return

        _check_enough_kv_cache_memory(
            available_memory,
            get_needed_memory,
            max_model_len,
            estimate_max_model_len,
        )

    setattr(
        qaic_check_enough_kv_cache_memory,
        _PATCH_ATTR,
        True,
    )
    kv_cache_utils._check_enough_kv_cache_memory = (
        qaic_check_enough_kv_cache_memory
    )
