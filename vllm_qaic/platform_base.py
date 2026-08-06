# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import ast
import importlib
import importlib.util
import json
import os
from typing import TYPE_CHECKING
import functools

import torch

from vllm.logger import init_logger
from vllm.platforms import Platform, PlatformEnum
from vllm.utils.import_utils import PlaceholderModule
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .utils import (
    QAIC_QUANTIZATION_LIST,
)

try:
    import torch_qaic
    import torch_qaic.qaic as qaic
except ImportError:
    qaic = PlaceholderModule("qaic")

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
else:
    ModelConfig = None
    VllmConfig = None

logger = init_logger(__name__)

DYNAMIC_RESOLUTION_MODELS = [
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_5",
    "qwen3_5_moe",
]


class QaicPlatform(Platform):
    _enum = PlatformEnum.OOT
    primary_attn_backend_cls = (
        "vllm_qaic.attention.backends"
        ".qaic_attn.QAicTorchAttentionBackend"
    )
    device_name: str = "qaic"
    # Set device type to cpu if it's AOT.
    # This is a workaround for online serving's AsyncEngineArgs.
    # The device type will be restored to qaic during pre_register_and_update.
    device_type: str = "cpu" if isinstance(qaic, PlaceholderModule) else "qaic"
    device_control_env_var: str = "QAIC_VISIBLE_DEVICES"
    supported_quantization: list[str] = QAIC_QUANTIZATION_LIST
    # for eager mode
    simple_compile_backend: str = "eager"

    _torch_qaic_installed: bool = importlib.util.find_spec("torch_qaic") is not None
    is_aot = not _torch_qaic_installed
    dispatch_key: str = "PrivateUse1"
    dist_backend: str = "qccl"
    worker_cls_name = "QaicWorkerAoT" if is_aot else "QaicWorkerPyt"
    worker_cls_pkg_root = "vllm_qaic.worker.worker"
    worker_cls = worker_cls_pkg_root + "." + worker_cls_name
    device_communicator_cls = "vllm_qaic.distributed.communicator.QAicCommunicator"

    @classmethod
    def import_kernels(cls) -> None:
        # QAIC has no CUDA kernels. Skip all kernel imports — importing vllm._C
        # on non-CUDA hardware triggers a C++ bad_alloc/terminate (SIGABRT), which
        # Python's except ImportError cannot catch. vllm._moe_C is also skipped
        # since QAIC uses its own on-chip operators, not CUDA MoE kernels.
        # TODO: import Hexagon/QAIC-specific .so here when available.
        pass

    @classmethod
    def get_supported_quantization(cls):
        return cls.supported_quantization

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.float32]

    @classmethod
    def is_aot_inference(cls) -> bool:
        return cls.is_aot

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "qaic"

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        # Not Implemented for aot
        if isinstance(qaic, PlaceholderModule):
            raise NotImplementedError
        # for eager mode single SoC
        return qaic.get_device_info(device_id).pci_bus_id

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        # Not Implemented for aot
        if isinstance(qaic, PlaceholderModule):
            raise NotImplementedError
        # for eager mode single SoC
        return qaic.get_device_info(device_id).total_ddr_size_kb * 1024

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return cls.device_communicator_cls

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        return False

    @classmethod
    @functools.cache
    def get_num_cores(cls, device_id: int = 0) -> int:
        if not cls.is_aot:
            return torch_qaic.qaic.get_device_info(device_id).num_cores
        else:
            pass

    @classmethod
    @functools.cache
    def get_num_hvx_threads(cls, device_id: int = 0) -> int:
        if not cls.is_aot:
            return torch_qaic.qaic.get_device_info(device_id).per_core_hvx_thread_count
        else:
            pass

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype):
        # for eager mode
        return dtype in [torch.float16, torch.float32]

    @classmethod
    def inference_mode(cls):
        return torch.inference_mode()

    @classmethod
    def set_device(cls, device: torch.device):
        # Not Implemented for aot
        if isinstance(qaic, PlaceholderModule):
            raise NotImplementedError
        # for eager mode
        qaic.set_device(device)

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        # for eager mode
        free, total = qaic.mem_get_info(device)
        return total - free

    @classmethod
    def get_worker_cls(cls):
        return cls.worker_cls

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        additional_config = vllm_config.additional_config
        device_config = vllm_config.device_config
        if additional_config:
            cfg = additional_config
            if not isinstance(cfg, (dict, str)):
                raise TypeError(
                    f"additional_config must be a dict or JSON/Python literal string, "
                    f"got {type(cfg).__name__!r}"
                )
            if isinstance(cfg, str):
                try:
                    cfg = json.loads(cfg)
                except json.JSONDecodeError:
                    try:
                        cfg = ast.literal_eval(cfg)
                    except (ValueError, SyntaxError) as e:
                        raise ValueError(
                            f"Failed to parse additional_config string "
                            f"{additional_config!r}. "
                            "Expected a valid JSON object or Python literal dict."
                        ) from e
                if not isinstance(cfg, dict):
                    raise TypeError(
                        f"additional_config must parse to a dict, "
                        f"got {type(cfg).__name__!r}"
                    )
                vllm_config.additional_config = cfg

        if cls.is_aot:
            if vllm_config.model_config.enforce_eager:
                logger.warning_once(
                    "setting inference mode to Ahead-of-Time since"
                    "torch_qaic is not installed"
                )
                vllm_config.model_config.enforce_eager = False
            device_config.device = torch.device("cpu")
        else:
            if not vllm_config.model_config.enforce_eager:
                logger.warning_once(
                    "setting inference mode to Eager mode since torch_qaic is installed"
                )
                vllm_config.model_config.enforce_eager = True
            device_config.device = torch.device("qaic")
            # set QAIC_VISIBLE_DEVICES from device_group
            # if not already set
            if (
                os.environ.get(cls.device_control_env_var) is None
                and "device_group" in additional_config
            ):
                logger.warning_once(
                    "setting inference mode to Eager mode since torch_qaic is installed"
                )
                os.environ[cls.device_control_env_var] = additional_config[
                    "device_group"
                ]

        # Shorthand used throughout this method
        override_qaic_config = additional_config.get("override_qaic_config", {})
        # Normalise types in override_qaic_config (e.g. strings from CLI parser)
        # before any downstream code reads from it, mirroring what
        # _get_qaic_compile_config previously did via its local _clean_config.
        if override_qaic_config:
            from vllm_qaic.utils.qaic_utils import _clean_config

            cleaned = _clean_config(override_qaic_config, vllm_config)
            override_qaic_config.update(cleaned)
            if additional_config and "override_qaic_config" in additional_config:
                additional_config["override_qaic_config"].update(cleaned)
        else:
            override_qaic_config = {}
            additional_config["override_qaic_config"] = override_qaic_config
        # a hack for online serving's Async EngineArgs
        if vllm_config.device_config.device_type != cls.device_type:
            vllm_config.device_config.device_type = cls.device_type

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = cls.get_worker_cls()

        if parallel_config.world_size > 1:
            parallel_config.distributed_executor_backend = "uni"
            if vllm_config.model_config.enforce_eager:
                # uni backend sets qualnet ip while mp backend sets localhost ip
                parallel_config.distributed_executor_backend = "mp"

        model_config = vllm_config.model_config
        scheduler_config = vllm_config.scheduler_config
        cache_config = vllm_config.cache_config

        assert not (vllm_config.lora_config and vllm_config.speculative_config), (
            "LORA with SPD is not yet supported for QAIC backend"
        )
        assert not (vllm_config.lora_config and model_config.is_multimodal_model), (
            "LORA with Multi-modality is not yet supported for QAIC backend"
        )
        assert not (
            vllm_config.speculative_config and model_config.is_multimodal_model
        ), "SPD with Multi-modality is not yet supported for QAIC backend"

        if (
            cls.is_aot
            and model_config.runner_type == "pooling"
            and not model_config.is_multimodal_model
        ):
            # QAIC pooling QPCs are compiled with seq_len == max_model_len.
            # Set both explicitly here (AOT and eager) so prefill_seq_len is correct.
            scheduler_config.enable_chunked_prefill = False
            scheduler_config.long_prefill_token_threshold = model_config.max_model_len

        if not cls.is_aot:
            assert vllm_config.kv_transfer_config is None, (
                "QAIC eager mode does not support disaggregated serving"
            )
            if vllm_config.speculative_config:
                raise ValueError(
                    "Speculative decoding (SpD) is not supported in eager mode on QAIC. "
                    "SpD requires AOT (non-eager) compilation."
                )
            if scheduler_config.async_scheduling:
                logger.warning_once(
                    "QAIC eager mode does not support async scheduling; "
                    "Falling back to non-async scheduling."
                )
                scheduler_config.async_scheduling = False

        cache_config = vllm_config.cache_config
        if cache_config:
            if model_config.enforce_eager:
                cache_config.block_size = 16
            else:
                if cache_config.enable_prefix_caching:
                    cache_config.enable_prefix_caching = False
                    cache_config.mamba_block_size = (
                        model_config.max_model_len
                    )  # reset to "no-op" default
                    cache_config.mamba_cache_mode = "none"  # reset to disabled
                    logger.warning_once(
                        "Prefix caching is not yet supported on v1 Engine. "
                        "Will automatically disable it."
                    )
                cache_config.block_size = model_config.max_model_len  # ctx_len

        if cls.is_aot:
            if model_config.hf_config.model_type == "whisper":
                # Whisper is an encoder-decoder model: vLLM disables chunked prefill
                # and sets long_prefill_token_threshold to 0, so the formula above
                # would give 0. Use max_source_positions (the encoder input length)
                # as the budget instead, matching the pattern in qaic_whisper.py.
                scheduler_config.max_num_batched_tokens = getattr(
                    model_config.hf_config, "max_source_positions", 1500
                )
            else:
                __prefill_seq_len = override_qaic_config.get("prefill_seq_len", 0)
                if not __prefill_seq_len:
                    if scheduler_config.long_prefill_token_threshold == 0:
                        __prefill_seq_len = min(128, model_config.max_model_len)
                    else:
                        __prefill_seq_len = min(
                            scheduler_config.long_prefill_token_threshold,
                            model_config.max_model_len,
                        )
                if not scheduler_config.enable_chunked_prefill:
                    # TODO: long_prefill_token_threshold should not be set when
                    # chunked prefill is disabled
                    logger.warning_once(
                        "Chunked prefill is disabled; chunk size=%d will be used"
                        " as prefill_seq_len.",
                        __prefill_seq_len,
                    )
                if "override_qaic_config" not in additional_config:
                    additional_config["override_qaic_config"] = {}
                additional_config["override_qaic_config"].update(
                    {"prefill_seq_len": __prefill_seq_len}
                )
                # max_num_batched_tokens.  QAIC's _prepare_qaic_inputs expands each
                # decode request's token count from 1 → (1 + num_spec_tokens), but the
                # scheduler counts decode requests as 1 token each when enforcing the
                # budget.  Setting max_num_batched_tokens to this value simultaneously
                # gives the scheduler the correct per-step budget AND sizes the buffers
                # large enough to never overflow after decode expansion.
                scheduler_config.max_num_batched_tokens = min(
                    scheduler_config.max_num_seqs * __prefill_seq_len,
                    scheduler_config.max_num_batched_tokens,
                )

        if cls.is_aot:
            # Intel OpenMP tuning — only activate when user has already preloaded
            # libiomp5.so (Intel OpenMP runtime).
            # Set VLLM_DISABLE_LD_PRELOAD_OPT=1 to skip this optimization.
            ld_preload_str = os.getenv("LD_PRELOAD", "")
            disable_opt = os.getenv("VLLM_DISABLE_LD_PRELOAD_OPT", "0") == "1"
            if not disable_opt and "libiomp5.so" in ld_preload_str:
                import platform

                cpu_info = platform.processor() or "unknown"
                print(
                    f"Updating KMP_BLOCKTIME, KMP_TPAUSE, "
                    f"KMP_FORKJOIN_BARRIER_PATTERN for {cpu_info}"
                )
                os.environ["KMP_BLOCKTIME"] = "1"
                os.environ["KMP_TPAUSE"] = "0"
                os.environ["KMP_FORKJOIN_BARRIER_PATTERN"] = "dist,dist"
                os.environ["KMP_PLAIN_BARRIER_PATTERN"] = "dist,dist"
                os.environ["KMP_REDUCTION_BARRIER_PATTERN"] = "dist,dist"

        from vllm.config import CompilationMode

        compilation_config = vllm_config.compilation_config
        # QAIC_FIXME: default uses torch.compile based custom vllm-backend.
        # Turning off all torch.compile modes to fallback to purely eager for now.
        if compilation_config and compilation_config.mode != CompilationMode.NONE:
            mode = "eager mode" if vllm_config.model_config.enforce_eager else "AOT"
            logger.warning_once(
                "vllm qaic platform doesn't support compilation mode = %s,"
                "disabling and running with %s...",
                compilation_config.mode,
                mode,
            )
            compilation_config.mode = CompilationMode.NONE
        on_device_sampling_en = override_qaic_config.get("aic_include_sampler", False)
        if isinstance(on_device_sampling_en, str):
            on_device_sampling_en = on_device_sampling_en.lower() in [
                "true",
                "1",
            ]

        assert not (on_device_sampling_en and vllm_config.speculative_config), (
            "SPD with On-device sampling is not yet supported for QAIC backend"
        )

        # Disaggregated prefill/decode is supported standalone for now
        if vllm_config.kv_transfer_config:
            assert not vllm_config.lora_config, (
                "LORA with Disaggregated serving not yet supported for QAIC backend"
            )
            assert (
                not vllm_config.speculative_config
                or vllm_config.speculative_config.method in ["ngram", "draft_model"]
            ), (
                "PLD and DLM based SPD Types are supported with Disaggregated "
                "serving, other SPD types such as Turbo is not yet supported "
                "with Disaggregated serving for QAIC backend"
            )
            assert not (
                vllm_config.kv_transfer_config.kv_role != "kv_producer"
                and vllm_config.cache_config.enable_prefix_caching
            ), (
                "Prefix caching with KV-role 'kv_consumer' or 'kv_both' not "
                "yet supported for QAIC backend"
            )
            if (
                on_device_sampling_en
                and vllm_config.kv_transfer_config.kv_role == "kv_producer"
            ):
                logger.warning_once(
                    "On-device sampling with Disaggregated serving is only "
                    "supported in Decode cluster with no support for "
                    "repetition penalty"
                )

            if vllm_config.kv_transfer_config.kv_role == "kv_producer":
                # Monkey patch uniproc_executor to QaicUniProcExecutor
                import vllm.v1.executor.uniproc_executor as uniproc_executor

                from vllm_qaic.qaic_uniproc_executor import QaicUniProcExecutor

                uniproc_executor.UniProcExecutor = QaicUniProcExecutor
                stages = int(override_qaic_config.get("stages"))
                assert (
                    stages is None
                    or int(stages) <= 1
                    or vllm_config.scheduler_config.max_num_seqs <= int(stages)
                ), (
                    f"max_num_seqs ({vllm_config.scheduler_config.max_num_seqs})"
                    f" must be less than or equal to prefill_pipeline_parallel_size"
                    f" ({stages}) when prefill streaming is enabled"
                )

        if cls.is_aot:
            model_type = model_config.hf_config.model_type
            if model_config.is_multimodal_model and model_type != "whisper":
                cls._configure_multimodal_model(
                    vllm_config, model_config, scheduler_config, model_type
                )

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.warning("Pin memory is not supported on Qaic.")
        return False

    def __getattr__(self, key: str):
        if key == "Event":
            return True
        device = getattr(torch, self.device_name, None)
        if device is not None and hasattr(device, key):
            return getattr(device, key)
        else:
            logger.warning(
                "Current platform %s does not have '%s' attribute.",
                self.device_name,
                key,
            )
            return None

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: AttentionBackendEnum,
        attn_selector_config,
    ) -> str:
        # for eager mode
        if attn_selector_config.use_mla:
            raise NotImplementedError("QAic doesn't support MLA attention")
        if attn_selector_config.use_sparse:
            raise NotImplementedError("QAic doesn't support sparse attention")

        return cls.primary_attn_backend_cls

    @classmethod
    def _configure_multimodal_model(
        cls,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        scheduler_config,
        model_type: str,
    ) -> None:
        additional_config = vllm_config.additional_config
        override_qaic_config = (additional_config or {}).get(
            "override_qaic_config"
        ) or {}

        # The QAIC backend does not support the multimodal processor cache.
        # Disable it unconditionally so input processing uses the
        # request-id/modality/index UUID path (see V1 input_processor).
        if model_config.multimodal_config is not None:
            model_config.multimodal_config.mm_processor_cache_gb = 0

        is_vision_encoder = (
            model_config.runner_type == "pooling"
            and "skip_vision" not in override_qaic_config
        )
        if is_vision_encoder and cls.is_aot:
            if scheduler_config.async_scheduling:
                logger.warning(
                    "QAIC currently does not support async scheduling "
                    "for vision encoder; Falling back to non-async scheduling."
                )
                scheduler_config.async_scheduling = False
            scheduler_config.enable_chunked_prefill = False
            scheduler_config.long_prefill_token_threshold = model_config.max_model_len
            # max_num_batched_tokens was set earlier as
            # max_num_seqs * long_prefill_token_threshold (with the original
            # seq_len value). Now that we have raised long_prefill_token_threshold
            # to max_model_len for the encoder runner, recompute the budget so
            # the scheduler allows prompts longer than the original seq_len.
            scheduler_config.max_num_batched_tokens = (
                scheduler_config.max_num_seqs
                * scheduler_config.long_prefill_token_threshold
            )
            if "override_qaic_config" not in vllm_config.additional_config:
                additional_config["override_qaic_config"] = {}
            additional_config["override_qaic_config"].update(
                {"prefill_seq_len": model_config.max_model_len}
            )

        if model_type in DYNAMIC_RESOLUTION_MODELS:
            if vllm_config.additional_config is None:
                vllm_config.additional_config = {}
            override_qaic_config = vllm_config.additional_config.setdefault(
                "override_qaic_config", {}
            )
            cls._apply_dynamic_resolution_config(model_config, override_qaic_config)

        from vllm_qaic.model_loader.qaic_custom_mm_processor import (
            register_qaic_custom_mm_processor,
        )

        register_qaic_custom_mm_processor(model_type)

    @classmethod
    def _apply_dynamic_resolution_config(
        cls, model_config: ModelConfig, override_qaic_config: dict
    ) -> None:
        """
        Configure multimodal processor settings for models with
        dynamic resolution support. Some vision-language models
        (e.g. Qwen2.5VL, Qwen3VL) can handle dynamic image resolutions by mapping them to
        a variable number of visual tokens. On QAIC hardware, the vision encoder requires
        fixed-size inputs, so this method registers a set of supported ``(height, width)``
        resolutions that a custom processor will snap images to at runtime.

        Currently only Qwen2.5VL and Qwen3VL are supported.
        """
        # For Qwen2.5VL/Qwen3VL on QAIC, min_pixels and max_pixels must match QEfficient
        # values and cannot be overridden per request.
        if model_config.mm_processor_kwargs is None:
            model_config.mm_processor_kwargs = {}
        mm_kwargs = model_config.mm_processor_kwargs
        mm_kwargs.setdefault("max_pixels", 1280 * 28 * 28)
        mm_kwargs.setdefault("min_pixels", 4 * 28 * 28)
        override_qaic_config["mm_processor_kwargs"] = {
            k: mm_kwargs[k] for k in ("max_pixels", "min_pixels")
        }

        if "height" in override_qaic_config or "width" in override_qaic_config:
            assert (
                "height" in override_qaic_config and "width" in override_qaic_config
            ), (
                "Both 'height' and 'width' must be provided together in "
                "'override_qaic_config'."
            )
            heights = override_qaic_config["height"]
            widths = override_qaic_config["width"]
            if isinstance(heights, int):
                heights = [heights]
            if isinstance(widths, int):
                widths = [widths]
            assert len(heights) == len(widths), (
                f"The number of provided heights ({len(heights)}) and "
                f"widths ({len(widths)}) must match."
            )
            override_qaic_config["height"] = heights
            override_qaic_config["width"] = widths

            from vllm_qaic.model_loader.qaic_custom_mm_processor import (
                _qwenvl_processor_config,
            )

            _qwenvl_processor_config[id(model_config)] = {
                "height": heights,
                "width": widths,
            }
