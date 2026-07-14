# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import importlib
import math
from collections.abc import Iterator, Set

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
)
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.tasks import PoolingTask
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

from .qaic import QaicCausalLM

logger = init_logger(__name__)


class VisionEncoderPooler(Pooler):
    """
    A dummy pooler to provide get_supported_tasks
    """

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"embed"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        raise NotImplementedError


class QaicMultiModal(QaicCausalLM, SupportsMultiModal, SupportsMRoPE):
    """
    Subclass of QaicCausalLM that adds multimodal (vision/audio) and MRoPE
    processing support.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__(vllm_config)

        model_config = vllm_config.model_config
        override_qaic_config = (vllm_config.additional_config or {}).get(
            "override_qaic_config", {}
        )

        if self.is_pooling_model:
            self.is_vision_encoder = "skip_vision" not in override_qaic_config

            # TODO: need to update if we can support reranker
            # TODO: Add additional args for vision embedding models
            self.pooler = VisionEncoderPooler()

        self.default_mm_kwargs: dict[str, np.ndarray] | None = None
        self.video_pruning_rate: float | None = getattr(
            model_config.multimodal_config, "video_pruning_rate", None
        )  # used by iter_mm_grid_hw / iter_mm_grid_thw
        self.mm_input_info: dict[str, tuple[list[int], np.dtype]] = {}
        self.mm_output_info: dict[str, tuple[list[int], np.dtype]] = {}

        if self.uses_mrope:
            # QEfficient requires position ids to be (4, batch_size, seq_len)
            self.pad: np.ndarray = np.tile(self.pad, (4, 1))
            self.decode_batch_inputs["position_ids"] = np.tile(
                self.decode_batch_inputs["position_ids"], (4, 1, 1)
            )
        self.is_qwenvl = self.config.model_type in (
            "qwen2_5_vl",
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
        )

    def _load_multimodal(self) -> None:
        mm_input_names = [
            "pixel_values",
            "input_features",
            "vision_embeds",
            "image_position_ids",
        ]
        for input_name in mm_input_names:
            if result := self.get_io_shape_and_dtype(input_name):
                self.mm_input_info[input_name] = (result[0], result[1])
        mm_output_names = ["vision_embeds", "deepstack_features"]
        for output_name in mm_output_names:
            if result := self.get_io_shape_and_dtype(output_name, is_input=False):
                self.mm_output_info[output_name] = (result[0], result[1])

        if "image_idx" in self.session.input_names:
            self.default_mm_kwargs = {
                "image_idx": np.array([[0]], dtype=np.int64),
                "image_idx_output": np.array([[0]], dtype=np.int64),
            }
            if "mm_token_type_ids" in self.session.input_names:
                self.default_mm_kwargs["mm_token_type_ids"] = np.zeros(
                    (1, 1), dtype=np.int64
                )
            self.decode_batch_inputs.update(self.default_mm_kwargs)

        if self.config.model_type == "whisper":
            self.default_mm_kwargs = {
                k: np.empty(v[0], dtype=v[1]) for k, v in self.mm_input_info.items()
            }
            self.decode_batch_inputs.update(self.default_mm_kwargs)

    def _to_np(self, t, dtype: np.dtype | None = None) -> np.ndarray | list[np.ndarray]:
        """
        Convert a tensor, numpy array, or list thereof to numpy array(s).
        - Single tensors/arrays are converted directly.
        - Single-element lists are unwrapped before conversion.
        - Multi-element lists are converted element-wise.
        If `dtype` is specified, the result is cast to that dtype
        (in-place if possible).
        """
        if isinstance(t, list):
            if len(t) == 1:
                return self._to_np(t[0], dtype)
            return [self._to_np(item, dtype) for item in t]

        if isinstance(t, torch.Tensor):
            t = t.numpy()

        if dtype is not None:
            return t.astype(dtype, copy=False)
        return t

    def _pad_or_crop(
        self, tensor: torch.Tensor, target_dims: list | tuple, dtype: np.dtype
    ) -> np.ndarray:
        """
        Pad or crop a tensor to the specified target dimensions, returning
        a numpy array directly to avoid an extra intermediate torch tensor.

        Padding is the expected operation in practice; cropping is supported
        as a fallback to prevent server crashes in unexpected edge cases.
        """
        if any(s > t for s, t in zip(tensor.shape, target_dims, strict=False)):
            logger.warning(
                "Tensor shape %s exceeds target dimensions %s; cropping will occur. "
                "This may indicate a misconfiguration.",
                list(tensor.shape),
                list(target_dims),
            )
        padded = np.zeros(target_dims, dtype=dtype)
        slices = tuple(
            slice(0, min(s, t)) for s, t in zip(tensor.shape, target_dims, strict=False)
        )
        padded[slices] = tensor[slices]
        return padded

    def _process_vision_embeds(
        self, image_embeds: torch.Tensor, mm_kwargs: dict[str, np.ndarray]
    ) -> None:
        """
        Process image embeddings into the format expected by the model session.

        Retrieves the expected shapes and dtype for 'vision_embeds' from
        `self.mm_input_info`, where each entry is a tuple of (shape, dtype).
        The embeddings are optionally reshaped, padded, or cropped to match the
        expected shape, then cast to the expected dtype. DeepStack features, if
        extracted, are processed similarly. Populates `mm_kwargs` with
        'vision_embeds' and optionally 'deepstack_features'.
        """
        if (embed_info := self.mm_input_info.get("vision_embeds")) is None:
            logger.warning_once(
                "Vision embeddings are missing from session input names. "
                "This is unexpected and may indicate a compiler regression."
            )
            mm_kwargs["vision_embeds"] = image_embeds.numpy()
            return

        embed_shape, embed_dtype = embed_info  # unpack (list[int], dtype)
        while image_embeds.ndim < len(embed_shape):
            image_embeds = image_embeds.unsqueeze(0)

        deepstack_features, image_embeds, deepstack_dims = (
            self._maybe_extract_deepstack_features(image_embeds, embed_shape)
        )

        # Flatten embeddings if needed
        if image_embeds.shape[0] != 1 and embed_shape[0] == 1:
            image_embeds = image_embeds.reshape(
                1, math.prod(image_embeds.shape[:-1]), image_embeds.shape[-1]
            )
            if deepstack_features is not None:
                deepstack_features = deepstack_features.reshape(
                    deepstack_features.shape[0],
                    1,
                    image_embeds.shape[1],
                    image_embeds.shape[2],
                )

        # Reshape if possible, otherwise pad or crop
        if list(image_embeds.shape) != embed_shape:
            if math.prod(image_embeds.shape) == math.prod(embed_shape):
                image_embeds = image_embeds.reshape(embed_shape)
                if deepstack_features is not None:
                    deepstack_features = deepstack_features.reshape(deepstack_dims)
            else:
                image_embeds = self._pad_or_crop(image_embeds, embed_shape, embed_dtype)
                if deepstack_features is not None and deepstack_dims is not None:
                    deepstack_features = self._pad_or_crop(
                        deepstack_features, deepstack_dims, embed_dtype
                    )

        mm_kwargs["vision_embeds"] = np.asarray(image_embeds, dtype=embed_dtype)
        if deepstack_features is not None:
            mm_kwargs["deepstack_features"] = np.asarray(
                deepstack_features, dtype=embed_dtype
            )

    def _init_vision_outputs(
        self, mm_input: dict[str, np.ndarray], num_mm_inputs: int
    ) -> list[dict[str, np.ndarray]]:
        """
        Initialize output arrays for vision_embeds (and optionally
        deepstack_features and image_grid_thw).

        For Qwen2.5VL / Qwen3VL multi-resolution inputs,
        Different image sizes produce vision_embeds of different sequence lengths.
        The number of vision tokens after spatial merging is:
            grid_t * grid_h * grid_w / spatial_merge_size²
        With the default spatial_merge_size of 2 the divisor is 4, so each
        placeholder has shape (prefill_bsz, prod(grid_thw[i]) // 4, hidden_size).

        For Qwen3VL models with DeepStack, a matching deepstack_features placeholder
        of shape (num_deepstack, prefill_bsz, num_vision_tokens, hidden_size) is
        also allocated.

        image_grid_thw is replaced with empty arrays of the correct shape because
        QEfficient only reads its dimensions, not its values.

        For other cases, init output arrays based on the fixed session output size.
        """
        mm_output = []
        if "image_grid_thw" in mm_input:
            image_grid_thw = mm_input["image_grid_thw"]
            if hasattr(self.config, "hidden_size"):
                hidden_size = self.config.hidden_size
            else:
                hidden_size = self.config.text_config.hidden_size
            dtype = self.mm_output_info["vision_embeds"][1]
            spatial_merge_size = self.config.vision_config.spatial_merge_size
            deepstack_visual_indexes = getattr(
                getattr(self.config, "vision_config", None),
                "deepstack_visual_indexes",
                None,
            )
            image_grid_thw_per_image = []
            for i in range(num_mm_inputs):
                embed = np.empty(
                    (
                        self.prefill_bsz,
                        np.prod(image_grid_thw[i])
                        // spatial_merge_size
                        // spatial_merge_size,
                        hidden_size,
                    ),
                    dtype=dtype,
                )
                single_output = {"vision_embeds": embed}
                if deepstack_visual_indexes is not None:
                    num_deepstack = len(deepstack_visual_indexes)
                    single_output["deepstack_features"] = np.empty(
                        (num_deepstack,) + embed.shape, dtype=dtype
                    )
                mm_output.append(single_output)

                image_grid_thw_per_image.append(
                    np.empty(
                        np.insert(image_grid_thw[i], 0, 1), dtype=image_grid_thw.dtype
                    )
                )
            mm_input["image_grid_thw"] = image_grid_thw_per_image
        else:
            mm_output = [
                {k: np.empty(v[0], dtype=v[1]) for k, v in self.mm_output_info.items()}
                for _ in range(num_mm_inputs)
            ]
        return mm_output

    def parse_and_validate_multimodal_raw_input(
        self, **kwargs: object
    ) -> tuple[dict[str, np.ndarray], int]:
        # Assumption: for all VLMs other than Qwen2.5VL and Qwen3VL,
        # QPC is compiled with only one specialization
        if "pixel_values_flat" in kwargs:
            # InternVL: split flat tensor by num_patches per image
            image_num_patches = kwargs["image_num_patches"]
            if isinstance(image_num_patches, torch.Tensor):
                image_num_patches = image_num_patches.tolist()
            pixel_values_list = list(
                torch.split(kwargs["pixel_values_flat"], image_num_patches, dim=0)
            )
            # Ideally we should support InternVL multi-resolution
            # by creating specializations for all num_patches
            # between min_dynamic_patch and max_dynamic_patch
            # However currently it's not supported, thus padding
            target_dims, dtype = self.mm_input_info["pixel_values"]
            kwargs["pixel_values"] = [
                self._pad_or_crop(pv, target_dims, dtype)
                if list(pv.shape) != target_dims
                else pv
                for pv in pixel_values_list
            ]
            # Track the actual (unpadded) patch count per image so that
            # _pixels_to_features can trim the encoder output back to the
            # actual number of vision tokens.
            kwargs["_actual_num_patches"] = image_num_patches
        elif "pixel_values" in kwargs and "image_grid_thw" in kwargs:
            # Qwen2.5VL / Qwen3VL: split by product of grid dimensions
            assert isinstance(kwargs["image_grid_thw"], torch.Tensor)
            split_sizes = kwargs["image_grid_thw"].prod(dim=-1).tolist()
            kwargs["pixel_values"] = list(
                torch.split(kwargs["pixel_values"], split_sizes, dim=0)
            )
        elif "input_features" in kwargs:
            assert isinstance(kwargs["input_features"], torch.Tensor)
            input_features_shape = self.mm_input_info["input_features"][0]
            kwargs["input_features"] = kwargs["input_features"].reshape(
                input_features_shape
            )
        # Gemma4: vLLM processor outputs position IDs as "pixel_position_ids"
        # but the QPC binding is named "image_position_ids". Rename before filtering.
        if "pixel_position_ids" in kwargs:
            kwargs["image_position_ids"] = kwargs.pop("pixel_position_ids")
        # Determine number of multimodal inputs
        pixel_values = kwargs.get("pixel_values")
        if pixel_values is not None:
            if isinstance(pixel_values, list):
                num_mm_inputs = len(pixel_values)
            elif isinstance(pixel_values, torch.Tensor):
                pixel_values_shape = self.mm_input_info["pixel_values"][0]
                num_mm_inputs = math.prod(pixel_values.shape) // math.prod(
                    pixel_values_shape
                )
            else:
                raise ValueError(f"Unsupported pixel_values type {type(pixel_values)}")
        elif "input_features" in kwargs:
            # Audio model. Currently only whisper is supported with a single
            # audio input.
            num_mm_inputs = 1
        else:
            raise ValueError(f"Unsupported multimodal inputs {kwargs.keys()}")

        valid_input = {
            k: self._to_np(
                v,
                self.mm_input_info.get(k, (None, None))[1],  # get dtype
            )
            for k, v in kwargs.items()
            if k in self.session.binding_index_map
        }

        # Pass the actual patch counts through to _pixels_to_features using a
        # private key that is not a session binding (so it is never sent to the
        # QPC).  _pixels_to_features pops this key before calling session.np_run().
        if "_actual_num_patches" in kwargs:
            valid_input["_actual_num_patches"] = kwargs["_actual_num_patches"]

        return valid_input, num_mm_inputs

    def _combine_deepstack_features(self, output: dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine vision_embeds and deepstack_features into a single tensor (Qwen3VL).
        """
        assert "deepstack_features" in output and "vision_embeds" in output

        # embeds: (batch_size, vision_size, hidden_size)
        # deepstack_features: (depth_of_deepstack, batch_size, vision_size, hidden_size)
        embeds = output["vision_embeds"]
        deepstack_features = output["deepstack_features"]

        assert embeds.ndim == 3, f"Expected vision_embeds be 3D, got {embeds.ndim}"
        assert deepstack_features.ndim == 4, (
            f"Expected deepstack_features be 4D, got {deepstack_features.ndim}"
        )

        B, V, H = embeds.shape
        D = deepstack_features.shape[0]

        combined = np.empty((B, V, H * (1 + D)), dtype=embeds.dtype)
        combined[..., :H] = embeds
        for i in range(D):
            combined[..., (i + 1) * H : (i + 2) * H] = deepstack_features[i]

        return combined

    def _maybe_extract_deepstack_features(
        self, image_embeds: torch.Tensor, embed_dims: list
    ) -> tuple[torch.Tensor, torch.Tensor, tuple] | tuple[None, torch.Tensor, None]:
        """
        Split deepstack features from the combined embedding tensor (Qwen3VL).
        """
        vis_cfg = getattr(self.config, "vision_config", None)
        if vis_cfg is None or not hasattr(vis_cfg, "deepstack_visual_indexes"):
            return None, image_embeds, None

        if image_embeds.ndim < 2:
            raise ValueError(
                f"Expected at least 2D tensor for deepstack features, "
                f"got {image_embeds.ndim}D"
            )
        B, V = image_embeds.shape[0], image_embeds.shape[1]
        H = embed_dims[-1]
        D = len(vis_cfg.deepstack_visual_indexes)

        deepstack_features = (
            image_embeds[..., H:]
            .reshape(B, V, D, H)
            .permute(2, 0, 1, 3)  # (D, B, V, H)
        )
        deepstack_dims = (D,) + tuple(embed_dims)
        image_embeds = image_embeds[..., :H]

        return deepstack_features, image_embeds, deepstack_dims

    def _pixels_to_features(
        self,
        mm_input: dict[str, np.ndarray],
        mm_output: list[dict[str, np.ndarray]],
        num_mm_inputs: int,
    ) -> list[torch.Tensor]:
        # Pop the actual patch counts if present (set by
        # parse_and_validate_multimodal_raw_input for InternVL).
        actual_num_patches: list[int] | None = mm_input.pop("_actual_num_patches", None)
        features = []
        for i in range(num_mm_inputs):
            # Split batched inputs into per-item dictionaries
            # because vision encoder's batch size = 1
            session_input = {}
            for k, v in mm_input.items():
                if isinstance(v, list):
                    session_input[k] = v[i]
                else:
                    chunk_size = v.shape[0] // num_mm_inputs
                    session_input[k] = v[chunk_size * i : chunk_size * (i + 1)]
            session_input.update(mm_output[i])
            exec_obj_idx = self.session.np_run(session_input, is_prefill=False)
            self.session.complete_inf(exec_obj_idx, is_prefill=False)
            if len(mm_output[i]) == 1:
                feature = next(iter(mm_output[i].values()))
            elif "deepstack_features" in mm_output[i]:  # For Qwen3VL
                feature = self._combine_deepstack_features(mm_output[i])
            else:
                raise ValueError(
                    f"Unexpected vision encoder outputs: {list(mm_output[i].keys())}. "
                    "Expected either a single output or outputs containing "
                    "'deepstack_features'."
                )

            feature = torch.from_numpy(feature)

            # For models that pad pixel_values to a fixed patch count (e.g.
            # InternVL), the encoder output contains embeddings for the padded
            # patches too.
            # Trim the output to the actual number of vision tokens.
            if actual_num_patches is not None:
                padded_patches = session_input["pixel_values"].shape[0]
                actual_patches = actual_num_patches[i]
                if actual_patches < padded_patches:
                    num_image_token = feature.shape[-2] // padded_patches
                    actual_tokens = actual_patches * num_image_token
                    feature = feature[..., :actual_tokens, :]

            # sanity_check_mm_encoder_outputs requires multimodal embeddings
            # to be a list/tuple of 2D tensors, or a single 3D tensor
            if feature.ndim == 3:
                features.extend([feature[j] for j in range(feature.shape[0])])
            else:
                features.append(feature)
        return features

    def prepare_embedding_mm_kwargs(
        self, mm_embed: torch.Tensor | None
    ) -> dict[str, np.ndarray]:
        mm_kwargs: dict[str, np.ndarray] = {}
        if mm_embed is not None:
            self._process_vision_embeds(mm_embed, mm_kwargs)
        if self.default_mm_kwargs is not None:
            mm_kwargs.update(self.default_mm_kwargs)
        return mm_kwargs

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        if "image_embeds" in kwargs:
            image_embeds = kwargs["image_embeds"]
            if isinstance(image_embeds, torch.Tensor) and image_embeds.ndim == 2:
                # image_embeds is a flat (total_tokens, D) tensor that concatenates
                # all images. Split it per-image using image_grid_thw
                image_grid_thw = kwargs.get("image_grid_thw")
                if (
                    image_grid_thw is not None
                    and isinstance(image_grid_thw, torch.Tensor)
                    and image_grid_thw.ndim == 2
                ):
                    spatial_merge_size = self.config.vision_config.spatial_merge_size
                    sizes = (
                        image_grid_thw.prod(-1)
                        // spatial_merge_size
                        // spatial_merge_size
                    ).tolist()
                    image_embeds = image_embeds.split(sizes)
                else:
                    image_embeds = image_embeds.unsqueeze(0)
            return image_embeds
        if "pixel_values" in kwargs or "pixel_values_flat" in kwargs:
            mm_input, num_mm_inputs = self.parse_and_validate_multimodal_raw_input(
                **kwargs
            )
            if not mm_input:
                return []
            mm_output = self._init_vision_outputs(mm_input, num_mm_inputs)
            return self._pixels_to_features(mm_input, mm_output, num_mm_inputs)
        return []

    # Registry mapping model_type strings to their MRoPE source class.
    # To add support for a new MRoPE model, add an entry here.
    _MROPE_CLASS_REGISTRY: dict[str, tuple[str, str]] = {
        "qwen2_5_vl": (
            "vllm.model_executor.models.qwen2_5_vl",
            "Qwen2_5_VLForConditionalGeneration",
        ),
        "qwen3_vl": (
            "vllm.model_executor.models.qwen3_vl",
            "Qwen3VLForConditionalGeneration",
        ),
        "qwen3_vl_moe": (
            "vllm.model_executor.models.qwen3_vl",
            "Qwen3VLForConditionalGeneration",
        ),
        "qwen3_5": (
            "vllm.model_executor.models.qwen3_5",
            "Qwen3_5ForConditionalGeneration",
        ),
        "qwen3_5_moe": (
            "vllm.model_executor.models.qwen3_5",
            "Qwen3_5MoeForConditionalGeneration",
        ),
    }

    def _get_mrope_source_class(self):
        model_type = self.config.model_type
        entry = self._MROPE_CLASS_REGISTRY.get(model_type)
        if entry is None:
            supported = list(self._MROPE_CLASS_REGISTRY)
            raise ValueError(
                f"Unsupported model type for MRoPE: {model_type!r}. "
                f"Supported types: {supported}. "
                "To add support, register the model in "
                "QaicMultiModal._MROPE_CLASS_REGISTRY."
            )
        module_path, class_name = entry
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        source_class = self._get_mrope_source_class()
        return source_class.get_mrope_input_positions(self, input_tokens, mm_features)

    def iter_mm_grid_hw(
        self, input_tokens: list[int], mm_features: list[MultiModalFeatureSpec]
    ) -> Iterator[tuple[int, int, int]]:
        # Helper for Qwen3-VL's get_mrope_input_positions.
        source_class = self._get_mrope_source_class()
        return source_class.iter_mm_grid_hw(self, input_tokens, mm_features)

    def iter_mm_grid_thw(
        self, mm_features: list[MultiModalFeatureSpec]
    ) -> Iterator[tuple[int, int, int, int, float]]:
        # Helper for Qwen2.5-VL's get_mrope_input_positions.
        source_class = self._get_mrope_source_class()
        return source_class.iter_mm_grid_thw(self, mm_features)
