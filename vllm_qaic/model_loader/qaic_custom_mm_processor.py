# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from packaging.version import Version as _Version
from qwen_vl_utils import smart_resize
from transformers import BatchFeature, Qwen2VLImageProcessorFast, TensorType
from transformers import __version__ as _transformers_version
from transformers.image_processing_utils import select_best_resolution
from transformers.image_transforms import group_images_by_shape, reorder_images
from transformers.image_utils import SizeDict
from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
from transformers.models.qwen3_vl import Qwen3VLProcessor
from transformers.processing_utils import ProcessorMixin
from transformers.utils.import_utils import (
    is_torchvision_available,
    is_torchvision_v2_available,
)
from vllm.logger import init_logger
from vllm.model_executor.models.gemma3_mm import (
    Gemma3DummyInputsBuilder,
    Gemma3ForConditionalGeneration,
    Gemma3MultiModalProcessor,
    Gemma3ProcessingInfo,
)
from vllm.model_executor.models.gemma4_mm import (
    Gemma4DummyInputsBuilder,
    Gemma4ForConditionalGeneration,
    Gemma4MultiModalProcessor,
    Gemma4ProcessingInfo,
)
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo,
)
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VLMultiModalDataParser,
    _create_qwen2vl_field_factory,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeProcessingInfo,
    Qwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.qwen3_vl_moe import (
    Qwen3VLMoeForConditionalGeneration,
    Qwen3VLMoeProcessingInfo,
)
from vllm.model_executor.models.transformers import TransformersMultiModalForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    ImageItem,
    ModalityData,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ImageEmbeddingItems,
    ImageProcessorItems,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import InputProcessingContext
from vllm.multimodal.processing.processor import (
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

logger = init_logger(__name__)
# transformers v5.5.4 replaced image_processor.min_pixels / .max_pixels class
# attributes with a size dict keyed by 'shortest_edge' / 'longest_edge'.
_TRANSFORMERS_NEW_IMAGE_PROCESSOR = _Version(_transformers_version) == _Version("5.5.4")


class QaicGemma3MultiModalProcessor(Gemma3MultiModalProcessor):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processed_outputs = super(Gemma3MultiModalProcessor, self)._call_hf_processor(
            prompt,
            mm_data,
            mm_kwargs,
            tok_kwargs,
        )

        processor = self.info.get_hf_processor(**mm_kwargs)
        images_kwargs = self.info._resolve_image_kwargs(processor, {"do_pan_and_scan"})
        do_pan_and_scan = images_kwargs["do_pan_and_scan"]
        if do_pan_and_scan:
            raise ValueError("QAIC does not support Gemma3 with pan-and-scan enabled.")

        if (images := mm_data.get("images")) is not None:
            parsed_images = (
                self._get_data_parser()
                .parse_mm_data({"image": images})
                .get_items("image", (ImageEmbeddingItems, ImageProcessorItems))
            )
            num_patches = [1] * len(parsed_images)
            processed_outputs["num_patches"] = torch.tensor(num_patches)

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_patches = hf_inputs.get("num_patches", torch.empty(0))

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", num_patches),
            num_patches=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_token = processor.boi_token

        def get_replacement_gemma3(item_idx: int):
            boi_token = processor.boi_token
            image_text = boi_token
            repl_full = image_text.replace(boi_token, processor.full_image_sequence)

            tokenizer = processor.tokenizer
            vocab = tokenizer.get_vocab()
            image_token_id = vocab[tokenizer.image_token]

            return PromptUpdateDetails.select_token_id(repl_full, image_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=get_replacement_gemma3,
            )
        ]


class QaicGemma4MultiModalProcessor(Gemma4MultiModalProcessor):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # Get all base fields (pixel_values, pixel_position_ids, audio, video)
        # from the open-source Gemma4MultiModalProcessor, then add image_embeds
        # which is only needed on the QAIC decode-instance path where
        # pre-computed embeddings are passed in instead of raw pixel values.
        fields = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        fields["image_embeds"] = MultiModalFieldConfig.batched("image")
        return fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        prompt_updates = []

        if "image" in mm_items:
            # Target image_token (<|image|>) — the single placeholder the
            # Gemma4 chat template inserts once per image in the prompt.
            image_token = hf_processor.image_token

            def get_replacement_image(item_idx: int):
                images = mm_items.get_items(
                    "image", (ImageEmbeddingItems, ImageProcessorItems)
                )
                if isinstance(images, ImageEmbeddingItems):
                    # QAIC decode-instance path: images arrive as pre-computed
                    # embeddings. Derive the placeholder token count directly
                    # from the embedding feature size instead of image dimensions,
                    # since get_image_size() is not available on embeddings.
                    num_soft = images.get_feature_size(item_idx)
                    config = self.info.get_hf_config()
                    token_ids = (
                        [config.boi_token_id]
                        + [hf_processor.image_token_id] * num_soft
                        + [config.eoi_token_id]
                    )
                    return PromptUpdateDetails.select_token_id(
                        token_ids, hf_processor.image_token_id
                    )
                # Standard path: delegate to the base class image replacement
                # logic which reads image dimensions from ImageProcessorItems.
                base_updates = super(
                    QaicGemma4MultiModalProcessor, self
                )._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs)
                image_updates = [u for u in base_updates if u.modality == "image"]
                return image_updates[0].replacement(item_idx)

            prompt_updates.append(
                PromptReplacement(
                    modality="image",
                    target=image_token,
                    replacement=get_replacement_image,
                )
            )

        # Video and audio updates are unchanged — delegate entirely to base class.
        base_updates = super()._get_prompt_updates(
            mm_items, hf_processor_mm_kwargs, out_mm_kwargs
        )
        prompt_updates.extend(u for u in base_updates if u.modality != "image")
        return prompt_updates


class QaicQwen2VLMultiModalDataParser(Qwen2VLMultiModalDataParser):
    def __init__(self, spatial_merge_size: int, *args, **kwargs):
        self.image_grid_thw_lookup = kwargs.pop("image_grid_thw_lookup", None)
        super().__init__(spatial_merge_size, *args, **kwargs)

    def _parse_image_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[ImageItem],
    ) -> ModalityDataItems[Any, Any] | None:
        # FIXME: Limitation for Qwen2.5VL and Qwen3VL model on QAIC:
        # All images in the same request must be in the same size.
        # This could be resolved once EC Connector is implemented
        if isinstance(data, dict):
            if (
                self.image_grid_thw_lookup
                and "image_embeds" in data
                and "image_grid_thw" in data
            ):
                image_grid_thw = data["image_grid_thw"]
                # _reduce_data (called in _merge_embeds) adds a leading batch
                # dim via unsqueeze(0), turning [N, 3] -> [1, N, 3].
                # Squeeze it back to [N, 3] to keep _parse_image_data correct
                # for both single-frame ([1,3]) and multi-frame ([N,3]) cases.
                if _TRANSFORMERS_NEW_IMAGE_PROCESSOR and image_grid_thw.ndim == 3:
                    image_grid_thw = image_grid_thw.squeeze(0)
                    data["image_grid_thw"] = image_grid_thw
                image_embeds = data["image_embeds"]
                num_frames = image_grid_thw.shape[0]

                if num_frames == 0:
                    return super()._parse_image_data(data)

                # Determine vision size
                # based on whether embeds are batched per frame or flattened.
                if image_embeds.shape[0] == num_frames:
                    vision_size = image_embeds.shape[-2]
                else:
                    vision_size = image_embeds.shape[-2] // num_frames

                if vision_size not in self.image_grid_thw_lookup:
                    available_sizes = list(self.image_grid_thw_lookup.keys())
                    raise ValueError(
                        f"Vision size (image_embeds tokens per image) {vision_size} "
                        f"does not match any of the resolutions compiled into the QPC. "
                        f"Available vision sizes: {available_sizes}. "
                        f"Please make sure to provide the same height and width values "
                        f"for vision encoder and language decoder. "
                    )

                # For e-pd, we use dummy placeholder [-1, -1, -1] for image_grid_thw
                # to avoid downloading and processing the image on the proxy server.
                # Replace dummy placeholder image_grid_thw with actual values
                # corresponding to supported resolutions.
                is_dummy_placeholder = image_grid_thw[0][0] == -1
                if is_dummy_placeholder or not any(
                    torch.equal(image_grid_thw[0], v)
                    for v in self.image_grid_thw_lookup.values()
                ):
                    data["image_grid_thw"] = self.image_grid_thw_lookup[
                        vision_size
                    ].repeat(image_grid_thw.shape[0], 1)

            return DictEmbeddingItems(
                data,
                modality="image",
                required_fields={"image_embeds", "image_grid_thw"},
                fields_factory=_create_qwen2vl_field_factory(self._spatial_merge_size),
            )
        return super()._parse_image_data(data)


class QaicQwen2VLImageProcessorFast(Qwen2VLImageProcessorFast):
    resolutions: list[tuple] | None = None

    if is_torchvision_available():
        if is_torchvision_v2_available():
            from torchvision.transforms.v2 import functional as F
        else:
            from torchvision.transforms import functional as F

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        # transformers 4.57.x uses 'interpolation'; 5.5.x uses 'resample'.
        # Accept both names via **kwargs to stay compatible with either version.
        # interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ):
        """
        Preprocess an image or batch of images.
        Copy of the `_preprocess` method from `Qwen2VLImageProcessorFast`,
        with the addition of `select_best_resolution` so that
        images are resized to one of the known resolutions compiled into the QPC.
        """
        # Resolve the resampling filter regardless of which name the base class uses.
        interpolation = kwargs.pop("resample", None) or kwargs.pop(
            "interpolation", None
        )
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(
            images, disable_grouping=disable_grouping
        )
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"]
                    if _TRANSFORMERS_NEW_IMAGE_PROCESSOR
                    else self.min_pixels,
                    max_pixels=size["longest_edge"]
                    if _TRANSFORMERS_NEW_IMAGE_PROCESSOR
                    else self.max_pixels,
                )
                if (
                    self.resolutions
                    and (resized_height, resized_width) not in self.resolutions
                ):
                    resized_height, resized_width = select_best_resolution(
                        (resized_height, resized_width), self.resolutions
                    )
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False,
        # or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(
            resized_images, disable_grouping=disable_grouping
        )
        processed_images_grouped = {}
        processed_grids = {}
        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]
            # Fused rescale and normalize
            patches = self.rescale_and_normalize(
                stacked_images,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            if patches.ndim == 4:
                # add a temporal dimension if we have images
                patches = patches.unsqueeze(1)
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)
            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            # Reorder dimensions to group grid and patch information
            # for subsequent flattening.
            # (batch, grid_t, grid_h, grid_w,
            #  merge_h, merge_w, channel,
            #  temp_patch_size, patch_h, patch_w)
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(
            processed_images_grouped, grouped_images_index
        )
        processed_grids = reorder_images(processed_grids, grouped_images_index)
        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )


# Keyed by id(model_config) so each LLM instance gets its own cache entry.
_qwenvl_processor_config: defaultdict[int, dict] = defaultdict(dict)


class QaicQwenVLProcessingInfoOverrideInitMixin:
    ctx: InputProcessingContext  # type hint
    _hf_processor: ProcessorMixin = None

    def get_hf_config(self) -> Any:
        # Stub for type checkers
        raise NotImplementedError

    def _compute_and_cache_resolutions(self) -> None:
        model_config = self.ctx.model_config
        cache_key = id(model_config)
        cache_entry = _qwenvl_processor_config[cache_key]

        if "resolutions" not in cache_entry:
            from QEfficient.utils.constants import QWEN2_5_VL_HEIGHT, QWEN2_5_VL_WIDTH

            height = cache_entry.get("height", [QWEN2_5_VL_HEIGHT])
            width = cache_entry.get("width", [QWEN2_5_VL_WIDTH])
            grid_t = 1  # video not supported

            vision_config = self.get_hf_config().vision_config
            patch_size = vision_config.patch_size
            merge_size = vision_config.spatial_merge_size
            mm_processor_kwargs = model_config.mm_processor_kwargs or {}

            image_grid_thw_lookup = {}
            resized_resolutions = []
            for h, w in zip(height, width, strict=False):
                resized_height, resized_width = smart_resize(
                    height=h,
                    width=w,
                    factor=patch_size * merge_size,
                    min_pixels=mm_processor_kwargs.get("min_pixels"),
                    max_pixels=mm_processor_kwargs.get("max_pixels"),
                )
                resized_resolutions.append((resized_height, resized_width))
                grid_h = resized_height // patch_size
                grid_w = resized_width // patch_size
                vision_size = grid_t * grid_h * grid_w // (merge_size**2)
                image_grid_thw_lookup[vision_size] = torch.tensor(
                    [grid_t, grid_h, grid_w]
                )

            cache_entry["resolutions"] = resized_resolutions
            cache_entry["image_grid_thw_lookup"] = image_grid_thw_lookup

        self._resolutions = cache_entry["resolutions"]
        self.image_grid_thw_lookup = cache_entry["image_grid_thw_lookup"]


class QaicQwen2_5_VLProcessingInfo(
    Qwen2_5_VLProcessingInfo,
    QaicQwenVLProcessingInfoOverrideInitMixin,
):
    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__(ctx)
        self._hf_processor = None
        self._compute_and_cache_resolutions()

    def get_hf_processor(self, **kwargs: object) -> Qwen2_5_VLProcessor:
        # Limitation (QAIC): kwargs are only consumed on the first call;
        # subsequent calls return the cached processor regardless of kwargs.
        # As a result, `min_pixels` and `max_pixels` must be defined at
        # initialization and cannot be overridden per request.
        if self._hf_processor is None:
            self._hf_processor = self.ctx.get_hf_processor(
                Qwen2_5_VLProcessor,
                use_fast=kwargs.pop("use_fast", True),
                **kwargs,
            )
            if not isinstance(
                self._hf_processor.image_processor, QaicQwen2VLImageProcessorFast
            ):
                image_processor = QaicQwen2VLImageProcessorFast.from_dict(
                    self._hf_processor.image_processor.to_dict()
                )
                if isinstance(image_processor, tuple):
                    image_processor = image_processor[0]
                image_processor.resolutions = self._resolutions
                self._hf_processor.image_processor = image_processor
        return self._hf_processor


class QaicQwen2_5_VLMultiModalProcessor(Qwen2_5_VLMultiModalProcessor):
    def _get_data_parser(self) -> MultiModalDataParser:
        return QaicQwen2VLMultiModalDataParser(
            self.info.get_hf_config().vision_config.spatial_merge_size,
            image_grid_thw_lookup=self.info.image_grid_thw_lookup,
        )


class QaicQwen3VLProcessingInfo(
    Qwen3VLProcessingInfo,
    QaicQwenVLProcessingInfoOverrideInitMixin,
):
    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__(ctx)
        self._hf_processor = None
        self._compute_and_cache_resolutions()

    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        # Limitation (QAIC): kwargs are only consumed on the first call;
        # subsequent calls return the cached processor regardless of kwargs.
        # As a result, `min_pixels` and `max_pixels` must be defined at
        # initialization and cannot be overridden per request.
        if self._hf_processor is None:
            self._hf_processor = self.ctx.get_hf_processor(
                Qwen3VLProcessor,
                use_fast=kwargs.pop("use_fast", True),
                **kwargs,
            )
            if not isinstance(
                self._hf_processor.image_processor, QaicQwen2VLImageProcessorFast
            ):
                image_processor = QaicQwen2VLImageProcessorFast.from_dict(
                    self._hf_processor.image_processor.to_dict()
                )
                if isinstance(image_processor, tuple):
                    image_processor = image_processor[0]
                image_processor.resolutions = self._resolutions
                self._hf_processor.image_processor = image_processor
        return self._hf_processor


class QaicQwen3VLMultiModalProcessor(Qwen3VLMultiModalProcessor):
    def _get_data_parser(self) -> MultiModalDataParser:
        return QaicQwen2VLMultiModalDataParser(
            self.info.get_hf_config().vision_config.spatial_merge_size,
            video_needs_metadata=True,
            image_grid_thw_lookup=self.info.image_grid_thw_lookup,
        )


class QaicQwen3VLMoeProcessingInfo(QaicQwen3VLProcessingInfo, Qwen3VLMoeProcessingInfo):
    pass


class QaicQwen3_5ProcessingInfo(QaicQwen3VLProcessingInfo, Qwen3_5ProcessingInfo):
    pass


class QaicQwen3_5MoeProcessingInfo(QaicQwen3VLProcessingInfo, Qwen3_5MoeProcessingInfo):
    pass


def register_qaic_custom_mm_processor(model_type: str):
    MODEL_PROCESSOR_MAP = {
        "qwen2_5_vl": (
            QaicQwen2_5_VLMultiModalProcessor,
            QaicQwen2_5_VLProcessingInfo,
            Qwen2_5_VLDummyInputsBuilder,
            Qwen2_5_VLForConditionalGeneration,
        ),
        "qwen3_vl": (
            QaicQwen3VLMultiModalProcessor,
            QaicQwen3VLProcessingInfo,
            Qwen3VLDummyInputsBuilder,
            Qwen3VLForConditionalGeneration,
        ),
        "qwen3_vl_moe": (
            QaicQwen3VLMultiModalProcessor,
            QaicQwen3VLMoeProcessingInfo,
            Qwen3VLDummyInputsBuilder,
            Qwen3VLMoeForConditionalGeneration,
        ),
        "qwen3_5": (
            QaicQwen3VLMultiModalProcessor,
            QaicQwen3_5ProcessingInfo,
            Qwen3VLDummyInputsBuilder,
            Qwen3_5ForConditionalGeneration,
        ),
        "qwen3_5_moe": (
            QaicQwen3VLMultiModalProcessor,
            QaicQwen3_5MoeProcessingInfo,
            Qwen3VLDummyInputsBuilder,
            Qwen3_5MoeForConditionalGeneration,
        ),
        "gemma3": (
            QaicGemma3MultiModalProcessor,
            Gemma3ProcessingInfo,
            Gemma3DummyInputsBuilder,
            Gemma3ForConditionalGeneration,
        ),
        "gemma4": (
            QaicGemma4MultiModalProcessor,
            Gemma4ProcessingInfo,
            Gemma4DummyInputsBuilder,
            Gemma4ForConditionalGeneration,
        ),
    }

    if model_type in MODEL_PROCESSOR_MAP:
        processor_cls, info_cls, dummy_cls, model_cls = MODEL_PROCESSOR_MAP[model_type]
        MULTIMODAL_REGISTRY.register_processor(
            processor_cls,
            info=info_cls,
            dummy_inputs=dummy_cls,
        )(model_cls)

        # For gemma4, the decode instance resolves to
        # TransformersMultiModalForCausalLM (the generic Transformers fallback)
        # rather than Gemma4ForConditionalGeneration. Re-register that class
        # with the same QAIC-aware processor so it handles ImageEmbeddingItems.
        if model_type == "gemma4":
            MULTIMODAL_REGISTRY.register_processor(
                QaicGemma4MultiModalProcessor,
                info=Gemma4ProcessingInfo,
                dummy_inputs=Gemma4DummyInputsBuilder,
            )(TransformersMultiModalForCausalLM)
