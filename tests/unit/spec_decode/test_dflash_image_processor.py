# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pytest
import torch
from transformers import Qwen2VLImageProcessorFast
from transformers.image_utils import PILImageResampling, SizeDict

from vllm_qaic.model_loader.qaic_custom_mm_processor import (
    QaicQwen2VLImageProcessorFast,
)


@pytest.mark.parametrize("shape", [(448, 448), (357, 519)])
@pytest.mark.parametrize(
    "resample", [PILImageResampling.NEAREST, PILImageResampling.BICUBIC]
)
@pytest.mark.parametrize("constrained", [False, True])
def test_qwen_image_processor_matches_reference(shape, resample, constrained):
    kwargs = dict(
        patch_size=16,
        merge_size=2,
        temporal_patch_size=2,
        min_pixels=4096,
        max_pixels=16777216,
        resample=resample,
    )
    processor = QaicQwen2VLImageProcessorFast(**kwargs)
    reference = Qwen2VLImageProcessorFast(**kwargs)
    height, width = shape
    image = torch.arange(3 * height * width).remainder(256).to(torch.uint8)
    image = image.reshape(3, height, width)
    reference_image = image
    if constrained:
        processor.resolutions = [(448, 448)]
        # The QAIC QPC requires the selected fixed resolution, not dynamic resizing.
        reference_image = super(Qwen2VLImageProcessorFast, reference).resize(
            image=image,
            size=SizeDict(height=448, width=448),
            resample=resample,
        )
    actual = processor(images=image, return_tensors="pt")
    expected = reference(
        images=reference_image, do_resize=not constrained, return_tensors="pt"
    )
    torch.testing.assert_close(actual["image_grid_thw"], expected["image_grid_thw"])
    torch.testing.assert_close(actual["pixel_values"], expected["pixel_values"])
    if constrained:
        assert actual["image_grid_thw"].tolist() == [[1, 28, 28]]
