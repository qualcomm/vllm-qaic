# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import vllm.engine.arg_utils as arg_utils

from vllm_qaic.platform import QaicPlatform

_orig_create_model_config = arg_utils.EngineArgs.create_model_config


def _patched_create_model_config(self):
    # ModelConfig resolves dtype="auto" from current_platform.supported_dtypes,
    # which depends on the AIC hardware version of the target device. The
    # device comes from additional_config, which the platform otherwise only
    # sees later in check_and_update_config, so hand it over first.
    QaicPlatform.set_engine_additional_config(self.additional_config)
    return _orig_create_model_config(self)


arg_utils.EngineArgs.create_model_config = _patched_create_model_config
