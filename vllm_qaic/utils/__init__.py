# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

QAIC_QUANTIZATION_METHOD = "mxfp6"
QAIC_KV_CACHE_DTYPE = "mxint8"

QAIC_QUANTIZATION_LIST = [
    QAIC_QUANTIZATION_METHOD,
    "awq",
    "gptq",
    "fp8",
    "compressed-tensors",
    "mxfp4",
]


def get_qaic_kv_connector_names(kv_transfer_config) -> set[str]:
    if kv_transfer_config is None:
        return set()
    names = {kv_transfer_config.kv_connector}
    if kv_transfer_config.kv_connector == "MultiConnector":
        names.update(
            child["kv_connector"]
            for child in kv_transfer_config.kv_connector_extra_config.get(
                "connectors", []
            )
            if "kv_connector" in child
        )
    return names
