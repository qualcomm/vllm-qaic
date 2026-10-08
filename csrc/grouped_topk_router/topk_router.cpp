// ---------------------------------------------------------------------------------------
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
// ---------------------------------------------------------------------------------------

#include "grouped_topk_router_common.h"

QAIC_KERNEL_API int32_t
multinsp_multithreaded_topk_router(const AicJitEntryPointConfig* entryConfig,
                                   const AicJitPointerArray* pointerArray) {
  return grouped_topk_router::regular_kernel_main(entryConfig, pointerArray);
}
