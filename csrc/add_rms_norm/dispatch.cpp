// ---------------------------------------------------------------------------------------
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
// ---------------------------------------------------------------------------------------

// Dtype-dispatching NSP entry point for fused add+RMS-norm.
// Reads params[3] to select between the FP16 and BF16 kernels at runtime.
// The wrapper must set params = {epsilon, M, N, dtype} where dtype is one of
// QAIC_RMS_NORM_DTYPE_FP16 (0) or QAIC_RMS_NORM_DTYPE_BF16 (1).
//
// Build behaviour:
//   v68 (AI100): kernel_bf16.cpp is excluded from the build; only FP16 is
//   compiled
//                into the .so. Requesting BF16 returns
//                JIT_DEV_ERROR_INVALID_PARAMETER.
//   v81+ (AI200): both kernels are compiled. Dispatch selects by dtype at
//   runtime.

#include "QAicHexagonPlatformIntf.h"
#include "QAicHexagonUtils.h"
#include "jit_dev_exe_function.h"
#include "jit_dev_status_codes.h"

#define QAIC_RMS_NORM_DTYPE_FP16 0
#define QAIC_RMS_NORM_DTYPE_BF16 1

// Forward declarations for symbols defined in kernel.cpp / kernel_bf16.cpp.
// Use plain extern "C" — QAIC_KERNEL_API adds visibility/alignment attributes
// that belong only on the definition, not the declaration.
extern "C" uint32_t rms_norm_multi_nsp(const AicJitEntryPointConfig *cfg,
                                       const AicJitPointerArray *ptrs);

#if __HEXAGON_ARCH__ >= 81
extern "C" uint32_t rms_norm_multi_nsp_bf16(const AicJitEntryPointConfig *cfg,
                                            const AicJitPointerArray *ptrs);
#endif

// rms_norm_dispatch — single entry point called by the wrapper.
// Pointer layout (same as the individual kernels):
//   ptrs->pointers[0] : attn_out    (input)
//   ptrs->pointers[1] : x           (input)
//   ptrs->pointers[2] : weight      [N]
//   ptrs->pointers[3] : dst         (normed output)
//   ptrs->pointers[4] : residual    (attn_out + x)
//   ptrs->pointers[5] : epsilon     float scalar
//   ptrs->pointers[6] : N           int scalar (weight.numel())
//   ptrs->pointers[7] : total_elems int scalar (attn_out.numel())
//   ptrs->pointers[8] : dtype       int scalar (0=FP16, 1=BF16)
QAIC_KERNEL_API uint32_t rms_norm_dispatch(const AicJitEntryPointConfig *cfg,
                                           const AicJitPointerArray *ptrs) {
  const int dtype = *(const int32_t *)ptrs->pointers[8];

#if __HEXAGON_ARCH__ >= 81
  if (dtype == QAIC_RMS_NORM_DTYPE_BF16) {
    return rms_norm_multi_nsp_bf16(cfg, ptrs);
  }
#else
  if (dtype == QAIC_RMS_NORM_DTYPE_BF16) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }
#endif

  return rms_norm_multi_nsp(cfg, ptrs);
}
