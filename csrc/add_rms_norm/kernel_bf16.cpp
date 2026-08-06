// ---------------------------------------------------------------------------------------
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
// ---------------------------------------------------------------------------------------

#include "QAicHexagonMath.h"
#include "QAicHexagonPlatformIntf.h"
#include "QAicHexagonReducer.h"
#include "QAicHexagonUtils.h"
#include "jit_dev_exe_function.h"
#include "jit_dev_status_codes.h"
#include "jit_qshim_api.h"
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#if __HEXAGON_ARCH__ < 81
#error "rms_norm_bf16 requires HVX V81 or later (AI200+)"
#endif

extern "C" void qaicSyncHVXThread(uint32_t threadId);

static inline void sync_hvx_threads(uint32_t threadID, uint32_t numThreads) {
  if (numThreads > 1) {
    qaicSyncHVXThread(threadID);
  }
}

static inline uint32_t align_up_u32(uint32_t x, uint32_t a) {
  return (x + a - 1) & ~(a - 1);
}

static inline uint8_t *align_up_ptr(uint8_t *p, uintptr_t a) {
  uintptr_t v = (uintptr_t)p;
  v = (v + a - 1) & ~(a - 1);
  return (uint8_t *)v;
}

static QShimUDmaHandle qaic_linear_udma_submit(uint32_t threadId, AicJitPtr src,
                                               uint32_t size, AicJitPtr dst,
                                               uint32_t udmaDescAttrsOrder,
                                               bool requireHandle,
                                               uint32_t *status) {
  AicJitUdmaDescCommonAttrs udmaDescAttrs = {};
  udmaDescAttrs.order = udmaDescAttrsOrder;
  // udmaDescAttrs.bypassOverride = 0;

  return qshimLinearUdmaSubmit(threadId, src, size, dst, &udmaDescAttrs,
                               requireHandle, status);
}

static inline uint32_t dma_copy_wait(uint32_t threadId, void *dst,
                                     const void *src, uint32_t bytes,
                                     uint32_t order = 0) {
  uint32_t status = JIT_DEV_STATUS_SUCCESS;

  QShimUDmaHandle h = qaic_linear_udma_submit(
      threadId, (AicJitPtr)src, bytes, (AicJitPtr)dst, order, true, &status);

  if (status != JIT_DEV_STATUS_SUCCESS || h == INVALID_UDMA_HANDLE) {
    return status;
  }

  return qshimUDmaWait(h);
}

static inline QShimUDmaHandle dma_copy_submit(uint32_t threadId, void *dst,
                                              const void *src, uint32_t bytes,
                                              uint32_t *status,
                                              uint32_t order = 0) {
  return qaic_linear_udma_submit(threadId, (AicJitPtr)src, bytes, (AicJitPtr)dst,
                                 order, true, status);
}

// Per-row fused add+RMS-norm kernel (BF16, requires HVX V81 / AI200+).
// Computes: residual = attn_out + x; dst = residual / rms(residual) * weight
extern "C" void _single_nsp_rms_norm_bf16(
    const uint16_t *attn_out, const uint16_t *x, const uint16_t *weight,
    uint16_t *out, // residual output (attn_out + x), written in phase 1
    uint16_t *dst, // normalized output, written in phase 2
    float epsilon, int N, uint32_t threadID, uint32_t localThreadID,
    uint32_t threadsPerCore, float *partial_sums) {
  constexpr int elems_per_vec = sizeof(HVX_Vector) / sizeof(uint16_t);
  const int vlen = N / elems_per_vec;
  float local_sum_f32 = 0.0f;
  SumReducerFloat sqReducer;

  // Phase 1: add residual and accumulate sum-of-squares for RMS.
  for (int i = (int)localThreadID; i < vlen; i += (int)threadsPerCore) {
    const uint16_t *src1_ptr = attn_out + i * elems_per_vec;
    const uint16_t *src2_ptr = x + i * elems_per_vec;
    uint16_t *tmp_ptr = out + i * elems_per_vec;

    HVX_Vector vec1 = *(HVX_Vector *)src1_ptr;
    HVX_Vector vec2 = *(HVX_Vector *)src2_ptr;

    HVX_VectorPair sum_f32 = Q6_Wsf_vadd_VbfVbf(vec1, vec2);

    *(HVX_Vector *)tmp_ptr =
        Q6_Vbf_vcvt_VsfVsf(Q6_V_lo_W(sum_f32), Q6_V_hi_W(sum_f32));

    HVX_Vector sq_lo =
        Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(sum_f32), Q6_V_lo_W(sum_f32));
    HVX_Vector sq_hi =
        Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(sum_f32), Q6_V_hi_W(sum_f32));
    sqReducer.reduce(sq_lo);
    sqReducer.reduce(sq_hi);
  }

  sqReducer.finish(&local_sum_f32);

  // Cross-thread reduction: each thread writes its partial sum, thread 0
  // computes inv_rms
  partial_sums[localThreadID] = local_sum_f32;

  sync_hvx_threads(threadID, threadsPerCore);
  if (localThreadID == 0) {
    float total_sum_f32 = 0.0f;
    for (int t = 0; t < (int)threadsPerCore; t++) {
      total_sum_f32 += partial_sums[t];
    }
    float inv_rms =
        1.0f / __builtin_sqrtf((total_sum_f32 / (float)N) + epsilon);
    partial_sums[0] = inv_rms;
  }

  // Phase 2: normalize and apply weight.
  sync_hvx_threads(threadID, threadsPerCore);
  float inv_rms = partial_sums[0];

  HVX_Vector inv_rms_sf = Q6_V_vsplat_R(*(const uint32_t *)&inv_rms);

  const HVX_Vector one_bf16 = Q6_Vh_vsplat_R(0x3F80);

  for (int i = (int)localThreadID; i < vlen; i += (int)threadsPerCore) {
    uint16_t *residual_ptr = out + i * elems_per_vec;
    const uint16_t *weight_ptr = weight + i * elems_per_vec;
    uint16_t *dst_ptr = dst + i * elems_per_vec;

    HVX_Vector residual_bf = *(HVX_Vector *)residual_ptr;
    HVX_Vector w_v = *(HVX_Vector *)weight_ptr;

    HVX_VectorPair res_f32 = Q6_Wsf_vmpy_VbfVbf(residual_bf, one_bf16);
    HVX_VectorPair w_f32 = Q6_Wsf_vmpy_VbfVbf(w_v, one_bf16);

    HVX_Vector norm_lo = Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(res_f32), inv_rms_sf);
    HVX_Vector norm_hi = Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(res_f32), inv_rms_sf);

    HVX_Vector out_lo = Q6_Vsf_vmpy_VsfVsf(norm_lo, Q6_V_lo_W(w_f32));
    HVX_Vector out_hi = Q6_Vsf_vmpy_VsfVsf(norm_hi, Q6_V_hi_W(w_f32));

    *(HVX_Vector *)dst_ptr = Q6_Vbf_vcvt_VsfVsf(out_lo, out_hi);
  }

  sync_hvx_threads(threadID, threadsPerCore);
}

// NSP entry point: drives rms_norm_bf16 row-by-row over a (B, M, N) tensor.
// Pointers layout: [0]=attn_out, [1]=x, [2]=weight, [3]=residual_out, [4]=dst
//   [5]=epsilon (float scalar), [6]=B (int scalar), [7]=M (int scalar), [8]=N (int scalar)
// Each core owns a stripe of rows across all B*M rows.
// Double-buffered DMA (slots 0/1) overlaps prefetch of row r+1 with
// compute on row r.
QAIC_KERNEL_API uint32_t rms_norm_multi_nsp_bf16(
    const AicJitEntryPointConfig *cfg, const AicJitPointerArray *ptrs) {
  const uint16_t *attn_out_ddr = (const uint16_t *)ptrs->pointers[0];
  const uint16_t *x_ddr = (const uint16_t *)ptrs->pointers[1];
  const uint16_t *weight_ddr = (const uint16_t *)ptrs->pointers[2];
  uint16_t *out_ddr = (uint16_t *)ptrs->pointers[3];
  uint16_t *dst_ddr = (uint16_t *)ptrs->pointers[4];
  float epsilon = *(const float *)ptrs->pointers[5];
  const int N = *(const int32_t *)ptrs->pointers[6];
  const int total_elems = *(const int32_t *)ptrs->pointers[7];
  const int total_rows = total_elems / N;

  uint32_t threadID = cfg->threadID;
  uint32_t numThreads = cfg->numThreads;
  uint32_t numCores = cfg->numCores;
  uint32_t coreID = cfg->coreID;

  // localThreadID is the index within this core's HVX thread pool
  uint32_t localThreadID = threadID % numThreads;

  constexpr uint32_t kAlign = 128;
  constexpr int elems_per_vec = sizeof(HVX_Vector) / sizeof(uint16_t);

  // N must be an exact multiple of the HVX vector width (128 bytes / 2 bytes =
  // 64 elements)
  if ((N % elems_per_vec) != 0) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  // VTCM layout (all 128-byte aligned):
  //   attn_vtcm[0/1]  — double-buffer for attn_out rows
  //   x_vtcm[0/1]     — double-buffer for x rows
  //   weight_vtcm     — weight row (loaded once, reused for every row)
  //   out_vtcm        — residual scratch (written by rms_norm_bf16 phase 1)
  //   dst_vtcm        — normalized output scratch (written by rms_norm_bf16
  //   phase 2) partial_sums    — per-thread FP32 accumulator array status_vtcm
  //   — single shared DMA status word prefetch_handle — in-flight x-prefetch
  //   DMA handle
  const uint32_t rowBytes = (uint32_t)N * sizeof(uint16_t);
  const uint32_t rowBytesAligned = align_up_u32(rowBytes, kAlign);
  const uint32_t partialBytes =
      align_up_u32(numThreads * sizeof(float), kAlign);
  const uint32_t statusBytes = align_up_u32(sizeof(uint32_t), kAlign);
  const uint32_t handleBytes = align_up_u32(sizeof(QShimUDmaHandle), kAlign);
  const uint32_t requiredVtcmBytes =
      7 * rowBytesAligned + partialBytes + statusBytes + handleBytes + kAlign;

  int64_t vtcmSize = 0;
  uint32_t ret = qshimQuery(DEV_ATTR_QSHIM_VTCM_SIZE, &vtcmSize);

  if (ret != JIT_DEV_STATUS_SUCCESS) {
    return ret;
  }

  if ((uint64_t)requiredVtcmBytes > (uint64_t)vtcmSize) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  uint8_t *vtcmBase = qshimGetBaseVtcmAddr();
  uint8_t *vtcmPtr = align_up_ptr(vtcmBase, kAlign);

  uint16_t *attn_vtcm[2];
  attn_vtcm[0] = (uint16_t *)vtcmPtr;
  vtcmPtr += rowBytesAligned;
  attn_vtcm[1] = (uint16_t *)vtcmPtr;
  vtcmPtr += rowBytesAligned;

  uint16_t *x_vtcm[2];
  x_vtcm[0] = (uint16_t *)vtcmPtr;
  vtcmPtr += rowBytesAligned;
  x_vtcm[1] = (uint16_t *)vtcmPtr;
  vtcmPtr += rowBytesAligned;

  uint16_t *weight_vtcm = (uint16_t *)vtcmPtr;
  vtcmPtr += rowBytesAligned;
  uint16_t *out_vtcm = (uint16_t *)vtcmPtr;
  vtcmPtr += rowBytesAligned;
  uint16_t *dst_vtcm = (uint16_t *)vtcmPtr;
  vtcmPtr += rowBytesAligned;

  float *partial_sums = (float *)align_up_ptr(vtcmPtr, kAlign);
  vtcmPtr = (uint8_t *)partial_sums + partialBytes;

  uint32_t *status_vtcm = (uint32_t *)align_up_ptr(vtcmPtr, kAlign);
  vtcmPtr = (uint8_t *)status_vtcm + statusBytes;

  QShimUDmaHandle *prefetch_handle_vtcm =
      (QShimUDmaHandle *)align_up_ptr(vtcmPtr, kAlign);

  // Weight is the same for all rows; load it once before the main loop
  if (localThreadID == 0) {
    *status_vtcm = dma_copy_wait(threadID, weight_vtcm, weight_ddr, rowBytes);
  }
  sync_hvx_threads(threadID, numThreads);
  if (*status_vtcm != JIT_DEV_STATUS_SUCCESS) {
    return *status_vtcm;
  }

  // Each core processes ceil(total_rows / numCores) rows
  const int rowIters = (total_rows + (int)numCores - 1) / (int)numCores;

  // Prime slot 0 with the first row assigned to this core before entering the
  // loop
  {
    const int r0 = (int)coreID;
    const bool valid0 = (r0 < total_rows);
    if (localThreadID == 0) {
      if (valid0) {
        *status_vtcm = dma_copy_wait(threadID, attn_vtcm[0],
                                     attn_out_ddr + r0 * N, rowBytes);
        if (*status_vtcm == JIT_DEV_STATUS_SUCCESS) {
          *status_vtcm =
              dma_copy_wait(threadID, x_vtcm[0], x_ddr + r0 * N, rowBytes);
        }
      } else {
        *status_vtcm = JIT_DEV_STATUS_SUCCESS;
      }
      *prefetch_handle_vtcm = INVALID_UDMA_HANDLE;
    }
    sync_hvx_threads(threadID, numThreads);
    if (*status_vtcm != JIT_DEV_STATUS_SUCCESS) {
      return *status_vtcm;
    }
  }

  for (int iter = 0; iter < rowIters; ++iter) {
    const int r      = iter * (int)numCores + (int)coreID;
    const int r_next = (iter + 1) * (int)numCores + (int)coreID;
    const bool validRow     = (r      < total_rows);
    const bool validNextRow = (r_next < total_rows);
    // Ping-pong between slot 0 and 1 each iteration
    const int cur_slot = iter & 1;
    const int next_slot = cur_slot ^ 1;

    uint16_t *row_dst_ddr = validRow ? (dst_ddr + r * N) : dst_ddr;
    uint16_t *row_out_ddr = validRow ? (out_ddr + r * N) : out_ddr;

    // Thread 0 submits async DMA for the next row while all threads compute the
    // current row. attn prefetch is waited immediately (sequential); x prefetch
    // is left in-flight and waited after compute completes to maximally overlap
    // with rms_norm_bf16 execution.
    if (localThreadID == 0) {
      if (validNextRow) {
        uint32_t dma_status = JIT_DEV_STATUS_SUCCESS;
        QShimUDmaHandle h_attn =
            dma_copy_submit(threadID, attn_vtcm[next_slot],
                            attn_out_ddr + r_next * N, rowBytes, &dma_status);
        if (dma_status != JIT_DEV_STATUS_SUCCESS ||
            h_attn == INVALID_UDMA_HANDLE) {
          *status_vtcm = (dma_status != JIT_DEV_STATUS_SUCCESS)
                             ? dma_status
                             : JIT_DEV_ERROR_INVALID_PARAMETER;
          *prefetch_handle_vtcm = INVALID_UDMA_HANDLE;
        } else {
          uint32_t wait_st = qshimUDmaWait(h_attn);
          if (wait_st != JIT_DEV_STATUS_SUCCESS) {
            *status_vtcm = wait_st;
            *prefetch_handle_vtcm = INVALID_UDMA_HANDLE;
          } else {
            // Kick off x DMA and leave it in-flight; save handle for
            // post-compute wait
            QShimUDmaHandle h_x =
                dma_copy_submit(threadID, x_vtcm[next_slot], x_ddr + r_next * N,
                                rowBytes, &dma_status);
            *prefetch_handle_vtcm = (dma_status == JIT_DEV_STATUS_SUCCESS)
                                        ? h_x
                                        : INVALID_UDMA_HANDLE;
            if (dma_status != JIT_DEV_STATUS_SUCCESS) {
              *status_vtcm = dma_status;
            }
          }
        }
      } else {
        *prefetch_handle_vtcm = INVALID_UDMA_HANDLE;
      }
    }

    if (validRow) {
      _single_nsp_rms_norm_bf16(attn_vtcm[cur_slot], x_vtcm[cur_slot], weight_vtcm,
                    out_vtcm, dst_vtcm, epsilon, N, threadID, localThreadID,
                    numThreads, partial_sums);
    } else {
      // Idle row: _single_nsp_rms_norm_bf16 contains 3 sync barriers; mirror them to keep
      // all threads in lockstep
      sync_hvx_threads(threadID, numThreads);
      sync_hvx_threads(threadID, numThreads);
      sync_hvx_threads(threadID, numThreads);
    }

    // After compute: wait for the in-flight x prefetch, then DMA the result
    // back to DDR
    if (localThreadID == 0) {
      QShimUDmaHandle h = *prefetch_handle_vtcm;
      if (h != INVALID_UDMA_HANDLE) {
        uint32_t wait_st = qshimUDmaWait(h);
        if (wait_st != JIT_DEV_STATUS_SUCCESS) {
          *status_vtcm = wait_st;
        }
      }
      if (validRow && *status_vtcm == JIT_DEV_STATUS_SUCCESS) {
        *status_vtcm = dma_copy_wait(threadID, row_dst_ddr, dst_vtcm, rowBytes);
      }
      if (validRow && *status_vtcm == JIT_DEV_STATUS_SUCCESS) {
        *status_vtcm = dma_copy_wait(threadID, row_out_ddr, out_vtcm, rowBytes);
      }
    }

    sync_hvx_threads(threadID, numThreads);

    if (*status_vtcm != JIT_DEV_STATUS_SUCCESS) {
      return *status_vtcm;
    }
  }

  sync_hvx_threads(threadID, numThreads);
  return JIT_DEV_STATUS_SUCCESS;
}
