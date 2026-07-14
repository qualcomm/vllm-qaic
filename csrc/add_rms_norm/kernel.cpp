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

extern "C" void qaicSyncHVXThread(uint32_t threadId);

static inline void sync_hvx_threads(uint32_t threadID, uint32_t numThreads) {
  if (numThreads > 1) {
    qaicSyncHVXThread(threadID);
  }
}

static inline uint32_t align_up_u32(uint32_t x, uint32_t a) {
  return (x + a - 1) & ~(a - 1);
}

static inline uint8_t* align_up_ptr(uint8_t* p, uintptr_t a) {
  uintptr_t v = (uintptr_t)p;
  v = (v + a - 1) & ~(a - 1);
  return (uint8_t*)v;
}

static QShimUDmaHandle qaic_linear_udma_submit(uint32_t threadId, AicJitPtr src,
                                               uint32_t size, AicJitPtr dst,
                                               uint32_t udmaDescAttrsOrder,
                                               bool requireHandle,
                                               uint32_t* status) {
  AicJitUdmaDescCommonAttrs udmaDescAttrs = {};
  udmaDescAttrs.order = udmaDescAttrsOrder;
  // udmaDescAttrs.bypassOverride = 0;

  return qshimLinearUdmaSubmit(threadId, src, size, dst, &udmaDescAttrs,
                               requireHandle, status);
}

static inline uint32_t dma_copy_wait(uint32_t threadId, void* dst,
                                     const void* src, uint32_t bytes,
                                     uint32_t order = 0) {
  uint32_t status = JIT_DEV_STATUS_SUCCESS;

  QShimUDmaHandle h = qaic_linear_udma_submit(
      threadId, (AicJitPtr)src, bytes, (AicJitPtr)dst, order, true, &status);

  if (status != JIT_DEV_STATUS_SUCCESS || h == INVALID_UDMA_HANDLE) {
    return status;
  }

  return qshimUDmaWait(h);
}

static inline QShimUDmaHandle dma_copy_submit(uint32_t threadId, void* dst,
                                              const void* src, uint32_t bytes,
                                              uint32_t* status,
                                              uint32_t order = 0) {
  return qaic_linear_udma_submit(threadId, (AicJitPtr)src, bytes,
                                 (AicJitPtr)dst, order, true, status);
}

// Per-row fused add+RMS-norm kernel (FP16).
// Computes: residual = attn_out + x; dst = residual / rms(residual) * weight
// Threads within a core stripe-mine the row in chunks of elems_per_vec.
// partial_sums[0..threadsPerCore-1] is a VTCM scratch shared across threads.
extern "C" void _single_nsp_rms_norm(
    const float16* attn_out, const float16* x, const float16* weight,
    float16* out,  // residual output (attn_out + x), written in phase 1
    float16* dst,  // normalized output, written in phase 2
    float epsilon, int N, uint32_t threadID, uint32_t localThreadID,
    uint32_t threadsPerCore, float* partial_sums) {
  constexpr int elems_per_vec = sizeof(HVX_Vector) / sizeof(float16);
  const int vlen = N / elems_per_vec;
  float local_sum_f32 = 0.0f;
  SumReducerFloat sqReducer;

  // Phase 1: add residual and accumulate sum-of-squares for RMS.
  for (int i = (int)localThreadID; i < vlen; i += (int)threadsPerCore) {
    const float16* src1_ptr = attn_out + i * elems_per_vec;
    const float16* src2_ptr = x + i * elems_per_vec;
    float16* tmp_ptr = out + i * elems_per_vec;

    HVX_Vector vec1 = *(HVX_Vector*)src1_ptr;
    HVX_Vector vec2 = *(HVX_Vector*)src2_ptr;

    HVX_Vector sum_qf = Q6_Vqf16_vadd_VhfVhf(vec1, vec2);
    HVX_Vector sum_hf = Q6_Vhf_equals_Vqf16(sum_qf);

    *(HVX_Vector*)tmp_ptr = sum_hf;

    HVX_VectorPair sq_qf32_pair = Q6_Wqf32_vmpy_Vqf16Vqf16(sum_qf, sum_qf);
    HVX_Vector sq_f32_lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(sq_qf32_pair));
    HVX_Vector sq_f32_hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(sq_qf32_pair));

    sqReducer.reduce(sq_f32_lo);
    sqReducer.reduce(sq_f32_hi);
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
  float16 inv_rms_hf = (float16)inv_rms;
  uint32_t inv_rms_bits = 0;
  memcpy(&inv_rms_bits, &inv_rms_hf, sizeof(float16));

  inv_rms_bits = (inv_rms_bits & 0xFFFF) | (inv_rms_bits << 16);
  HVX_Vector inv_rms_vec = Q6_V_vsplat_R(inv_rms_bits);

  for (int i = (int)localThreadID; i < vlen; i += (int)threadsPerCore) {
    float16* residual_ptr = out + i * elems_per_vec;
    const float16* weight_ptr = weight + i * elems_per_vec;
    float16* dst_ptr = dst + i * elems_per_vec;
    HVX_Vector residual_hf = *(HVX_Vector*)residual_ptr;
    HVX_Vector w_v = *(HVX_Vector*)weight_ptr;

    HVX_Vector norm_qf = Q6_Vqf16_vmpy_VhfVhf(residual_hf, inv_rms_vec);

    HVX_Vector out_qf = Q6_Vqf16_vmpy_Vqf16Vhf(norm_qf, w_v);
    HVX_Vector out_hf = Q6_Vhf_equals_Vqf16(out_qf);

    *(HVX_Vector*)dst_ptr = out_hf;
  }

  sync_hvx_threads(threadID, threadsPerCore);
}

// NSP entry point: drives rms_norm row-by-row over an (M, N) matrix.
// Pointers layout: [0]=attn_out, [1]=x, [2]=weight, [3]=residual_out, [4]=dst,
// [5]=params{eps,M,N} Each core owns a stripe of rows (coreID, coreID+numCores,
// ...). Double-buffered DMA (slots 0/1) overlaps prefetch of row m+1 with
// compute on row m.
QAIC_KERNEL_API uint32_t rms_norm_multi_nsp(const AicJitEntryPointConfig* cfg,
                                            const AicJitPointerArray* ptrs) {
  const float16* attn_out_ddr = (const float16*)ptrs->pointers[0];
  const float16* x_ddr = (const float16*)ptrs->pointers[1];
  const float16* weight_ddr = (const float16*)ptrs->pointers[2];
  float16* out_ddr = (float16*)ptrs->pointers[3];
  float16* dst_ddr = (float16*)ptrs->pointers[4];
  const float* params = (const float*)ptrs->pointers[5];

  float epsilon = params[0];
  int M = (int)params[1];
  int N = (int)params[2];

  uint32_t threadID = cfg->threadID;
  uint32_t numThreads = cfg->numThreads;
  uint32_t numCores = cfg->numCores;
  uint32_t coreID = cfg->coreID;

  // localThreadID is the index within this core's HVX thread pool
  uint32_t localThreadID = threadID % numThreads;

  constexpr uint32_t kAlign = 128;
  constexpr int elems_per_vec = sizeof(HVX_Vector) / sizeof(float16);

  // N must be an exact multiple of the HVX vector width (128 bytes / 2 bytes =
  // 64 elements)
  if ((N % elems_per_vec) != 0) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  // VTCM layout (all 128-byte aligned):
  //   attn_vtcm[0/1]  — double-buffer for attn_out rows
  //   x_vtcm[0/1]     — double-buffer for x rows
  //   weight_vtcm     — weight row (loaded once, reused for every row)
  //   out_vtcm        — residual scratch (written by rms_norm phase 1)
  //   dst_vtcm        — normalized output scratch (written by rms_norm phase 2)
  //   partial_sums    — per-thread FP32 accumulator array
  //   status_vtcm     — single shared DMA status word
  //   prefetch_handle — in-flight x-prefetch DMA handle
  const uint32_t rowBytes = (uint32_t)N * sizeof(float16);
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

  uint8_t* vtcmBase = qshimGetBaseVtcmAddr();
  uint8_t* vtcmPtr = align_up_ptr(vtcmBase, kAlign);

  float16* attn_vtcm[2];
  attn_vtcm[0] = (float16*)vtcmPtr;
  vtcmPtr += rowBytesAligned;
  attn_vtcm[1] = (float16*)vtcmPtr;
  vtcmPtr += rowBytesAligned;

  float16* x_vtcm[2];
  x_vtcm[0] = (float16*)vtcmPtr;
  vtcmPtr += rowBytesAligned;
  x_vtcm[1] = (float16*)vtcmPtr;
  vtcmPtr += rowBytesAligned;

  float16* weight_vtcm = (float16*)vtcmPtr;
  vtcmPtr += rowBytesAligned;
  float16* out_vtcm = (float16*)vtcmPtr;
  vtcmPtr += rowBytesAligned;
  float16* dst_vtcm = (float16*)vtcmPtr;
  vtcmPtr += rowBytesAligned;

  float* partial_sums = (float*)align_up_ptr(vtcmPtr, kAlign);
  vtcmPtr = (uint8_t*)partial_sums + partialBytes;

  uint32_t* status_vtcm = (uint32_t*)align_up_ptr(vtcmPtr, kAlign);
  vtcmPtr = (uint8_t*)status_vtcm + statusBytes;

  QShimUDmaHandle* prefetch_handle_vtcm =
      (QShimUDmaHandle*)align_up_ptr(vtcmPtr, kAlign);

  // Weight is the same for all rows; load it once before the main loop
  if (localThreadID == 0) {
    *status_vtcm = dma_copy_wait(threadID, weight_vtcm, weight_ddr, rowBytes);
  }
  sync_hvx_threads(threadID, numThreads);
  if (*status_vtcm != JIT_DEV_STATUS_SUCCESS) {
    return *status_vtcm;
  }

  // Each core processes ceil(M / numCores) rows
  const int rowIters = (M + (int)numCores - 1) / (int)numCores;

  // Prime slot 0 with the first row assigned to this core before entering the
  // loop
  {
    const int m0 = (int)coreID;
    const bool valid0 = (m0 < M);
    if (localThreadID == 0) {
      if (valid0) {
        *status_vtcm = dma_copy_wait(threadID, attn_vtcm[0],
                                     attn_out_ddr + m0 * N, rowBytes);
        if (*status_vtcm == JIT_DEV_STATUS_SUCCESS) {
          *status_vtcm =
              dma_copy_wait(threadID, x_vtcm[0], x_ddr + m0 * N, rowBytes);
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
    const int m = iter * (int)numCores + (int)coreID;
    const int m_next = (iter + 1) * (int)numCores + (int)coreID;
    const bool validRow = (m < M);
    const bool validNextRow = (m_next < M);
    // Ping-pong between slot 0 and 1 each iteration
    const int cur_slot = iter & 1;
    const int next_slot = cur_slot ^ 1;

    float16* row_dst_ddr = validRow ? (dst_ddr + m * N) : dst_ddr;
    float16* row_out_ddr = validRow ? (out_ddr + m * N) : out_ddr;

    // Thread 0 submits async DMA for the next row while all threads compute the
    // current row. attn prefetch is waited immediately (sequential); x prefetch
    // is left in-flight and waited after compute completes to maximally overlap
    // with rms_norm execution.
    if (localThreadID == 0) {
      if (validNextRow) {
        uint32_t dma_status = JIT_DEV_STATUS_SUCCESS;
        QShimUDmaHandle h_attn =
            dma_copy_submit(threadID, attn_vtcm[next_slot],
                            attn_out_ddr + m_next * N, rowBytes, &dma_status);
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
                dma_copy_submit(threadID, x_vtcm[next_slot], x_ddr + m_next * N,
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
      _single_nsp_rms_norm(attn_vtcm[cur_slot], x_vtcm[cur_slot], weight_vtcm,
                           out_vtcm, dst_vtcm, epsilon, N, threadID,
                           localThreadID, numThreads, partial_sums);
    } else {
      // Idle row: rms_norm contains 3 sync barriers; mirror them to keep all
      // threads in lockstep
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