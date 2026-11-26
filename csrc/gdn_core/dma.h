// ---------------------------------------------------------------------------------------
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
// ---------------------------------------------------------------------------------------

#ifndef MATMUL_CUSTOM_OP_DMA_H
#define MATMUL_CUSTOM_OP_DMA_H

#include "jit_qshim_api.h"

namespace custom_matmul {
#ifdef _DEBUG
static inline int32_t check2DDMAAttributes(uint32_t height, uint32_t width,
                                           uint32_t destStride,
                                           uint32_t srcStride) {
  if (!(height > 0 && height <= UINT16_MAX))
    qshimLog(LOG_MASK_FATAL, 0,
             "QAIC_ML_FW_ERROR: 2DUdma Failure, Invalid Height: %d \n", height);
  if (!(width > 0 && width <= UINT16_MAX))
    qshimLog(LOG_MASK_FATAL, 0,
             "QAIC_ML_FW_ERROR: 2DUdma Failure, Invalid Width: %d \n", width);
  if (!(destStride > 0 && destStride <= UINT16_MAX))
    qshimLog(LOG_MASK_FATAL, 0,
             "QAIC_ML_FW_ERROR: 2DUdma Failure, Invalid Dest Stride: %d \n",
             destStride);
  if (!(srcStride > 0 && srcStride <= UINT16_MAX))
    qshimLog(LOG_MASK_FATAL, 0,
             "QAIC_ML_FW_ERROR: 2DUdma Failure, Invalid Src Stride: %d \n",
             srcStride);

  return JIT_DEV_STATUS_SUCCESS;
}
#endif  //_DEBUG

static QShimUDmaHandle qaic_2D_udma_submit(uint32_t threadId, void* srcPtr,
                                           void* dstPtr, uint32_t height,
                                           uint32_t width, uint32_t destStride,
                                           uint32_t srcStride,
                                           uint32_t udmaDescAttrsOrder) {
  QShimUDmaHandle dmaHandle;
  AicJit2DUdmaDescAttrs dmaAttrs;
  AicJitUdmaDescCommonAttrs udmaDescAttrs;

  if ((destStride <= UINT16_MAX) && (srcStride <= UINT16_MAX) &&
      (height <= UINT16_MAX)) {
#ifdef _DEBUG
    check2DDMAAttributes(height, width, destStride, srcStride);
#endif  //_DEBUG

    // Initialize UDMA Descriptor
    udmaDescAttrs.order = udmaDescAttrsOrder;
    udmaDescAttrs.bypassOverride = 0;
    dmaAttrs.height = height;
    dmaAttrs.width = width;
    dmaAttrs.destStride = destStride;
    dmaAttrs.srcStride = srcStride;
    dmaAttrs.udmaDescCommonAttrs = &udmaDescAttrs;
    uint32_t status = JIT_DEV_STATUS_SUCCESS;
    dmaHandle = qshim2DUdmaSubmit(threadId, (AicJitPtr)srcPtr,
                                  (AicJitPtr)dstPtr, &dmaAttrs, true, &status);
  } else {
    if ((destStride > UINT16_MAX) || (srcStride > UINT16_MAX)) {
      for (uint32_t i = 0; i < height; i++) {
        char* dstPtr1 = (char*)dstPtr + (i * destStride);
        char* srcPtr1 = (char*)srcPtr + (i * srcStride);
        for (uint32_t j = 0; j < width; j += UINT16_MAX) {
          bool lastIteration = false;
          if ((i == (height - 1)) && ((j + UINT16_MAX) >= width)) {
            lastIteration = true;
          }
          char* dstPtr2 = dstPtr1 + j;
          char* srcPtr2 = srcPtr1 + j;
          uint32_t numElements =
              ((j + UINT16_MAX) < width ? UINT16_MAX : (width - j));
          // Initiate UDMA
          if (!(lastIteration)) {
            udmaDescAttrs.order = 0;
            udmaDescAttrs.bypassOverride = 0;
            uint32_t status = JIT_DEV_STATUS_SUCCESS;
            // TODO: Analyse performance degradation issues when requireHandle
            // param is enabled.
            qshimLinearUdmaSubmit(threadId, (AicJitPtr)srcPtr2, numElements,
                                  (AicJitPtr)dstPtr2, &udmaDescAttrs, false,
                                  &status);
          } else {
            udmaDescAttrs.order = 1;
            udmaDescAttrs.bypassOverride = 0;
            uint32_t status = JIT_DEV_STATUS_SUCCESS;
            dmaHandle = qshimLinearUdmaSubmit(threadId, (AicJitPtr)srcPtr2,
                                              numElements, (AicJitPtr)dstPtr2,
                                              &udmaDescAttrs, true, &status);
          }
        }
      }
    } else {
      for (uint32_t i = 0; i < height; i += UINT16_MAX) {
        char* dstPtr1 = (char*)dstPtr + (i * destStride);
        char* srcPtr1 = (char*)srcPtr + (i * srcStride);
        int32_t numRowsPerIter =
            ((i + UINT16_MAX) < height ? UINT16_MAX : (height - i));

#ifdef _DEBUG
        check2DDMAAttributes(numRowsPerIter, width, destStride, srcStride);
#endif  //_DEBUG

        // Initiate UDMA
        dmaAttrs.height = numRowsPerIter;
        dmaAttrs.width = width;
        dmaAttrs.destStride = destStride;
        dmaAttrs.srcStride = srcStride;

        if ((i + UINT16_MAX) < height) {
          udmaDescAttrs.order = 0;
          udmaDescAttrs.bypassOverride = 0;
          dmaAttrs.udmaDescCommonAttrs = &udmaDescAttrs;
          uint32_t status = JIT_DEV_STATUS_SUCCESS;
          qshim2DUdmaSubmit(threadId, (AicJitPtr)srcPtr1, (AicJitPtr)dstPtr1,
                            &dmaAttrs, false, &status);
        } else {
          udmaDescAttrs.order = 1;
          udmaDescAttrs.bypassOverride = 0;
          dmaAttrs.udmaDescCommonAttrs = &udmaDescAttrs;
          uint32_t status = JIT_DEV_STATUS_SUCCESS;
          dmaHandle =
              qshim2DUdmaSubmit(threadId, (AicJitPtr)srcPtr1,
                                (AicJitPtr)dstPtr1, &dmaAttrs, true, &status);
        }
      }
    }
  }
  return (dmaHandle);
}

static QShimUDmaHandle qaic_linear_udma_submit(uint32_t threadId, AicJitPtr src,
                                               uint32_t size, AicJitPtr dst,
                                               uint32_t udmaDescAttrsOrder,
                                               bool requireHandle,
                                               uint32_t* status) {
  AicJitUdmaDescCommonAttrs udmaDescAttrs;
  udmaDescAttrs.order = udmaDescAttrsOrder;
  udmaDescAttrs.bypassOverride = 0;
  return qshimLinearUdmaSubmit(threadId, src, size, dst, &udmaDescAttrs,
                               requireHandle, status);
}
}  // namespace custom_matmul

#endif  // MATMUL_CUSTOM_OP_DMA_H
