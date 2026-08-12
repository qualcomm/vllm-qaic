// ---------------------------------------------------------------------------------------
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
// ---------------------------------------------------------------------------------------

// Fused block-scaled FP8 GEMM (HVX only) for small M ("decode-shaped" GEMMs).
//
// Computes, with per-token x per-K-block activation scales and a 2-D
// [block_n, block_k] weight scale grid:
//
//   out[m,n] = sum_kb As[m,kb] * Bs[n/bn,kb] * ( sum_{k in block kb} A[m,k]*B[n,k] )
//
// The scale is constant along K *within* a block, so it hoists out of the inner
// sum. Each block's raw fp8xfp8 dot product is therefore formed first and scaled
// once, and the scaled block contributions are accumulated in IEEE fp32. That
// makes the result exact with respect to the fp8 operands: none of the
// power-of-two folding / output-correction machinery the fp16-accumulator
// PyTorch path needs (see vllm_qaic/quantization/qaic_fp8_block_scaled_mm.py)
// appears here.
//
// fp32 accumulation is required, not a nicety: a block_k=128 run of fp8xfp8
// products reaches 128*448*448 ~= 2.6e7, far past fp16's 65504 ceiling.
//
// Why HVX and not HMX: out-of-tree csrc kernels only see the SDK JIT headers,
// which do not include QAicHexagonHMX.h, and link only libgcc/libc -- the HMX
// matmul and its crouton layout converters are unreachable from here. That is
// acceptable precisely because this kernel targets small M, where the GEMM is
// LPDDR-bandwidth-bound rather than compute-bound: consuming the fp8 weight
// directly avoids materializing an fp16 copy of it, which is the whole win.
//
// Parallelization: N is the parallel axis (each output column is independent).
// N is split across cores, then across the HVX threads within a core. Because B
// is [N, K] row-major contiguous, a core's row range is one contiguous LPDDR
// block, so a batch of B rows moves in a single large linear DMA, double
// buffered against compute.
//
// Pointer array:
//   [0]  A            fp8  [M, K]              contiguous
//   [1]  B            fp8  [N, K]              contiguous
//   [2]  As           fp32 [M, kBlocks]
//   [3]  Bs           fp32 [nBlocks, kBlocks]
//   [4]  out          fp32 [M, N]
//   [5]  M            int32
//   [6]  N            int32
//   [7]  K            int32
//   [8]  block_n      int32
//   [9]  block_k      int32
//   [10] fp8_dtype_id int32   (0 = e4m3fn, 1 = e4m3fnuz)

#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <stdint.h>
#include <string.h> // must precede the memcpy redirect below

// The extension is built with -ffreestanding, which in clang implies
// -fno-builtin: a plain memcpy is emitted as a real libc call and is never
// folded, even for a 2-byte copy of a compile-time constant. That matters here
// because HVX_VectorTyped's splat constructor routes its scalar through memcpy,
// and each SDK fp8 decoder performs ~20 constant splats -- left out of line
// those calls dominate this kernel's inner loop (measured: 19-21 memcpy
// relocations per decoder, plus the decoders themselves failing to inline).
// __builtin_memcpy is available regardless of -fno-builtin and folds to a
// register move. The redirect only has to be live while the constructor bodies
// are parsed, which is here, so it is scoped to these includes.
#define memcpy __builtin_memcpy
#include "QAicHexagonFP8.h"
#include "QAicHexagonPlatformIntf.h"
#include "QAicHexagonTypes.h"
#include "QAicHexagonUtils.h"
#include "jit_dev_exe_function.h"
#include "jit_dev_status_codes.h"
#include "jit_qshim_api.h"
#undef memcpy

#define QAIC_FP8_DTYPE_E4M3FN 0
#define QAIC_FP8_DTYPE_E4M3FNUZ 1

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
    return (status != JIT_DEV_STATUS_SUCCESS) ? status
                                              : JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  return qshimUDmaWait(h);
}

static inline QShimUDmaHandle dma_copy_submit(uint32_t threadId, void *dst,
                                             const void *src, uint32_t bytes,
                                             uint32_t *status,
                                             uint32_t order = 0) {
  return qaic_linear_udma_submit(threadId, (AicJitPtr)src, bytes,
                                 (AicJitPtr)dst, order, true, status);
}

// Number of fp8 elements decoded per HVX vector, and the fp16 half-vector width.
static const int kElemsPerFp8Vec = HVX_VectorSize;              // 128
static const int kFp16PerVec = HVX_VectorSize / (int)sizeof(__fp16); // 64

// Horizontal sum of the 32 fp32 lanes of an IEEE-fp32 HVX vector.
// Log-tree over rotates: 16, 8, 4, 2, 1 lanes.
static inline float hvx_hsum_sf(HVX_Vector v) {
  const int es = (int)sizeof(float);
  HVX_Vector s = v;
  s = Q6_Vsf_vadd_VsfVsf(s, Q6_V_vror_VR(s, 16 * es));
  s = Q6_Vsf_vadd_VsfVsf(s, Q6_V_vror_VR(s, 8 * es));
  s = Q6_Vsf_vadd_VsfVsf(s, Q6_V_vror_VR(s, 4 * es));
  s = Q6_Vsf_vadd_VsfVsf(s, Q6_V_vror_VR(s, 2 * es));
  s = Q6_Vsf_vadd_VsfVsf(s, Q6_V_vror_VR(s, 1 * es));
  HVX_VectorSF t = s;
  return t[0];
}

// Decode one HVX vector (128 values) of fp8 to a pair of fp16 vectors.
template <int DTYPE> static inline HVX_VectorWHF decode_fp8(HVX_Vector v) {
  if constexpr (DTYPE == QAIC_FP8_DTYPE_E4M3FNUZ) {
    return qaic_convert_fp8_e4m3uz_to_hf(v);
  } else {
    return qaic_convert_fp8_e4m3fn_to_hf(v);
  }
}

// Lane-wise fp32 products of one 128-element chunk, reduced to a single fp32
// vector of 32 partial sums. The caller horizontally reduces once per (m,n),
// not once per chunk, which is exact because scaling and accumulating vectors
// lane-wise commutes with the final horizontal sum.
//
// Note the decoder's internal lane permutation is irrelevant here: A and B go
// through the identical decoder, and a dot product is permutation-invariant, so
// pairing A's lo half with B's lo half (and hi with hi) is sufficient.
static inline HVX_Vector chunk_partials_sf(HVX_Vector a_lo, HVX_Vector a_hi,
                                           HVX_Vector b_lo, HVX_Vector b_hi) {
  HVX_VectorPair p_lo = Q6_Wqf32_vmpy_VhfVhf(a_lo, b_lo);
  HVX_VectorPair p_hi = Q6_Wqf32_vmpy_VhfVhf(a_hi, b_hi);
  HVX_Vector s0 =
      Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p_lo), Q6_V_hi_W(p_lo));
  HVX_Vector s1 =
      Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p_hi), Q6_V_hi_W(p_hi));
  return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(s0, s1));
}

// Decode A ([M, K] fp8) into a_hf ([M, K] fp16) once, cooperatively across the
// core's threads. Hoisting this out of the n loop is what keeps the inner loop
// to a plain fp16 multiply; the decode would otherwise be repeated for every
// output column. The decode is exact (every fp8 value is representable in
// fp16), so no range normalisation is needed.
template <int DTYPE>
static void decode_a(const uint8_t *a_fp8, __fp16 *a_hf, int totalChunks,
                     uint32_t localThreadID, uint32_t threadsPerCore) {
  for (int c = (int)localThreadID; c < totalChunks; c += (int)threadsPerCore) {
    HVX_Vector v = *(const HVX_Vector *)(a_fp8 + (size_t)c * kElemsPerFp8Vec);
    HVX_VectorWHF hf = decode_fp8<DTYPE>(v);
    __fp16 *dst = a_hf + (size_t)c * kElemsPerFp8Vec;
    *(HVX_Vector *)(dst) = Q6_V_lo_W(hf.rawVectorPair);
    *(HVX_Vector *)(dst + kFp16PerVec) = Q6_V_hi_W(hf.rawVectorPair);
  }
}

// Rows of A held in HVX accumulators at once, and so the top of the tile ladder
// below. A taller tile amortizes the B decode over more m rows, but 4 is the
// largest height whose accumulators still fit in registers alongside the fp8
// decoder's temporaries and hoisted constants: at 8 and at 16 the accumulator
// array is spilled and the multiply loop regains a vector store (verified in the
// disassembly, `vmem(...) = v` inside the MAC loop, at both heights).
//
// This bound comes from register pressure, not from measured throughput --
// nothing here has been timed on device, so the height is worth re-tuning once
// it can be.
static const int kMTile = 4;
static_assert(kMTile >= 1 && (kMTile & (kMTile - 1)) == 0,
              "the halving ladder in compute_ladder needs a power-of-two tile");

// Accumulate MTILE rows of A against one output column, and write their fp32
// results.
//
// MTILE is a template parameter rather than a runtime bound for a specific
// codegen reason: acc[] is indexed by the innermost loop variable, and only a
// compile-time trip count lets clang fully unroll that loop and keep the
// accumulators in HVX registers. With a runtime bound the array is spilled to
// the stack and every multiply-accumulate pays an extra vector load and store --
// which is what the first version of this kernel did (a post-incremented
// vmem load/store pair sat in the innermost hardware loop, visible in the
// disassembly, alongside a ~2.3 KB stack frame).
template <int DTYPE, int MTILE>
static void compute_mtile(const uint8_t *b_row, const __fp16 *a_hf,
                          const float *as, const float *bs_row, float *out_vtcm,
                          int m0, int k, int kBlocks, int vecsPerBlock,
                          int outStride, int r) {
  HVX_Vector acc[MTILE];
  for (int j = 0; j < MTILE; ++j) {
    acc[j] = HVX_VectorSF(0.0f);
  }

  for (int kb = 0; kb < kBlocks; ++kb) {
    const float sb = bs_row[kb];
    for (int v = 0; v < vecsPerBlock; ++v) {
      const int chunk = kb * vecsPerBlock + v;
      HVX_Vector bvec =
          *(const HVX_Vector *)(b_row + (size_t)chunk * kElemsPerFp8Vec);
      HVX_VectorWHF bhf = decode_fp8<DTYPE>(bvec);
      // Decoded once per (n, chunk) and reused across the whole m tile.
      const HVX_Vector b_lo = Q6_V_lo_W(bhf.rawVectorPair);
      const HVX_Vector b_hi = Q6_V_hi_W(bhf.rawVectorPair);

      for (int j = 0; j < MTILE; ++j) {
        const __fp16 *arow =
            a_hf + (size_t)(m0 + j) * k + (size_t)chunk * kElemsPerFp8Vec;
        HVX_Vector a_lo = *(const HVX_Vector *)(arow);
        HVX_Vector a_hi = *(const HVX_Vector *)(arow + kFp16PerVec);

        HVX_Vector partials = chunk_partials_sf(a_lo, a_hi, b_lo, b_hi);
        const float s = as[(size_t)(m0 + j) * kBlocks + kb] * sb;
        HVX_Vector scaled = Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vmpy_VsfVsf(partials, HVX_VectorSF(s)));
        acc[j] = Q6_Vsf_vadd_VsfVsf(acc[j], scaled);
      }
    }
  }

  for (int j = 0; j < MTILE; ++j) {
    out_vtcm[(size_t)(m0 + j) * outStride + r] = hvx_hsum_sf(acc[j]);
  }
}

// Drain the m rows through a halving ladder of tile heights: as many MTILE-row
// passes as fit, then MTILE/2, down to 1. The ladder exists so that an m which
// is not a multiple of kMTile still amortizes the B decode over several rows --
// M=7 costs three passes over B (4+2+1), not seven.
template <int DTYPE, int MTILE>
static void compute_ladder(const uint8_t *b_row, const __fp16 *a_hf,
                           const float *as, const float *bs_row, float *out_vtcm,
                           int &m0, int m, int k, int kBlocks, int vecsPerBlock,
                           int outStride, int r) {
  while (m - m0 >= MTILE) {
    compute_mtile<DTYPE, MTILE>(b_row, a_hf, as, bs_row, out_vtcm, m0, k,
                                kBlocks, vecsPerBlock, outStride, r);
    m0 += MTILE;
  }
  if constexpr (MTILE > 1) {
    compute_ladder<DTYPE, MTILE / 2>(b_row, a_hf, as, bs_row, out_vtcm, m0, m, k,
                                     kBlocks, vecsPerBlock, outStride, r);
  }
}

// Compute out_vtcm[m, r] for the current batch of B rows.
// b_vtcm is [rows, K] fp8; a_hf is [M, K] fp16; out_vtcm is [M, outStride] fp32.
template <int DTYPE>
static void compute_batch(const uint8_t *b_vtcm, const __fp16 *a_hf,
                          const float *as, const float *bs, float *out_vtcm,
                          int m, int k, int kBlocks, int blockK, int blockN,
                          int nStart, int rows, int outStride,
                          uint32_t localThreadID, uint32_t threadsPerCore) {
  const int vecsPerBlock = blockK / kElemsPerFp8Vec;

  // One division at entry rather than one per output column. Hexagon has no
  // integer-divide instruction, so n / blockN compiles to a libgcc call; n only
  // ever increases here, so the scale-block index can just walk forward.
  int nsBlock = (nStart + (int)localThreadID) / blockN;
  int nsLimit = (nsBlock + 1) * blockN;

  for (int r = (int)localThreadID; r < rows; r += (int)threadsPerCore) {
    const int n = nStart + r;
    // Ragged N needs no special case: the tail rows land in the final scale
    // block naturally, exactly as n / blockN would place them.
    while (n >= nsLimit) {
      ++nsBlock;
      nsLimit += blockN;
    }
    const float *bs_row = bs + (size_t)nsBlock * kBlocks;
    const uint8_t *b_row = b_vtcm + (size_t)r * k;

    int m0 = 0;
    compute_ladder<DTYPE, kMTile>(b_row, a_hf, as, bs_row, out_vtcm, m0, m, k,
                                  kBlocks, vecsPerBlock, outStride, r);
  }
}

// Per-core driver: owns a contiguous stripe of N, streams B row batches through
// a VTCM ping/pong, and writes fp32 results back per batch.
template <int DTYPE>
static uint32_t
run(const uint8_t *a_ddr, const uint8_t *b_ddr, const float *as_ddr,
    const float *bs_ddr, float *out_ddr, int m, int n, int k, int blockN,
    int blockK, uint32_t threadID, uint32_t localThreadID, uint32_t numThreads,
    uint32_t numCores, uint32_t coreID) {
  const int kBlocks = k / blockK;
  const int nBlocks = (n + blockN - 1) / blockN;

  // This core's contiguous stripe of output columns.
  const int nPerCore = (n + (int)numCores - 1) / (int)numCores;
  const int nCoreStart = (int)coreID * nPerCore;
  if (nCoreStart >= n) {
    // Depends only on coreID, so every thread of this core agrees and no
    // barrier is skipped asymmetrically.
    return JIT_DEV_STATUS_SUCCESS;
  }
  const int nCoreEnd =
      (nCoreStart + nPerCore < n) ? (nCoreStart + nPerCore) : n;
  const int nCoreRows = nCoreEnd - nCoreStart;

  constexpr uint32_t kAlign = 128;

  // Resident VTCM: A as fp8 (staging, dead after the decode), A as fp16, and
  // both scale grids (which are tiny: [M, kBlocks] and [nBlocks, kBlocks]).
  const uint32_t aFp8Bytes = align_up_u32((uint32_t)m * (uint32_t)k, kAlign);
  const uint32_t aHfBytes =
      align_up_u32((uint32_t)m * (uint32_t)k * (uint32_t)sizeof(__fp16), kAlign);
  const uint32_t asBytes = align_up_u32(
      (uint32_t)m * (uint32_t)kBlocks * (uint32_t)sizeof(float), kAlign);
  const uint32_t bsBytes = align_up_u32(
      (uint32_t)nBlocks * (uint32_t)kBlocks * (uint32_t)sizeof(float), kAlign);
  const uint32_t statusBytes = align_up_u32(sizeof(uint32_t), kAlign);

  int64_t vtcmSize = 0;
  uint32_t ret = qshimQuery(DEV_ATTR_QSHIM_VTCM_SIZE, &vtcmSize);
  if (ret != JIT_DEV_STATUS_SUCCESS) {
    return ret;
  }

  // Leave room for the initial alignment plus each buffer's own alignment slack.
  const uint32_t fixedBytes =
      aFp8Bytes + aHfBytes + asBytes + bsBytes + statusBytes + 8 * kAlign;
  if ((uint64_t)fixedBytes >= (uint64_t)vtcmSize) {
    // Residents alone do not fit; the Python path handles this shape.
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  // Size the B row batch from what is left: two B slots plus one fp32 output
  // column per row.
  const uint32_t perRowBytes =
      2u * (uint32_t)k + (uint32_t)m * (uint32_t)sizeof(float);
  int batchRows = (int)(((uint64_t)vtcmSize - (uint64_t)fixedBytes) /
                        (uint64_t)perRowBytes);
  if (batchRows < 1) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }
  if (batchRows > nCoreRows) {
    batchRows = nCoreRows;
  }

  const uint32_t bSlotBytes = align_up_u32((uint32_t)batchRows * (uint32_t)k,
                                           kAlign);
  const uint32_t outBytes = align_up_u32(
      (uint32_t)m * (uint32_t)batchRows * (uint32_t)sizeof(float), kAlign);

  uint8_t *vtcmPtr = align_up_ptr(qshimGetBaseVtcmAddr(), kAlign);

  uint8_t *a_fp8_vtcm = vtcmPtr;
  vtcmPtr += aFp8Bytes;
  __fp16 *a_hf_vtcm = (__fp16 *)vtcmPtr;
  vtcmPtr += aHfBytes;
  float *as_vtcm = (float *)vtcmPtr;
  vtcmPtr += asBytes;
  float *bs_vtcm = (float *)vtcmPtr;
  vtcmPtr += bsBytes;
  uint8_t *b_vtcm[2];
  b_vtcm[0] = vtcmPtr;
  vtcmPtr += bSlotBytes;
  b_vtcm[1] = vtcmPtr;
  vtcmPtr += bSlotBytes;
  float *out_vtcm = (float *)vtcmPtr;
  vtcmPtr += outBytes;
  uint32_t *status_vtcm = (uint32_t *)align_up_ptr(vtcmPtr, kAlign);

  // Load A and both scale grids, then decode A to fp16 once.
  if (localThreadID == 0) {
    *status_vtcm = dma_copy_wait(threadID, a_fp8_vtcm, a_ddr,
                                 (uint32_t)m * (uint32_t)k);
    if (*status_vtcm == JIT_DEV_STATUS_SUCCESS) {
      *status_vtcm = dma_copy_wait(threadID, as_vtcm, as_ddr,
                                   (uint32_t)m * (uint32_t)kBlocks *
                                       (uint32_t)sizeof(float));
    }
    if (*status_vtcm == JIT_DEV_STATUS_SUCCESS) {
      *status_vtcm = dma_copy_wait(threadID, bs_vtcm, bs_ddr,
                                   (uint32_t)nBlocks * (uint32_t)kBlocks *
                                       (uint32_t)sizeof(float));
    }
  }
  sync_hvx_threads(threadID, numThreads);
  if (*status_vtcm != JIT_DEV_STATUS_SUCCESS) {
    return *status_vtcm;
  }

  decode_a<DTYPE>(a_fp8_vtcm, a_hf_vtcm, (m * k) / kElemsPerFp8Vec,
                  localThreadID, numThreads);
  sync_hvx_threads(threadID, numThreads);

  const int numBatches = (nCoreRows + batchRows - 1) / batchRows;

  // Prime slot 0.
  if (localThreadID == 0) {
    const int rows0 =
        (batchRows < nCoreRows) ? batchRows : nCoreRows;
    *status_vtcm = dma_copy_wait(threadID, b_vtcm[0],
                                 b_ddr + (size_t)nCoreStart * k,
                                 (uint32_t)rows0 * (uint32_t)k);
  }
  sync_hvx_threads(threadID, numThreads);
  if (*status_vtcm != JIT_DEV_STATUS_SUCCESS) {
    return *status_vtcm;
  }

  for (int batch = 0; batch < numBatches; ++batch) {
    const int cur = batch & 1;
    const int next = cur ^ 1;
    const int nStart = nCoreStart + batch * batchRows;
    const int remaining = nCoreEnd - nStart;
    const int rows = (batchRows < remaining) ? batchRows : remaining;

    // Thread 0 kicks off the next batch's B DMA and leaves it in flight while
    // every thread (including 0) computes the current batch.
    QShimUDmaHandle inflight = INVALID_UDMA_HANDLE;
    if (localThreadID == 0 && batch + 1 < numBatches) {
      const int nStartNext = nStart + batchRows;
      const int remainingNext = nCoreEnd - nStartNext;
      const int rowsNext =
          (batchRows < remainingNext) ? batchRows : remainingNext;
      uint32_t dma_status = JIT_DEV_STATUS_SUCCESS;
      inflight = dma_copy_submit(threadID, b_vtcm[next],
                                 b_ddr + (size_t)nStartNext * k,
                                 (uint32_t)rowsNext * (uint32_t)k, &dma_status);
      if (dma_status != JIT_DEV_STATUS_SUCCESS) {
        *status_vtcm = dma_status;
        inflight = INVALID_UDMA_HANDLE;
      }
    }

    compute_batch<DTYPE>(b_vtcm[cur], a_hf_vtcm, as_vtcm, bs_vtcm, out_vtcm, m,
                         k, kBlocks, blockK, blockN, nStart, rows, batchRows,
                         localThreadID, numThreads);

    // All threads must be done writing out_vtcm before thread 0 ships it.
    sync_hvx_threads(threadID, numThreads);

    if (localThreadID == 0) {
      if (inflight != INVALID_UDMA_HANDLE) {
        uint32_t wait_st = qshimUDmaWait(inflight);
        if (wait_st != JIT_DEV_STATUS_SUCCESS) {
          *status_vtcm = wait_st;
        }
      }
      // out is [M, N]; this batch owns a contiguous run of columns per row.
      for (int mi = 0; mi < m && *status_vtcm == JIT_DEV_STATUS_SUCCESS; ++mi) {
        *status_vtcm = dma_copy_wait(
            threadID, out_ddr + (size_t)mi * n + nStart,
            out_vtcm + (size_t)mi * batchRows, (uint32_t)rows * sizeof(float));
      }
    }

    sync_hvx_threads(threadID, numThreads);
    if (*status_vtcm != JIT_DEV_STATUS_SUCCESS) {
      return *status_vtcm;
    }
  }

  return JIT_DEV_STATUS_SUCCESS;
}

QAIC_KERNEL_API int32_t multinsp_multithreaded_fp8_block_scaled_mm(
    const AicJitEntryPointConfig *entryConfig,
    const AicJitPointerArray *pointerArray) {
  const uint8_t *a_ddr = (const uint8_t *)pointerArray->pointers[0];
  const uint8_t *b_ddr = (const uint8_t *)pointerArray->pointers[1];
  const float *as_ddr = (const float *)pointerArray->pointers[2];
  const float *bs_ddr = (const float *)pointerArray->pointers[3];
  float *out_ddr = (float *)pointerArray->pointers[4];
  const int m = *(const int32_t *)pointerArray->pointers[5];
  const int n = *(const int32_t *)pointerArray->pointers[6];
  const int k = *(const int32_t *)pointerArray->pointers[7];
  const int blockN = *(const int32_t *)pointerArray->pointers[8];
  const int blockK = *(const int32_t *)pointerArray->pointers[9];
  const int dtypeId = *(const int32_t *)pointerArray->pointers[10];

  if (m <= 0 || n <= 0 || k <= 0 || blockN <= 0 || blockK <= 0) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }
  // One K block must be a whole number of HVX vectors, and K a whole number of
  // K blocks, so the inner loop never needs cross-block masking.
  if ((blockK % kElemsPerFp8Vec) != 0 || (k % blockK) != 0) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  const uint32_t threadID = entryConfig->threadID;
  const uint32_t numThreads = entryConfig->numThreads;
  const uint32_t localThreadID = threadID % numThreads;

  if (dtypeId == QAIC_FP8_DTYPE_E4M3FNUZ) {
    return (int32_t)run<QAIC_FP8_DTYPE_E4M3FNUZ>(
        a_ddr, b_ddr, as_ddr, bs_ddr, out_ddr, m, n, k, blockN, blockK, threadID,
        localThreadID, numThreads, entryConfig->numCores, entryConfig->coreID);
  }
  if (dtypeId == QAIC_FP8_DTYPE_E4M3FN) {
    return (int32_t)run<QAIC_FP8_DTYPE_E4M3FN>(
        a_ddr, b_ddr, as_ddr, bs_ddr, out_ddr, m, n, k, blockN, blockK, threadID,
        localThreadID, numThreads, entryConfig->numCores, entryConfig->coreID);
  }
  return JIT_DEV_ERROR_INVALID_PARAMETER;
}
