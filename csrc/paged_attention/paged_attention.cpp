// ---------------------------------------------------------------------------------------
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
// ---------------------------------------------------------------------------------------

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "hexagon_types.h"
#include "hexagon_protos.h"
#include "QAicHexagonHMX.h"
#include "QAicHexagonHMXDefs.h"
#include "QAicHexagonHVX.h"
#include "QAicHexagonPlatformIntf.h"
#include "QAicHexagonActivations.h"
#include "QAicHexagonReducer.h"
#include "QAicHexagonUtils.h"
#include "QAicHexagonTypes.h"
#include "jit_dev_status_codes.h"
#include "jit_qshim_api.h"

namespace {

constexpr int32_t kF16PerHvx = HVX_VectorSize / sizeof(float16);
constexpr int32_t kF32PerHvx = HVX_VectorSize / sizeof(float);
constexpr int32_t kMaxHeadDim = 256;
// K and V gathers are pipelined concurrently. Keep each buffer to half of the
// 256-entry UDMA queue so the pair cannot exhaust the device descriptors.
constexpr int32_t kMaxDmaHandles = 128;
constexpr uint64_t kHmxVtcmSafetyBytes = 4096;

static inline bool is_core_master_hvx_thread(uint32_t thread_id,
                                             int64_t hmx_thread_id) {
  return thread_id != (uint32_t)hmx_thread_id && thread_id == 0;
}

static inline uint64_t align_up_u64(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

static inline void* vtcm_alloc(uint8_t** cur, uint64_t alignment,
                               uint64_t bytes) {
  uint64_t ptr = align_up_u64((uint64_t)*cur, alignment);
  *cur = (uint8_t*)(ptr + bytes);
  return (void*)ptr;
}

static inline QShimUDmaHandle qaic_linear_udma_submit_local(
    uint32_t thread_id, AicJitPtr src, uint32_t size, AicJitPtr dst,
    uint32_t order, bool require_handle, uint32_t* status) {
  AicJitUdmaDescCommonAttrs attrs = {};
  attrs.order = order;
  return qshimLinearUdmaSubmit(thread_id, src, size, dst, &attrs,
                               require_handle, status);
}

static inline QShimUDmaHandle dma_copy_submit_local(
    uint32_t thread_id, void* dst, const void* src, uint32_t bytes,
    uint32_t* status) {
  return qaic_linear_udma_submit_local(thread_id, (AicJitPtr)src, bytes,
                                       (AicJitPtr)dst, 0, true, status);
}

static inline QShimUDmaHandle dma_copy_2d_submit_local(
    uint32_t thread_id, void* dst, const void* src, uint32_t height,
    uint32_t width, uint32_t dst_stride, uint32_t src_stride,
    uint32_t* status) {
  AicJitUdmaDescCommonAttrs common = {};
  common.order = 0;
  common.bypassOverride = 0;
  AicJit2DUdmaDescAttrs attrs = {};
  attrs.height = height;
  attrs.width = width;
  attrs.destStride = dst_stride;
  attrs.srcStride = src_stride;
  attrs.udmaDescCommonAttrs = &common;
  return qshim2DUdmaSubmit(thread_id, (AicJitPtr)src, (AicJitPtr)dst, &attrs,
                           true, status);
}

static inline int32_t wait_dma_handles(QShimUDmaHandle* handles,
                                       int32_t n_handles) {
  for (int32_t i = 0; i < n_handles; ++i) {
    if (handles[i] == INVALID_UDMA_HANDLE) {
      return JIT_DEV_ERROR_INVALID_PARAMETER;
    }
    const uint32_t ret = qshimUDmaWait(handles[i]);
    if (ret != JIT_DEV_STATUS_SUCCESS) {
      return ret;
    }
  }
  return JIT_DEV_STATUS_SUCCESS;
}

static inline uint64_t hmx_lhs_crouton_bytes(int32_t m, int32_t k) {
  int shape[2] = {m, k};
  const QAicHMXDims<QAIC_HMX_FlatNXYD> flat_dims(
      shape, sizeof(float16), HMXShapeKind::YD);
  const QAicHMXDims<QAIC_HMX_MATMUL_CROUTON_16B> crouton_dims(flat_dims);
  return crouton_dims.sizeInBytes();
}

static inline uint64_t hmx_out_crouton_bytes(int32_t m, int32_t n) {
  int shape[2] = {m, n};
  const QAicHMXDims<QAIC_HMX_FlatNXYD> flat_dims(
      shape, sizeof(float16), HMXShapeKind::YD);
  const QAicHMXDims<QAIC_HMX_MATMUL_CROUTON_16B> crouton_dims(flat_dims);
  return crouton_dims.sizeInBytes();
}

struct HmxPagedAttentionScratch {
  float16* q_rm;
  float16* k_rhs_rm[2];
  float16* s_rm;
  float16* p_rm;
  float16* v_rhs_rm[2];
  float16* pv_rm;
  float16* q_lhs_crouton;
  float16* lhs_crouton;
  float16* rhs_crouton;
  float16* out_crouton;
  float* out_acc;
  float* row_max;
  float* row_sum;
  int32_t* hmx_status;
};

struct HmxPagedAttentionLayout {
  uint64_t q_bytes;
  uint64_t k_rhs_bytes;
  uint64_t s_bytes;
  uint64_t p_bytes;
  uint64_t v_rhs_bytes;
  uint64_t pv_bytes;
  uint64_t q_lhs_crouton_bytes;
  uint64_t lhs_crouton_bytes;
  uint64_t rhs_crouton_bytes;
  uint64_t out_crouton_bytes;
  uint64_t acc_bytes;
  uint64_t row_bytes;
  uint64_t status_bytes;
  uint64_t total_bytes;
};

struct HmxPagedAttentionTile {
  int32_t q_block;
  int32_t kv_block;
  int32_t req_block;
  int32_t max_rows;
  int32_t max_cols;
  HmxPagedAttentionLayout layout;
};

static inline bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t* out) {
  if (a != 0 && b > UINT64_MAX / a) {
    return false;
  }
  *out = a * b;
  return true;
}

static inline bool checked_add_u64(uint64_t a, uint64_t b, uint64_t* out) {
  if (b > UINT64_MAX - a) {
    return false;
  }
  *out = a + b;
  return true;
}

static inline bool aligned_alloc_size(uint64_t* offset, uint64_t alignment,
                                      uint64_t bytes) {
  const uint64_t aligned = align_up_u64(*offset, alignment);
  uint64_t next = 0;
  if (!checked_add_u64(aligned, bytes, &next)) {
    return false;
  }
  *offset = next;
  return true;
}

static inline bool hmx_attention_layout_build(int32_t max_rows,
                                              int32_t max_cols,
                                              int32_t head_dim,
                                              HmxPagedAttentionLayout* layout) {
  if (max_rows <= 0 || max_cols <= 0 || head_dim <= 0) {
    return false;
  }

  uint64_t rows_d = 0;
  uint64_t cols_d = 0;
  uint64_t rows_cols = 0;
  if (!checked_mul_u64((uint64_t)max_rows, (uint64_t)head_dim, &rows_d) ||
      !checked_mul_u64((uint64_t)max_cols, (uint64_t)head_dim, &cols_d) ||
      !checked_mul_u64((uint64_t)max_rows, (uint64_t)max_cols, &rows_cols)) {
    return false;
  }

  layout->q_bytes = rows_d * sizeof(float16);
  layout->k_rhs_bytes = cols_d * sizeof(float16);
  layout->s_bytes = rows_cols * sizeof(float16);
  layout->p_bytes = rows_cols * sizeof(float16);
  layout->v_rhs_bytes = cols_d * sizeof(float16);
  layout->pv_bytes = rows_d * sizeof(float16);
  layout->acc_bytes = rows_d * sizeof(float);
  layout->row_bytes = (uint64_t)max_rows * sizeof(float);
  layout->status_bytes = sizeof(int32_t);

  layout->q_lhs_crouton_bytes = hmx_lhs_crouton_bytes(max_rows, head_dim);
  layout->lhs_crouton_bytes = hmx_lhs_crouton_bytes(max_rows, max_cols);

  layout->rhs_crouton_bytes = qaic_compute_rhs_croutons_size_in_bytes(
      1, max_cols, head_dim, sizeof(float16));
  const uint64_t rhs_pv_bytes = qaic_compute_rhs_croutons_size_in_bytes(
      1, head_dim, max_cols, sizeof(float16));
  if (rhs_pv_bytes > layout->rhs_crouton_bytes) {
    layout->rhs_crouton_bytes = rhs_pv_bytes;
  }

  layout->out_crouton_bytes = hmx_out_crouton_bytes(max_rows, max_cols);
  const uint64_t out_pv_bytes = hmx_out_crouton_bytes(max_rows, head_dim);
  if (out_pv_bytes > layout->out_crouton_bytes) {
    layout->out_crouton_bytes = out_pv_bytes;
  }

  uint64_t offset = 0;
  if (!aligned_alloc_size(&offset, QAIC_HMX_INPUT_ALIGNMENT, layout->q_bytes) ||
      !aligned_alloc_size(&offset, QAIC_HMX_INPUT_ALIGNMENT,
                          layout->k_rhs_bytes) ||
      !aligned_alloc_size(&offset, QAIC_HMX_INPUT_ALIGNMENT,
                          layout->k_rhs_bytes) ||
      !aligned_alloc_size(&offset, QAIC_HMX_INPUT_ALIGNMENT, layout->s_bytes) ||
      !aligned_alloc_size(&offset, QAIC_HMX_INPUT_ALIGNMENT, layout->p_bytes) ||
      !aligned_alloc_size(&offset, QAIC_HMX_INPUT_ALIGNMENT,
                          layout->v_rhs_bytes) ||
      !aligned_alloc_size(&offset, QAIC_HMX_INPUT_ALIGNMENT,
                          layout->v_rhs_bytes) ||
      !aligned_alloc_size(&offset, QAIC_HMX_INPUT_ALIGNMENT,
                          layout->pv_bytes) ||
      !aligned_alloc_size(&offset, QAIC_HMX_ALIGNMENT,
                          layout->q_lhs_crouton_bytes) ||
      !aligned_alloc_size(&offset, QAIC_HMX_ALIGNMENT,
                          layout->lhs_crouton_bytes) ||
      !aligned_alloc_size(&offset, QAIC_HMX_WEIGHT_ALIGNMENT,
                          layout->rhs_crouton_bytes) ||
      !aligned_alloc_size(&offset, QAIC_HMX_ALIGNMENT,
                          layout->out_crouton_bytes) ||
      !aligned_alloc_size(&offset, HVX_VectorSize, layout->acc_bytes) ||
      !aligned_alloc_size(&offset, HVX_VectorSize, layout->row_bytes) ||
      !aligned_alloc_size(&offset, HVX_VectorSize, layout->row_bytes) ||
      !aligned_alloc_size(&offset, HVX_VectorSize, layout->status_bytes) ||
      !checked_add_u64(offset, kHmxVtcmSafetyBytes, &layout->total_bytes)) {
    return false;
  }

  return true;
}

static inline bool hmx_prefill_pair_layout_expand(
    HmxPagedAttentionLayout* layout) {
  if (layout->total_bytes < kHmxVtcmSafetyBytes) {
    return false;
  }
  uint64_t offset = layout->total_bytes - kHmxVtcmSafetyBytes;
  const uint64_t pair_work_bytes =
      layout->q_bytes > layout->p_bytes ? layout->q_bytes : layout->p_bytes;
  if (!aligned_alloc_size(&offset, QAIC_HMX_INPUT_ALIGNMENT,
                          pair_work_bytes) ||
      !aligned_alloc_size(&offset, QAIC_HMX_ALIGNMENT,
                          layout->q_lhs_crouton_bytes) ||
      !aligned_alloc_size(&offset, HVX_VectorSize, layout->acc_bytes) ||
      !aligned_alloc_size(&offset, HVX_VectorSize, layout->row_bytes) ||
      !aligned_alloc_size(&offset, HVX_VectorSize, layout->row_bytes) ||
      !checked_add_u64(offset, kHmxVtcmSafetyBytes, &layout->total_bytes)) {
    return false;
  }
  return true;
}

static inline void hmx_attention_scratch_alloc(
    HmxPagedAttentionScratch* scratch, uint8_t** vtcm_cur,
    const HmxPagedAttentionLayout* layout) {
  scratch->q_rm = (float16*)vtcm_alloc(vtcm_cur, QAIC_HMX_INPUT_ALIGNMENT,
                                       layout->q_bytes);
  scratch->k_rhs_rm[0] = (float16*)vtcm_alloc(
      vtcm_cur, QAIC_HMX_INPUT_ALIGNMENT, layout->k_rhs_bytes);
  scratch->k_rhs_rm[1] = (float16*)vtcm_alloc(
      vtcm_cur, QAIC_HMX_INPUT_ALIGNMENT, layout->k_rhs_bytes);
  scratch->s_rm = (float16*)vtcm_alloc(vtcm_cur, QAIC_HMX_INPUT_ALIGNMENT,
                                       layout->s_bytes);
  scratch->p_rm = (float16*)vtcm_alloc(vtcm_cur, QAIC_HMX_INPUT_ALIGNMENT,
                                       layout->p_bytes);
  scratch->v_rhs_rm[0] = (float16*)vtcm_alloc(
      vtcm_cur, QAIC_HMX_INPUT_ALIGNMENT, layout->v_rhs_bytes);
  scratch->v_rhs_rm[1] = (float16*)vtcm_alloc(
      vtcm_cur, QAIC_HMX_INPUT_ALIGNMENT, layout->v_rhs_bytes);
  scratch->pv_rm = (float16*)vtcm_alloc(vtcm_cur, QAIC_HMX_INPUT_ALIGNMENT,
                                        layout->pv_bytes);
  scratch->q_lhs_crouton = (float16*)vtcm_alloc(
      vtcm_cur, QAIC_HMX_ALIGNMENT, layout->q_lhs_crouton_bytes);
  scratch->lhs_crouton = (float16*)vtcm_alloc(
      vtcm_cur, QAIC_HMX_ALIGNMENT, layout->lhs_crouton_bytes);
  scratch->rhs_crouton = (float16*)vtcm_alloc(
      vtcm_cur, QAIC_HMX_WEIGHT_ALIGNMENT, layout->rhs_crouton_bytes);
  scratch->out_crouton = (float16*)vtcm_alloc(
      vtcm_cur, QAIC_HMX_ALIGNMENT, layout->out_crouton_bytes);
  scratch->out_acc = (float*)vtcm_alloc(vtcm_cur, HVX_VectorSize,
                                        layout->acc_bytes);
  scratch->row_max = (float*)vtcm_alloc(vtcm_cur, HVX_VectorSize,
                                        layout->row_bytes);
  scratch->row_sum = (float*)vtcm_alloc(vtcm_cur, HVX_VectorSize,
                                        layout->row_bytes);
  scratch->hmx_status = (int32_t*)vtcm_alloc(vtcm_cur, HVX_VectorSize,
                                             layout->status_bytes);
}

static inline bool hmx_choose_prefill_tile(int32_t gqa, int32_t head_dim,
                                           uint64_t vtcm_size,
                                           HmxPagedAttentionTile* tile) {
  static const int32_t q_candidates[] = {64, 32, 16, 8, 4, 2, 1};
  static const int32_t kv_candidates[] = {
      2048, 1792, 1536, 1280, 1024, 512, 256, 128, 64, 32, 16};

  uint64_t best_score = 0;
  bool found = false;
  HmxPagedAttentionTile best = {};

  for (uint32_t qi = 0; qi < sizeof(q_candidates) / sizeof(q_candidates[0]);
       ++qi) {
    for (uint32_t ki = 0;
         ki < sizeof(kv_candidates) / sizeof(kv_candidates[0]); ++ki) {
      const int32_t q_block = q_candidates[qi];
      const int32_t kv_block = kv_candidates[ki];
      const int64_t rows64 = (int64_t)q_block * (int64_t)gqa;
      if (rows64 <= 0 || rows64 > INT32_MAX) {
        continue;
      }

      HmxPagedAttentionLayout layout = {};
      if (!hmx_attention_layout_build((int32_t)rows64, kv_block, head_dim,
                                      &layout) ||
          !hmx_prefill_pair_layout_expand(&layout)) {
        continue;
      }
      if (layout.total_bytes > vtcm_size) {
        continue;
      }

      const uint64_t score = (uint64_t)rows64 * (uint64_t)kv_block;
      if (!found || score > best_score ||
          (score == best_score && kv_block > best.kv_block)) {
        best.q_block = q_block;
        best.kv_block = kv_block;
        best.req_block = 0;
        best.max_rows = (int32_t)rows64;
        best.max_cols = kv_block;
        best.layout = layout;
        best_score = score;
        found = true;
      }
    }
  }

  if (found) {
    *tile = best;
  }
  return found;
}

static inline bool hmx_choose_decode_tile(int32_t gqa, int32_t head_dim,
                                          int32_t num_reqs,
                                          int32_t max_seq_len,
                                          int32_t block_size,
                                          uint64_t vtcm_size,
                                          HmxPagedAttentionTile* tile) {
  static const int32_t req_candidates[] = {16, 8, 4, 2, 1};
  static const int32_t kv_candidates[] = {
      2048, 1792, 1536, 1280, 1024, 512, 256, 128, 64, 32, 16};

  if (block_size <= 0) {
    return false;
  }
  const int64_t aligned_seq_len64 =
      max_seq_len > 0
          ? (((int64_t)max_seq_len + block_size - 1) / block_size) *
                block_size
          : block_size;
  if (aligned_seq_len64 > INT32_MAX) {
    return false;
  }
  const int32_t aligned_seq_len = (int32_t)aligned_seq_len64;

  uint64_t best_score = 0;
  bool found = false;
  HmxPagedAttentionTile best = {};

  for (uint32_t ri = 0;
       ri < sizeof(req_candidates) / sizeof(req_candidates[0]); ++ri) {
    int32_t req_block = req_candidates[ri];
    if (req_block > num_reqs) {
      req_block = num_reqs;
    }
    if (req_block <= 0) {
      continue;
    }

    for (uint32_t ki = 0;
         ki <= sizeof(kv_candidates) / sizeof(kv_candidates[0]); ++ki) {
      const int32_t kv_block =
          ki == 0 ? aligned_seq_len : kv_candidates[ki - 1];
      if (kv_block > aligned_seq_len ||
          (ki > 0 && kv_block == aligned_seq_len)) {
        continue;
      }
      // submit_decode_v_dma emits one handle per physical cache-block
      // fragment. Keeping KV tiles block-aligned makes this bound exact for
      // every tile and prevents a grouped decode from overrunning its fixed
      // handle arrays before the first matmul is launched.
      if (kv_block % block_size != 0 ||
          (int64_t)req_block * (kv_block / block_size) > kMaxDmaHandles) {
        continue;
      }
      const int64_t rows64 = (int64_t)req_block * (int64_t)gqa;
      const int64_t cols64 = (int64_t)req_block * (int64_t)kv_block;
      if (rows64 <= 0 || cols64 <= 0 || rows64 > INT32_MAX ||
          cols64 > INT32_MAX) {
        continue;
      }

      HmxPagedAttentionLayout layout = {};
      if (!hmx_attention_layout_build((int32_t)rows64, (int32_t)cols64,
                                      head_dim, &layout)) {
        continue;
      }
      if (layout.total_bytes > vtcm_size) {
        continue;
      }

      const uint64_t useful_score =
          (uint64_t)req_block * (uint64_t)gqa * (uint64_t)kv_block;
      const uint64_t occupancy_score = (uint64_t)rows64 * (uint64_t)cols64;
      const uint64_t score = useful_score * 4 + occupancy_score;
      if (!found || score > best_score ||
          (score == best_score && kv_block > best.kv_block)) {
        best.q_block = 0;
        best.kv_block = kv_block;
        best.req_block = req_block;
        best.max_rows = (int32_t)rows64;
        best.max_cols = (int32_t)cols64;
        best.layout = layout;
        best_score = score;
        found = true;
      }
    }
  }

  if (found) {
    *tile = best;
  }
  return found;
}

static inline int32_t ceil_div_i32(int32_t a, int32_t b) {
  return (a + b - 1) / b;
}

static inline int32_t min_i32(int32_t a, int32_t b) {
  return a < b ? a : b;
}

static inline int32_t snake_task_owner(int32_t task_id, int32_t num_cores) {
  const int32_t batch = task_id / num_cores;
  const int32_t lane = task_id - batch * num_cores;
  return (batch & 1) != 0 ? num_cores - 1 - lane : lane;
}

static inline int32_t paired_prefill_task_owner(
    int32_t pair_index, int32_t kv_head, int32_t num_kv_heads,
    int32_t num_cores, int32_t fallback_task_id) {
  if (num_kv_heads <= 0 || num_cores < num_kv_heads ||
      num_cores % num_kv_heads != 0) {
    return snake_task_owner(fallback_task_id, num_cores);
  }
  // Keep each KV head within a core group and reverse alternating Q batches so
  // later, causally heavier pairs do not accumulate on the same NSPs.
  const int32_t core_groups = num_cores / num_kv_heads;
  const int32_t batch = pair_index / core_groups;
  const int32_t lane = pair_index - batch * core_groups;
  const int32_t group = (batch & 1) == 0 ? core_groups - 1 - lane : lane;
  return group * num_kv_heads + kv_head;
}

static inline int32_t worker_id(const AicJitEntryPointConfig* cfg) {
  return cfg->coreID * cfg->numThreads + (cfg->threadID % cfg->numThreads);
}

static inline int32_t num_workers(const AicJitEntryPointConfig* cfg) {
  return cfg->numCores * cfg->numThreads;
}

static inline bool hmx_thread_is_launched(
    const AicJitEntryPointConfig* cfg, int64_t hmx_thread_id) {
  return hmx_thread_id >= 0 && hmx_thread_id < (int64_t)cfg->numThreads;
}

static inline int32_t hmx_local_hvx_thread(
    uint32_t thread_id, int64_t hmx_thread_id, bool hmx_launched) {
  if (hmx_launched && thread_id > (uint32_t)hmx_thread_id) {
    return (int32_t)thread_id - 1;
  }
  return (int32_t)thread_id;
}

static inline int32_t cache_offset(int32_t block_id, int32_t kv_head,
                                   int32_t block_offset, int32_t dim,
                                   int32_t num_kv_heads, int32_t block_size,
                                   int32_t head_dim) {
  return (((block_id * num_kv_heads + kv_head) * block_size + block_offset) *
          head_dim) +
         dim;
}

constexpr int32_t kCacheStoreRowsPerHmx = 2;

struct PagedAttentionStoreState {
  const float16* key;
  const float16* value;
  float16* key_cache;
  float16* value_cache;
  const int32_t* slot_mapping;
  int32_t num_kv_heads;
  int32_t head_dim;
  int32_t block_size;
  int32_t next_row;
  int32_t end_row;
};

static inline void store_cache_rows_chunk(PagedAttentionStoreState* state,
                                          int32_t max_rows) {
  if (state == NULL || state->next_row >= state->end_row) {
    return;
  }

  int32_t end = state->next_row + max_rows;
  if (end > state->end_row) {
    end = state->end_row;
  }
  const uint32_t row_bytes = (uint32_t)state->head_dim * sizeof(float16);
  for (int32_t row = state->next_row; row < end; ++row) {
    const int32_t token = row / state->num_kv_heads;
    const int32_t kv_head = row - token * state->num_kv_heads;
    const int32_t slot = state->slot_mapping[token];
    if (slot < 0) {
      continue;
    }
    const int32_t block_id = slot / state->block_size;
    const int32_t block_offset = slot - block_id * state->block_size;
    const int32_t src =
        (token * state->num_kv_heads + kv_head) * state->head_dim;
    const int32_t dst =
        cache_offset(block_id, kv_head, block_offset, 0,
                     state->num_kv_heads, state->block_size,
                     state->head_dim);
    memcpy(state->key_cache + dst, state->key + src, row_bytes);
    memcpy(state->value_cache + dst, state->value + src, row_bytes);
  }
  state->next_row = end;
}

static inline int32_t submit_prefill_v_dma(
    uint32_t thread_id, const float16* value, const float16* value_cache,
    float16* v_dst, const int32_t* block_table, int32_t req, int32_t req_q_start,
    int32_t prefix_len, int32_t kv_head, int32_t kv_start, int32_t kv_rows,
    int32_t num_kv_heads, int32_t block_size, int32_t head_dim,
    int32_t max_blocks_per_seq, QShimUDmaHandle* handles,
    int32_t* n_handles) {
  *n_handles = 0;
  int32_t c = 0;
  while (c < kv_rows) {
    const int32_t kv_pos = kv_start + c;
    if (value != NULL && kv_pos >= prefix_len) {
      const int32_t token = req_q_start + (kv_pos - prefix_len);
      const int32_t chunk_rows = kv_rows - c;
      const float16* v_src =
          value + (token * num_kv_heads + kv_head) * head_dim;
      float16* dst = v_dst + c * head_dim;
      const uint32_t width = (uint32_t)head_dim * sizeof(float16);
      const uint32_t dst_stride = width;
      const uint32_t src_stride =
          (uint32_t)num_kv_heads * (uint32_t)head_dim * sizeof(float16);
      if (*n_handles >= kMaxDmaHandles || chunk_rows > UINT16_MAX ||
          width > UINT16_MAX || dst_stride > UINT16_MAX ||
          src_stride > UINT16_MAX) {
        return JIT_DEV_ERROR_INVALID_PARAMETER;
      }
      uint32_t status = JIT_DEV_STATUS_SUCCESS;
      handles[*n_handles] = dma_copy_2d_submit_local(
          thread_id, dst, v_src, (uint32_t)chunk_rows, width, dst_stride,
          src_stride, &status);
      if (status != JIT_DEV_STATUS_SUCCESS ||
          handles[*n_handles] == INVALID_UDMA_HANDLE) {
        return status != JIT_DEV_STATUS_SUCCESS
                   ? status
                   : JIT_DEV_ERROR_INVALID_PARAMETER;
      }
      ++(*n_handles);
      c += chunk_rows;
      continue;
    }

    const int32_t logical_block = kv_pos / block_size;
    const int32_t block_offset = kv_pos - logical_block * block_size;
    int32_t chunk_rows = min_i32(kv_rows - c, block_size - block_offset);
    if (value != NULL) {
      chunk_rows = min_i32(chunk_rows, prefix_len - kv_pos);
    }
    const int32_t block_id =
        block_table[req * max_blocks_per_seq + logical_block];
    const float16* v_src =
        value_cache + cache_offset(block_id, kv_head, block_offset, 0,
                                   num_kv_heads, block_size, head_dim);
    float16* dst = v_dst + c * head_dim;
    const uint64_t bytes64 =
        (uint64_t)chunk_rows * (uint64_t)head_dim * sizeof(float16);
    if (*n_handles >= kMaxDmaHandles || bytes64 > UINT32_MAX) {
      return JIT_DEV_ERROR_INVALID_PARAMETER;
    }
    uint32_t status = JIT_DEV_STATUS_SUCCESS;
    handles[*n_handles] = dma_copy_submit_local(
        thread_id, dst, v_src, (uint32_t)bytes64, &status);
    if (status != JIT_DEV_STATUS_SUCCESS ||
        handles[*n_handles] == INVALID_UDMA_HANDLE) {
      return status != JIT_DEV_STATUS_SUCCESS ? status
                                               : JIT_DEV_ERROR_INVALID_PARAMETER;
    }
    ++(*n_handles);
    c += chunk_rows;
  }
  return JIT_DEV_STATUS_SUCCESS;
}

static inline int32_t submit_decode_v_dma(
    uint32_t thread_id, const float16* value_cache, float16* v_dst,
    const int32_t* block_table, int32_t req_base, int32_t req_tile,
    int32_t kv_head, int32_t kv_start, int32_t kv_block_tile, int32_t n_cols,
    const int32_t* seq_lens, int32_t num_kv_heads, int32_t block_size,
    int32_t head_dim, int32_t max_blocks_per_seq, QShimUDmaHandle* handles,
    int32_t* n_handles) {
  *n_handles = 0;
  for (int32_t rb = 0; rb < req_tile; ++rb) {
    const int32_t req = req_base + rb;
    const int32_t seq_len = seq_lens[req];
    int32_t off = 0;
    while (off < kv_block_tile) {
      const int32_t kv_pos = kv_start + off;
      const int32_t col = rb * kv_block_tile + off;
      if (kv_pos >= seq_len) {
        memset(v_dst + col * head_dim, 0,
               (kv_block_tile - off) * head_dim * sizeof(float16));
        break;
      }

      const int32_t logical_block = kv_pos / block_size;
      const int32_t block_offset = kv_pos - logical_block * block_size;
      int32_t chunk_rows =
          min_i32(kv_block_tile - off, block_size - block_offset);
      if (kv_pos + chunk_rows > seq_len) {
        chunk_rows = seq_len - kv_pos;
      }
      const int32_t block_id =
          block_table[req * max_blocks_per_seq + logical_block];
      const float16* v_src =
          value_cache + cache_offset(block_id, kv_head, block_offset, 0,
                                     num_kv_heads, block_size, head_dim);
      float16* dst = v_dst + col * head_dim;
      const uint64_t bytes64 =
          (uint64_t)chunk_rows * (uint64_t)head_dim * sizeof(float16);
      if (*n_handles >= kMaxDmaHandles || bytes64 > UINT32_MAX) {
        return JIT_DEV_ERROR_INVALID_PARAMETER;
      }
      uint32_t status = JIT_DEV_STATUS_SUCCESS;
      handles[*n_handles] = dma_copy_submit_local(
          thread_id, dst, v_src, (uint32_t)bytes64, &status);
      if (status != JIT_DEV_STATUS_SUCCESS ||
          handles[*n_handles] == INVALID_UDMA_HANDLE) {
        return status != JIT_DEV_STATUS_SUCCESS
                   ? status
                   : JIT_DEV_ERROR_INVALID_PARAMETER;
      }
      ++(*n_handles);
      off += chunk_rows;
    }
  }
  (void)n_cols;
  return JIT_DEV_STATUS_SUCCESS;
}

static inline int32_t flat_kv_offset(int32_t token, int32_t kv_head,
                                     int32_t dim, int32_t num_kv_heads,
                                     int32_t head_dim) {
  return (token * num_kv_heads + kv_head) * head_dim + dim;
}

static inline int32_t flat_q_offset(int32_t token, int32_t head, int32_t dim,
                                    int32_t num_heads, int32_t head_dim) {
  return (token * num_heads + head) * head_dim + dim;
}

static inline uint32_t float_bits(float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static inline void zero_f16_row_hvx(float16* row, int32_t count) {
  const HVX_Vector zero = Q6_V_vzero();
  int32_t i = 0;
  for (; i + kF16PerHvx <= count; i += kF16PerHvx) {
    StoreUnalignedHVX((int8_t*)(row + i), zero);
  }
  if (i < count) {
    StoreUnalignedHVX((int8_t*)(row + i), zero,
                      (count - i) * sizeof(float16));
  }
}

static inline void scale_f16_row_hvx(float16* row, int32_t count,
                                      float scale) {
  const float16 scale_hf = (float16)scale;
  const HVX_Vector scale_vhf = Q6_Vh_vsplat_R(*(const uint16_t*)&scale_hf);
  int32_t i = 0;
  for (; i + kF16PerHvx <= count; i += kF16PerHvx) {
    const HVX_Vector in_hf =
        LoadUnaligned<HVX_Vector>((const int8_t*)(row + i));
    StoreUnalignedHVX((int8_t*)(row + i),
                      Q6_Vhf_vmpy_VhfVhf(in_hf, scale_vhf));
  }
  if (i < count) {
    const int32_t rem = count - i;
    const HVX_Vector in_hf =
        LoadUnaligned<HVX_Vector>((const int8_t*)(row + i),
                                  rem * sizeof(float16));
    StoreUnalignedHVX((int8_t*)(row + i),
                      Q6_Vhf_vmpy_VhfVhf(in_hf, scale_vhf),
                      rem * sizeof(float16));
  }
}

static inline HVX_Vector splat_f16(float value) {
  const float16 value_hf = (float16)value;
  return Q6_Vh_vsplat_R(*(const uint16_t*)&value_hf);
}

// Approximate exp(x) for x in [-16, 0].  Range reduction keeps the degree-5
// polynomial on [-0.5, 0], then five squarings recover the original range.
static inline HVX_Vector softmax_exp_hvx(HVX_Vector centered_hf) {
  const HVX_Vector neg_limit_hf = splat_f16(-16.0f);
  const HVX_Vector inv_32_hf = splat_f16(1.0f / 32.0f);
  const HVX_Vector one_hf = splat_f16(1.0f);

  HVX_Vector x_hf =
      Q6_Vhf_vmax_VhfVhf(centered_hf, neg_limit_hf);
  x_hf = Q6_Vhf_vmpy_VhfVhf(x_hf, inv_32_hf);

  HVX_Vector p_hf = splat_f16(1.0f / 120.0f);
  p_hf = Q6_Vhf_vadd_VhfVhf(
      Q6_Vhf_vmpy_VhfVhf(p_hf, x_hf), splat_f16(1.0f / 24.0f));
  p_hf = Q6_Vhf_vadd_VhfVhf(
      Q6_Vhf_vmpy_VhfVhf(p_hf, x_hf), splat_f16(1.0f / 6.0f));
  p_hf = Q6_Vhf_vadd_VhfVhf(
      Q6_Vhf_vmpy_VhfVhf(p_hf, x_hf), splat_f16(0.5f));
  p_hf =
      Q6_Vhf_vadd_VhfVhf(Q6_Vhf_vmpy_VhfVhf(p_hf, x_hf), one_hf);
  p_hf =
      Q6_Vhf_vadd_VhfVhf(Q6_Vhf_vmpy_VhfVhf(p_hf, x_hf), one_hf);

  for (int32_t i = 0; i < 5; ++i) {
    p_hf = Q6_Vhf_vmpy_VhfVhf(p_hf, p_hf);
  }
  return p_hf;
}

static inline void reduce_f16_as_f32(SumReducerFloat* reducer,
                                     HVX_Vector values_hf) {
  const HVX_VectorPair values_sf = Q6_Wsf_vcvt_Vhf(values_hf);
  reducer->reduce(Q6_V_lo_W(values_sf));
  reducer->reduce(Q6_V_hi_W(values_sf));
}

static inline float online_rescale_exp_hvx(float delta) {
  const HVX_Vector delta_sf = Q6_V_vsplat_R(float_bits(delta));
  float values[kF32PerHvx] __attribute__((aligned(HVX_VectorSize)));
  StoreHVX(values, qaic_exp_sf(delta_sf));
  return values[0];
}

static inline void online_softmax_block_hvx(
    const float16* scores_row, float16* probs_row, int32_t n_cols,
    int32_t valid_begin, int32_t valid_count, float scale,
    float* block_max_out, float* block_sum_out) {
  *block_max_out = -FLT_MAX;
  *block_sum_out = 0.0f;

  if (valid_count <= 0) {
    zero_f16_row_hvx(probs_row, n_cols);
    return;
  }
  if (valid_begin > 0) {
    zero_f16_row_hvx(probs_row, valid_begin);
  }
  const int32_t valid_end = valid_begin + valid_count;
  if (valid_end < n_cols) {
    zero_f16_row_hvx(probs_row + valid_end, n_cols - valid_end);
  }
  const float16* valid_scores = scores_row + valid_begin;
  float16* valid_probs = probs_row + valid_begin;
  const float16 scale_hf = (float16)scale;
  const HVX_Vector scale_vhf = Q6_Vh_vsplat_R(*(const uint16_t*)&scale_hf);

  MaxReducerFloat16 max_reducer;
  int32_t c = 0;
  for (; c + kF16PerHvx <= valid_count; c += kF16PerHvx) {
    const HVX_Vector scores_hf =
        LoadUnaligned<HVX_Vector>((const int8_t*)(valid_scores + c));
    const HVX_Vector scaled_hf = Q6_Vhf_vmpy_VhfVhf(scores_hf, scale_vhf);
    max_reducer.reduce(scaled_hf);
    StoreUnalignedHVX((int8_t*)(valid_probs + c), scaled_hf);
  }
  if (c < valid_count) {
    const int32_t rem = valid_count - c;
    const HVX_Vector scores_hf = LoadUnaligned<HVX_Vector>(
        (const int8_t*)(valid_scores + c), rem * sizeof(float16));
    const HVX_Vector scaled_hf = Q6_Vhf_vmpy_VhfVhf(scores_hf, scale_vhf);
    max_reducer.reduce(scaled_hf, rem);
    StoreUnalignedHVX((int8_t*)(valid_probs + c), scaled_hf,
                      rem * sizeof(float16));
  }

  const HVX_Vector max_hf = max_reducer.finishSplat();
  float16 max_values[kF16PerHvx] __attribute__((aligned(HVX_VectorSize)));
  StoreHVX(max_values, max_hf);
  *block_max_out = (float)max_values[0];

  SumReducerFloat sum_reducer;
  for (c = 0; c + kF16PerHvx <= valid_count; c += kF16PerHvx) {
    const HVX_Vector scaled_hf =
        LoadUnaligned<HVX_Vector>((const int8_t*)(valid_probs + c));
    const HVX_Vector centered_hf =
        Q6_Vhf_vsub_VhfVhf(scaled_hf, max_hf);
    const HVX_Vector exp_hf = softmax_exp_hvx(centered_hf);
    reduce_f16_as_f32(&sum_reducer, exp_hf);
    StoreUnalignedHVX((int8_t*)(valid_probs + c), exp_hf);
  }
  if (c < valid_count) {
    const int32_t rem = valid_count - c;
    const HVX_Vector scaled_hf = LoadUnaligned<HVX_Vector>(
        (const int8_t*)(valid_probs + c), rem * sizeof(float16));
    const HVX_Vector centered_hf =
        Q6_Vhf_vsub_VhfVhf(scaled_hf, max_hf);
    HVX_Vector exp_hf = softmax_exp_hvx(centered_hf);
    const HVX_VectorB valid = Q6_Q_vsetq2_R(rem * sizeof(float16));
    exp_hf = Q6_V_vmux_QVV(valid, exp_hf, Q6_V_vzero());
    reduce_f16_as_f32(&sum_reducer, exp_hf);
    StoreUnalignedHVX((int8_t*)(valid_probs + c), exp_hf,
                      rem * sizeof(float16));
  }

  float sum_values[kF32PerHvx] __attribute__((aligned(HVX_VectorSize)));
  StoreHVX(sum_values, sum_reducer.finishSplat());
  *block_sum_out = sum_values[0];
}
static inline float dot_f16_f32_hvx(const float16* q, const float16* k,
                                    int32_t head_dim) {
  SumReducerFloat reducer;
  int32_t dim = 0;
  for (; dim + kF16PerHvx <= head_dim; dim += kF16PerHvx) {
    const HVX_Vector q_hf = LoadUnaligned<HVX_Vector>((const int8_t*)(q + dim));
    const HVX_Vector k_hf = LoadUnaligned<HVX_Vector>((const int8_t*)(k + dim));
    const HVX_VectorPair q_sf = Q6_Wsf_vcvt_Vhf(q_hf);
    const HVX_VectorPair k_sf = Q6_Wsf_vcvt_Vhf(k_hf);
    reducer.reduce(Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(q_sf), Q6_V_lo_W(k_sf)));
    reducer.reduce(Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(q_sf), Q6_V_hi_W(k_sf)));
  }

  float dot = 0.0f;
  reducer.finish(&dot);
  for (; dim < head_dim; ++dim) {
    dot += (float)q[dim] * (float)k[dim];
  }
  return dot;
}

static inline void zero_accumulator(float* acc, int32_t head_dim) {
  int32_t dim = 0;
  for (; dim + kF32PerHvx <= head_dim; dim += kF32PerHvx) {
    StoreHVX(acc + dim, Q6_V_vzero());
  }
  for (; dim < head_dim; ++dim) {
    acc[dim] = 0.0f;
  }
}

static inline void scale_accumulator_hvx(float* acc, int32_t head_dim,
                                         float scale) {
  const HVX_Vector scale_sf = Q6_V_vsplat_R(float_bits(scale));
  int32_t dim = 0;
  for (; dim + kF32PerHvx <= head_dim; dim += kF32PerHvx) {
    StoreHVX(acc + dim, Q6_Vsf_vmpy_VsfVsf(LoadHVX(acc + dim), scale_sf));
  }
  for (; dim < head_dim; ++dim) {
    acc[dim] *= scale;
  }
}

static inline void accumulate_value_hvx(float* acc, const float16* value,
                                        int32_t head_dim, float weight) {
  const HVX_Vector weight_sf = Q6_V_vsplat_R(float_bits(weight));
  int32_t dim = 0;
  for (; dim + kF16PerHvx <= head_dim; dim += kF16PerHvx) {
    const HVX_Vector v_hf =
        LoadUnaligned<HVX_Vector>((const int8_t*)(value + dim));
    const HVX_VectorPair v_sf = Q6_Wsf_vcvt_Vhf(v_hf);

    HVX_Vector acc_lo = LoadHVX(acc + dim);
    HVX_Vector acc_hi = LoadHVX(acc + dim + kF32PerHvx);
    acc_lo = Q6_Vsf_vadd_VsfVsf(
        acc_lo, Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(v_sf), weight_sf));
    acc_hi = Q6_Vsf_vadd_VsfVsf(
        acc_hi, Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(v_sf), weight_sf));
    StoreHVX(acc + dim, acc_lo);
    StoreHVX(acc + dim + kF32PerHvx, acc_hi);
  }

  for (; dim < head_dim; ++dim) {
    acc[dim] += weight * (float)value[dim];
  }
}

static inline void add_value_hvx(float* acc, const float16* value,
                                 int32_t head_dim) {
  int32_t dim = 0;
  for (; dim + kF16PerHvx <= head_dim; dim += kF16PerHvx) {
    const HVX_Vector value_hf =
        LoadUnaligned<HVX_Vector>((const int8_t*)(value + dim));
    const HVX_VectorPair value_sf = Q6_Wsf_vcvt_Vhf(value_hf);
    StoreHVX(acc + dim,
             Q6_Vsf_vadd_VsfVsf(LoadHVX(acc + dim),
                                Q6_V_lo_W(value_sf)));
    StoreHVX(acc + dim + kF32PerHvx,
             Q6_Vsf_vadd_VsfVsf(LoadHVX(acc + dim + kF32PerHvx),
                                Q6_V_hi_W(value_sf)));
  }

  for (; dim < head_dim; ++dim) {
    acc[dim] += (float)value[dim];
  }
}

static inline void store_normalized_hvx(float16* output, const float* acc,
                                        int32_t head_dim, float inv_denom) {
  const HVX_Vector inv_sf = Q6_V_vsplat_R(float_bits(inv_denom));
  int32_t dim = 0;
  for (; dim + kF16PerHvx <= head_dim; dim += kF16PerHvx) {
    const HVX_Vector out_lo =
        Q6_Vsf_vmpy_VsfVsf(LoadHVX(acc + dim), inv_sf);
    const HVX_Vector out_hi =
        Q6_Vsf_vmpy_VsfVsf(LoadHVX(acc + dim + kF32PerHvx), inv_sf);
    const HVX_Vector out_hf = Q6_Vhf_vcvt_VsfVsf(out_lo, out_hi);
    StoreUnalignedHVX((int8_t*)(output + dim), out_hf);
  }
  // Q6_Vhf_vcvt_VsfVsf packs even/odd fp16 lanes from two fp32 vectors.
  // Residual dimensions are kept in normal scalar order, so store them scalarly.
  for (; dim < head_dim; ++dim) {
    output[dim] = (float16)(acc[dim] * inv_denom);
  }
}

static inline int32_t hmx_prepare_lhs_crouton_16b(uint32_t thread_id,
                                                  int64_t hmx_thread_id,
                                                  float16* lhs_rm,
                                                  float16* lhs_crouton,
                                                  int32_t* shared_status,
                                                  int32_t m,
                                                  int32_t k) {
  int lhs_shape[2] = {m, k};
  const QAicHMXDims<QAIC_HMX_FlatNXYD> lhs_flat_dims(
      lhs_shape, sizeof(float16), HMXShapeKind::YD);
  const QAicHMXDims<QAIC_HMX_MATMUL_CROUTON_16B> lhs_crouton_dims(
      lhs_flat_dims);

  if (is_core_master_hvx_thread(thread_id, hmx_thread_id)) {
    *shared_status = JIT_DEV_STATUS_SUCCESS;
  }
  qaicSyncHVXAndHMXThreads(thread_id);

  int32_t local_status = JIT_DEV_STATUS_SUCCESS;
  if (thread_id != (uint32_t)hmx_thread_id) {
    JitDevStatusCode_t ret = qaic_hmx_rm_matmul_lhs_to_crouton_16b(
        (int16_t*)lhs_crouton, (const int16_t*)lhs_rm, &lhs_crouton_dims,
        &lhs_flat_dims, thread_id);
    if (ret != JIT_DEV_STATUS_SUCCESS) {
      local_status = ret;
      *shared_status = local_status;
    }
  }

  qaicSyncHVXAndHMXThreads(thread_id);
  return *shared_status;
}

static inline int32_t hmx_matmul_prepared_lhs_rm_16b(
    uint32_t thread_id,
    int64_t hmx_thread_id,
    float16* lhs_crouton,
    float16* rhs_rm,
    float16* out_rm,
    float16* rhs_crouton,
    float16* out_crouton,
    int32_t* shared_status,
    int32_t m,
    int32_t k,
    int32_t n,
    bool rhs_col_major,
    bool rhs_prepared,
    PagedAttentionStoreState* store_state) {
  int lhs_shape[2] = {m, k};
  const QAicHMXDims<QAIC_HMX_FlatNXYD> lhs_flat_dims(
      lhs_shape, sizeof(float16), HMXShapeKind::YD);
  const QAicHMXDims<QAIC_HMX_MATMUL_CROUTON_16B> lhs_crouton_dims(
      lhs_flat_dims);

  int out_shape[2] = {m, n};
  const QAicHMXDims<QAIC_HMX_FlatNXYD> out_flat_dims(
      out_shape, sizeof(float16), HMXShapeKind::YD);
  const QAicHMXDims<QAIC_HMX_MATMUL_CROUTON_16B> out_crouton_dims(
      out_flat_dims);

  int rhs_dims[3] = {1, k, n};

  if (is_core_master_hvx_thread(thread_id, hmx_thread_id)) {
    *shared_status = JIT_DEV_STATUS_SUCCESS;
  }
  qaicSyncHVXAndHMXThreads(thread_id);

  int32_t local_status = JIT_DEV_STATUS_SUCCESS;
  if (thread_id != (uint32_t)hmx_thread_id) {
    JitDevStatusCode_t ret = JIT_DEV_STATUS_SUCCESS;
    if (!rhs_prepared) {
      ret = rhs_col_major
                ? qaic_hmx_cm_matmul_rhs_to_crouton_16b(
                      (int16_t*)rhs_crouton, (const int16_t*)rhs_rm, rhs_dims,
                      thread_id)
                : qaic_hmx_rm_matmul_rhs_to_crouton_16b(
                      (int16_t*)rhs_crouton, (const int16_t*)rhs_rm, rhs_dims,
                      thread_id);
    }
    if (ret != JIT_DEV_STATUS_SUCCESS) {
      local_status = ret;
      *shared_status = local_status;
    }
  }

  qaicSyncHVXAndHMXThreads(thread_id);

  if (thread_id == (uint32_t)hmx_thread_id &&
      *shared_status == JIT_DEV_STATUS_SUCCESS) {
    const float16* lhs_inputs[1] = {lhs_crouton};
    qaic_hmx_matmul_with_crouton_16b(
        out_crouton, lhs_inputs, rhs_crouton, &out_flat_dims, &out_crouton_dims,
        &lhs_crouton_dims, rhs_dims, true, true, false);
  } else if (thread_id != (uint32_t)hmx_thread_id) {
    store_cache_rows_chunk(store_state, kCacheStoreRowsPerHmx);
  }

  qaicSyncHVXAndHMXThreads(thread_id);

  if (thread_id != (uint32_t)hmx_thread_id &&
      *shared_status == JIT_DEV_STATUS_SUCCESS) {
    JitDevStatusCode_t ret = qaic_hmx_matmul_crouton_to_rm_16b(
        (int16_t*)out_rm, (const int16_t*)out_crouton, &out_flat_dims,
        &out_crouton_dims, thread_id);
    if (ret != JIT_DEV_STATUS_SUCCESS) {
      local_status = ret;
      *shared_status = local_status;
    }
  }

  qaicSyncHVXAndHMXThreads(thread_id);
  return *shared_status;
}

static inline int32_t hmx_matmul_rm_16b(uint32_t thread_id,
                                        int64_t hmx_thread_id,
                                        float16* lhs_rm,
                                        float16* rhs_rm,
                                        float16* out_rm,
                                        float16* lhs_crouton,
                                        float16* rhs_crouton,
                                        float16* out_crouton,
                                        int32_t* shared_status,
                                        int32_t m,
                                        int32_t k,
                                        int32_t n,
                                        bool rhs_col_major,
                                        bool rhs_prepared,
                                        PagedAttentionStoreState* store_state) {
  int lhs_shape[2] = {m, k};
  const QAicHMXDims<QAIC_HMX_FlatNXYD> lhs_flat_dims(
      lhs_shape, sizeof(float16), HMXShapeKind::YD);
  const QAicHMXDims<QAIC_HMX_MATMUL_CROUTON_16B> lhs_crouton_dims(
      lhs_flat_dims);

  int out_shape[2] = {m, n};
  const QAicHMXDims<QAIC_HMX_FlatNXYD> out_flat_dims(
      out_shape, sizeof(float16), HMXShapeKind::YD);
  const QAicHMXDims<QAIC_HMX_MATMUL_CROUTON_16B> out_crouton_dims(
      out_flat_dims);

  int rhs_dims[3] = {1, k, n};

  if (is_core_master_hvx_thread(thread_id, hmx_thread_id)) {
    *shared_status = JIT_DEV_STATUS_SUCCESS;
  }
  qaicSyncHVXAndHMXThreads(thread_id);

  int32_t local_status = JIT_DEV_STATUS_SUCCESS;
  if (thread_id != (uint32_t)hmx_thread_id) {
    JitDevStatusCode_t ret = JIT_DEV_STATUS_SUCCESS;
    if (!rhs_prepared) {
      ret = rhs_col_major
                ? qaic_hmx_cm_matmul_rhs_to_crouton_16b(
                      (int16_t*)rhs_crouton, (const int16_t*)rhs_rm, rhs_dims,
                      thread_id)
                : qaic_hmx_rm_matmul_rhs_to_crouton_16b(
                      (int16_t*)rhs_crouton, (const int16_t*)rhs_rm, rhs_dims,
                      thread_id);
    }
    if (ret != JIT_DEV_STATUS_SUCCESS) {
      local_status = ret;
    }

    if (local_status == JIT_DEV_STATUS_SUCCESS) {
      ret = qaic_hmx_rm_matmul_lhs_to_crouton_16b(
          (int16_t*)lhs_crouton, (const int16_t*)lhs_rm, &lhs_crouton_dims,
          &lhs_flat_dims, thread_id);
      if (ret != JIT_DEV_STATUS_SUCCESS) {
        local_status = ret;
      }
    }

    if (local_status != JIT_DEV_STATUS_SUCCESS) {
      *shared_status = local_status;
    }
  }

  qaicSyncHVXAndHMXThreads(thread_id);

  if (thread_id == (uint32_t)hmx_thread_id &&
      *shared_status == JIT_DEV_STATUS_SUCCESS) {
    const float16* lhs_inputs[1] = {lhs_crouton};
    qaic_hmx_matmul_with_crouton_16b(
        out_crouton, lhs_inputs, rhs_crouton, &out_flat_dims, &out_crouton_dims,
        &lhs_crouton_dims, rhs_dims, true, true, false);
  } else if (thread_id != (uint32_t)hmx_thread_id) {
    store_cache_rows_chunk(store_state, kCacheStoreRowsPerHmx);
  }

  qaicSyncHVXAndHMXThreads(thread_id);

  if (thread_id != (uint32_t)hmx_thread_id &&
      *shared_status == JIT_DEV_STATUS_SUCCESS) {
    JitDevStatusCode_t ret = qaic_hmx_matmul_crouton_to_rm_16b(
        (int16_t*)out_rm, (const int16_t*)out_crouton, &out_flat_dims,
        &out_crouton_dims, thread_id);
    if (ret != JIT_DEV_STATUS_SUCCESS) {
      local_status = ret;
      *shared_status = local_status;
    }
  }

  qaicSyncHVXAndHMXThreads(thread_id);
  return *shared_status;
}

static inline bool all_requests_are_decode(const int32_t* query_start_loc,
                                           int32_t num_reqs) {
  for (int32_t req = 0; req < num_reqs; ++req) {
    if (query_start_loc[req + 1] - query_start_loc[req] != 1) {
      return false;
    }
  }
  return true;
}

static inline bool map_prefill_ordered_row(
    int32_t work_row, const int32_t* query_start_loc, int32_t num_reqs,
    int32_t num_kv_heads, int32_t gqa, int32_t* req_out, int32_t* token_out,
    int32_t* head_out, int32_t* kv_head_out, int32_t* local_q_out) {
  for (int32_t req = 0; req < num_reqs; ++req) {
    const int32_t q_start = query_start_loc[req];
    const int32_t q_len = query_start_loc[req + 1] - q_start;
    const int32_t rows_this_req = q_len * num_kv_heads * gqa;
    if (work_row >= rows_this_req) {
      work_row -= rows_this_req;
      continue;
    }

    const int32_t rows_per_kv_head = q_len * gqa;
    const int32_t kv_head = work_row / rows_per_kv_head;
    const int32_t rem = work_row - kv_head * rows_per_kv_head;
    const int32_t local_q = rem / gqa;
    const int32_t h_in_group = rem - local_q * gqa;

    *req_out = req;
    *token_out = q_start + local_q;
    *head_out = kv_head * gqa + h_in_group;
    *kv_head_out = kv_head;
    *local_q_out = local_q;
    return true;
  }
  return false;
}

static inline bool map_decode_ordered_row(
    int32_t work_row, const int32_t* query_start_loc, int32_t num_reqs,
    int32_t gqa, int32_t* req_out, int32_t* token_out, int32_t* head_out,
    int32_t* kv_head_out, int32_t* local_q_out) {
  const int32_t rows_per_kv_head = gqa * num_reqs;
  const int32_t kv_head = work_row / rows_per_kv_head;
  const int32_t rem = work_row - kv_head * rows_per_kv_head;
  const int32_t h_in_group = rem / num_reqs;
  const int32_t req = rem - h_in_group * num_reqs;

  *req_out = req;
  *token_out = query_start_loc[req];
  *head_out = kv_head * gqa + h_in_group;
  *kv_head_out = kv_head;
  *local_q_out = 0;
  return true;
}

static inline int32_t paged_attention_hmx_prefill(
    const AicJitEntryPointConfig* cfg, const float16* query,
    const float16* key, const float16* value, float16* key_cache,
    float16* value_cache, const int32_t* slot_mapping, float16* output,
    const int32_t* block_table,
    const int32_t* query_start_loc, const int32_t* seq_lens, int32_t num_reqs,
    int32_t num_tokens, int32_t num_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t block_size, int32_t max_blocks_per_seq, bool causal, float scale) {
  if (head_dim <= 0 || num_kv_heads <= 0 ||
      num_heads % num_kv_heads != 0) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  const int32_t gqa = num_heads / num_kv_heads;
  if (gqa <= 0) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  const uint32_t thread_id = cfg->threadID;
  int64_t hmx_thread_id = QAIC_V68_HVX_THREADS_COUNT;
  qshimQuery(DEV_ATTR_QSHIM_HMX_THREAD_ID, &hmx_thread_id);

  if (thread_id == (uint32_t)hmx_thread_id) {
    qaic_hmx_matmul_bias_init_16b(false);
  }
  qaicSyncHVXAndHMXThreads(thread_id);

  const bool is_hmx_thread = thread_id == (uint32_t)hmx_thread_id;
  const bool hmx_launched = hmx_thread_is_launched(cfg, hmx_thread_id);
  const int32_t hvx_threads =
      (int32_t)cfg->numThreads - (hmx_launched ? 1 : 0);
  const int32_t local_hvx_thread =
      hmx_local_hvx_thread(thread_id, hmx_thread_id, hmx_launched);

  PagedAttentionStoreState store_state = {};
  if (!is_hmx_thread && key != NULL && value != NULL &&
      slot_mapping != NULL) {
    const int32_t total_store_rows = num_tokens * num_kv_heads;
    const int32_t store_workers = (int32_t)cfg->numCores * hvx_threads;
    const int32_t store_worker =
        (int32_t)cfg->coreID * hvx_threads + local_hvx_thread;
    const int32_t store_chunk =
        ceil_div_i32(total_store_rows, store_workers);
    const int32_t store_start = store_worker * store_chunk;
    int32_t store_end = store_start + store_chunk;
    if (store_end > total_store_rows) {
      store_end = total_store_rows;
    }

    store_state.key = key;
    store_state.value = value;
    store_state.key_cache = key_cache;
    store_state.value_cache = value_cache;
    store_state.slot_mapping = slot_mapping;
    store_state.num_kv_heads = num_kv_heads;
    store_state.head_dim = head_dim;
    store_state.block_size = block_size;
    store_state.next_row =
        store_start < total_store_rows ? store_start : total_store_rows;
    store_state.end_row =
        store_end > store_state.next_row ? store_end : store_state.next_row;
  }

  int64_t vtcm_size = 0;
  if (qshimQuery(DEV_ATTR_QSHIM_VTCM_SIZE, &vtcm_size) !=
      JIT_DEV_STATUS_SUCCESS) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  HmxPagedAttentionTile tile = {};
  if (!hmx_choose_prefill_tile(gqa, head_dim, (uint64_t)vtcm_size, &tile)) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }
  const int32_t q_block_tile = tile.q_block;
  const int32_t kv_block_tile = tile.kv_block;

  uint8_t* vtcm_cur = (uint8_t*)qshimGetBaseVtcmAddr();
  HmxPagedAttentionScratch scratch;
  hmx_attention_scratch_alloc(&scratch, &vtcm_cur, &tile.layout);
  const uint64_t pair_work_bytes =
      tile.layout.q_bytes > tile.layout.p_bytes ? tile.layout.q_bytes
                                                : tile.layout.p_bytes;
  // The second Q row-major input is dead before its P probabilities are built.
  float16* pair_work = (float16*)vtcm_alloc(
      &vtcm_cur, QAIC_HMX_INPUT_ALIGNMENT, pair_work_bytes);
  float16* pair_q_rm = pair_work;
  float16* pair_p_rm = pair_work;
  float16* pair_q_lhs_crouton = (float16*)vtcm_alloc(
      &vtcm_cur, QAIC_HMX_ALIGNMENT, tile.layout.q_lhs_crouton_bytes);
  float* pair_out_acc = (float*)vtcm_alloc(
      &vtcm_cur, HVX_VectorSize, tile.layout.acc_bytes);
  float* pair_row_max = (float*)vtcm_alloc(
      &vtcm_cur, HVX_VectorSize, tile.layout.row_bytes);
  float* pair_row_sum = (float*)vtcm_alloc(
      &vtcm_cur, HVX_VectorSize, tile.layout.row_bytes);
  qaicSyncHVXAndHMXThreads(thread_id);
  QShimUDmaHandle k_dma_handles[2][kMaxDmaHandles];
  int32_t k_dma_count[2] = {0, 0};
  QShimUDmaHandle v_dma_handles[2][kMaxDmaHandles];
  int32_t v_dma_count[2] = {0, 0};

  int32_t task_id = 0;
  for (int32_t req = 0; req < num_reqs; ++req) {
    const int32_t req_q_start = query_start_loc[req];
    const int32_t q_len = query_start_loc[req + 1] - req_q_start;
    const int32_t seq_len = seq_lens[req];
    const int32_t prefix_len = seq_len - q_len;

    for (int32_t kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
      for (int32_t q_block = 0; q_block < q_len;
           q_block += 2 * q_block_tile) {
        const int32_t pair_index = q_block / (2 * q_block_tile);
        const bool owns_task = paired_prefill_task_owner(
            pair_index, kv_head, num_kv_heads, (int32_t)cfg->numCores,
            task_id) == (int32_t)cfg->coreID;
        ++task_id;
        if (!owns_task) {
          continue;
        }

        const int32_t q_blocks[2] = {q_block, q_block + q_block_tile};
        const int32_t pair_count = q_blocks[1] < q_len ? 2 : 1;
        int32_t q_rows[2] = {0, 0};
        int32_t m[2] = {0, 0};
        int32_t kv_ends[2] = {0, 0};
        for (int32_t pair = 0; pair < pair_count; ++pair) {
          q_rows[pair] = (q_len - q_blocks[pair]) < q_block_tile
                             ? (q_len - q_blocks[pair])
                             : q_block_tile;
          m[pair] = q_rows[pair] * gqa;
          kv_ends[pair] =
              causal
                  ? min_i32(seq_len,
                            prefix_len + q_blocks[pair] + q_rows[pair])
                  : seq_len;
        }
        const int32_t kv_end = kv_ends[pair_count - 1];

        float16* q_rm_buffers[2] = {scratch.q_rm, pair_q_rm};
        float16* p_rm_buffers[2] = {scratch.p_rm, pair_p_rm};
        float16* q_crouton_buffers[2] = {scratch.q_lhs_crouton,
                                         pair_q_lhs_crouton};
        float* acc_buffers[2] = {scratch.out_acc, pair_out_acc};
        float* row_max_buffers[2] = {scratch.row_max, pair_row_max};
        float* row_sum_buffers[2] = {scratch.row_sum, pair_row_sum};

        if (is_hmx_thread) {
          *scratch.hmx_status = JIT_DEV_STATUS_SUCCESS;
          if (kv_end > 0) {
            const int32_t first_kv_rows =
                kv_end < kv_block_tile ? kv_end : kv_block_tile;
            *scratch.hmx_status = submit_prefill_v_dma(
                thread_id, key, key_cache, scratch.k_rhs_rm[0],
                block_table, req, req_q_start, prefix_len, kv_head, 0,
                first_kv_rows, num_kv_heads, block_size, head_dim,
                max_blocks_per_seq, k_dma_handles[0], &k_dma_count[0]);
            if (*scratch.hmx_status == JIT_DEV_STATUS_SUCCESS) {
              *scratch.hmx_status = submit_prefill_v_dma(
                thread_id, value, value_cache, scratch.v_rhs_rm[0],
                block_table, req, req_q_start, prefix_len, kv_head, 0,
                first_kv_rows, num_kv_heads, block_size, head_dim,
                max_blocks_per_seq, v_dma_handles[0], &v_dma_count[0]);
            }
          }
        }

        if (!is_hmx_thread) {
          for (int32_t pair = 0; pair < pair_count; ++pair) {
            for (int32_t r = local_hvx_thread; r < m[pair];
                 r += hvx_threads) {
              const int32_t local_q = r / gqa;
              const int32_t h_in_group = r - local_q * gqa;
              const int32_t token =
                  req_q_start + q_blocks[pair] + local_q;
              const int32_t head = kv_head * gqa + h_in_group;
              const float16* q_src =
                  query + flat_q_offset(token, head, 0, num_heads, head_dim);
              memcpy(q_rm_buffers[pair] + r * head_dim, q_src,
                     head_dim * sizeof(float16));
              row_max_buffers[pair][r] = -FLT_MAX;
              row_sum_buffers[pair][r] = 0.0f;
            }
          }
        }

        for (int32_t pair = 0; pair < pair_count; ++pair) {
          const int32_t total_acc = m[pair] * head_dim;
          for (int32_t i = local_hvx_thread; i < total_acc;
               i += hvx_threads) {
            acc_buffers[pair][i] = 0.0f;
          }
        }

        qaicSyncHVXAndHMXThreads(thread_id);
        if (*scratch.hmx_status != JIT_DEV_STATUS_SUCCESS) {
          return *scratch.hmx_status;
        }

        int32_t ret = JIT_DEV_STATUS_SUCCESS;
        for (int32_t pair = 0; pair < pair_count; ++pair) {
          ret = hmx_prepare_lhs_crouton_16b(
              thread_id, hmx_thread_id, q_rm_buffers[pair],
              q_crouton_buffers[pair], scratch.hmx_status, m[pair],
              head_dim);
          if (ret != JIT_DEV_STATUS_SUCCESS) {
            return ret;
          }
        }

        for (int32_t kv_start = 0; kv_start < kv_end;
             kv_start += kv_block_tile) {
          const int32_t kv_rows = (kv_end - kv_start) < kv_block_tile
                                      ? (kv_end - kv_start)
                                      : kv_block_tile;
          const int32_t kv_buf = ((kv_start / kv_block_tile) & 1);

          if (is_hmx_thread) {
            int32_t dma_status =
                wait_dma_handles(k_dma_handles[kv_buf], k_dma_count[kv_buf]);
            if (dma_status == JIT_DEV_STATUS_SUCCESS) {
              const int32_t next_kv_start = kv_start + kv_block_tile;
              if (next_kv_start < kv_end) {
                const int32_t next_kv_rows =
                    (kv_end - next_kv_start) < kv_block_tile
                        ? (kv_end - next_kv_start)
                        : kv_block_tile;
                const int32_t next_kv_buf = 1 - kv_buf;
                dma_status = submit_prefill_v_dma(
                    thread_id, key, key_cache, scratch.k_rhs_rm[next_kv_buf],
                    block_table, req, req_q_start, prefix_len, kv_head,
                    next_kv_start, next_kv_rows, num_kv_heads, block_size,
                    head_dim, max_blocks_per_seq, k_dma_handles[next_kv_buf],
                    &k_dma_count[next_kv_buf]);
              }
            }
            if (dma_status != JIT_DEV_STATUS_SUCCESS) {
              *scratch.hmx_status = dma_status;
            }
          }
          qaicSyncHVXAndHMXThreads(thread_id);
          if (*scratch.hmx_status != JIT_DEV_STATUS_SUCCESS) {
            return *scratch.hmx_status;
          }

          bool k_rhs_prepared = false;
          for (int32_t pair = 0; pair < pair_count; ++pair) {
            if (kv_start >= kv_ends[pair]) {
              continue;
            }
            ret = hmx_matmul_prepared_lhs_rm_16b(
                thread_id, hmx_thread_id, q_crouton_buffers[pair],
                scratch.k_rhs_rm[kv_buf], scratch.s_rm, scratch.rhs_crouton,
                scratch.out_crouton, scratch.hmx_status, m[pair], head_dim,
                kv_rows, true, k_rhs_prepared, &store_state);
            if (ret != JIT_DEV_STATUS_SUCCESS) {
              return ret;
            }
            k_rhs_prepared = true;

            if (!is_hmx_thread) {
              for (int32_t r = local_hvx_thread; r < m[pair];
                   r += hvx_threads) {
                const int32_t local_q = r / gqa;
                const int32_t q_pos =
                    prefix_len + q_blocks[pair] + local_q;
                int32_t kv_limit = causal ? (q_pos + 1) : seq_len;
                if (kv_limit > seq_len) {
                  kv_limit = seq_len;
                }

                int32_t valid_count = kv_limit - kv_start;
                if (valid_count > kv_rows) {
                  valid_count = kv_rows;
                }
                float16* p_row = p_rm_buffers[pair] + r * kv_rows;
                if (valid_count <= 0) {
                  zero_f16_row_hvx(p_row, kv_rows);
                  continue;
                }

                const float old_max = row_max_buffers[pair][r];
                float block_max = -FLT_MAX;
                float block_sum = 0.0f;
                online_softmax_block_hvx(
                    scratch.s_rm + r * kv_rows, p_row, kv_rows, 0,
                    valid_count, scale, &block_max, &block_sum);
                const float new_max =
                    block_max > old_max ? block_max : old_max;
                float prev_rescale = 1.0f;
                float curr_rescale = 1.0f;
                if (block_max > old_max) {
                  prev_rescale =
                      old_max == -FLT_MAX
                          ? 0.0f
                          : online_rescale_exp_hvx(old_max - block_max);
                } else if (block_max < old_max) {
                  curr_rescale =
                      online_rescale_exp_hvx(block_max - old_max);
                }
                if (prev_rescale != 1.0f) {
                  scale_accumulator_hvx(acc_buffers[pair] + r * head_dim,
                                        head_dim, prev_rescale);
                }
                if (curr_rescale != 1.0f) {
                  scale_f16_row_hvx(p_row, valid_count, curr_rescale);
                }

                row_max_buffers[pair][r] = new_max;
                row_sum_buffers[pair][r] =
                    row_sum_buffers[pair][r] * prev_rescale +
                    block_sum * curr_rescale;
              }
            }
          }

          if (is_hmx_thread) {
            int32_t wait_status =
                wait_dma_handles(v_dma_handles[kv_buf], v_dma_count[kv_buf]);
            if (wait_status == JIT_DEV_STATUS_SUCCESS) {
              const int32_t next_kv_start = kv_start + kv_block_tile;
              if (next_kv_start < kv_end) {
                const int32_t next_kv_rows =
                    (kv_end - next_kv_start) < kv_block_tile
                        ? (kv_end - next_kv_start)
                        : kv_block_tile;
                const int32_t next_kv_buf = 1 - kv_buf;
                wait_status = submit_prefill_v_dma(
                    thread_id, value, value_cache,
                    scratch.v_rhs_rm[next_kv_buf], block_table, req,
                    req_q_start, prefix_len, kv_head, next_kv_start,
                    next_kv_rows, num_kv_heads, block_size, head_dim,
                    max_blocks_per_seq, v_dma_handles[next_kv_buf],
                    &v_dma_count[next_kv_buf]);
              }
            }
            if (wait_status != JIT_DEV_STATUS_SUCCESS) {
              *scratch.hmx_status = wait_status;
            }
          }

          qaicSyncHVXAndHMXThreads(thread_id);
          if (*scratch.hmx_status != JIT_DEV_STATUS_SUCCESS) {
            return *scratch.hmx_status;
          }

          bool v_rhs_prepared = false;
          for (int32_t pair = 0; pair < pair_count; ++pair) {
            if (kv_start >= kv_ends[pair]) {
              continue;
            }
            ret = hmx_matmul_rm_16b(
                thread_id, hmx_thread_id, p_rm_buffers[pair],
                scratch.v_rhs_rm[kv_buf], scratch.pv_rm,
                scratch.lhs_crouton, scratch.rhs_crouton,
                scratch.out_crouton, scratch.hmx_status, m[pair], kv_rows,
                head_dim, false, v_rhs_prepared, &store_state);
            if (ret != JIT_DEV_STATUS_SUCCESS) {
              return ret;
            }
            v_rhs_prepared = true;

            if (!is_hmx_thread) {
              for (int32_t r = local_hvx_thread; r < m[pair];
                   r += hvx_threads) {
                add_value_hvx(acc_buffers[pair] + r * head_dim,
                              scratch.pv_rm + r * head_dim, head_dim);
              }
            }

            qaicSyncHVXAndHMXThreads(thread_id);
          }
        }

        if (!is_hmx_thread) {
          for (int32_t pair = 0; pair < pair_count; ++pair) {
            for (int32_t r = local_hvx_thread; r < m[pair];
                 r += hvx_threads) {
              const int32_t local_q = r / gqa;
              const int32_t h_in_group = r - local_q * gqa;
              const int32_t token =
                  req_q_start + q_blocks[pair] + local_q;
              const int32_t head = kv_head * gqa + h_in_group;
              float16* out_row =
                  output + flat_q_offset(token, head, 0, num_heads, head_dim);
              const float denom = row_sum_buffers[pair][r];
              const float inv_denom = denom == 0.0f ? 0.0f : 1.0f / denom;
              store_normalized_hvx(out_row,
                                   acc_buffers[pair] + r * head_dim,
                                   head_dim, inv_denom);
            }
          }
        }

        qaicSyncHVXAndHMXThreads(thread_id);
      }
    }
  }

  if (!is_hmx_thread) {
    store_cache_rows_chunk(&store_state,
                           store_state.end_row - store_state.next_row);
  }
  qaicSyncHVXAndHMXThreads(thread_id);

  return JIT_DEV_STATUS_SUCCESS;
}

static inline int32_t paged_attention_hmx_decode(
    const AicJitEntryPointConfig* cfg, const float16* query,
    const float16* key, const float16* value, float16* key_cache,
    float16* value_cache, const int32_t* slot_mapping, float16* output,
    const int32_t* block_table, const int32_t* query_start_loc,
    const int32_t* seq_lens, int32_t num_reqs, int32_t num_heads,
    int32_t num_kv_heads, int32_t head_dim, int32_t block_size,
    int32_t max_blocks_per_seq, bool causal, float scale) {
  (void)causal;
  if (head_dim <= 0 || num_kv_heads <= 0 ||
      num_heads % num_kv_heads != 0) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  const int32_t gqa = num_heads / num_kv_heads;
  if (gqa <= 0) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  const uint32_t thread_id = cfg->threadID;
  int64_t hmx_thread_id = QAIC_V68_HVX_THREADS_COUNT;
  qshimQuery(DEV_ATTR_QSHIM_HMX_THREAD_ID, &hmx_thread_id);

  if (thread_id == (uint32_t)hmx_thread_id) {
    qaic_hmx_matmul_bias_init_16b(false);
  }
  qaicSyncHVXAndHMXThreads(thread_id);

  const bool is_hmx_thread = thread_id == (uint32_t)hmx_thread_id;
  const bool hmx_launched = hmx_thread_is_launched(cfg, hmx_thread_id);
  const int32_t hvx_threads =
      (int32_t)cfg->numThreads - (hmx_launched ? 1 : 0);
  const int32_t local_hvx_thread =
      hmx_local_hvx_thread(thread_id, hmx_thread_id, hmx_launched);

  int64_t vtcm_size = 0;
  if (qshimQuery(DEV_ATTR_QSHIM_VTCM_SIZE, &vtcm_size) !=
      JIT_DEV_STATUS_SUCCESS) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  int32_t global_max_seq_len = 0;
  for (int32_t req = 0; req < num_reqs; ++req) {
    if (seq_lens[req] > global_max_seq_len) {
      global_max_seq_len = seq_lens[req];
    }
  }

  HmxPagedAttentionTile tile = {};
  if (!hmx_choose_decode_tile(gqa, head_dim, num_reqs, global_max_seq_len,
                              block_size, (uint64_t)vtcm_size, &tile)) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }
  const int32_t kv_block_tile = tile.kv_block;
  const int32_t reqs_per_task = tile.req_block;

  uint8_t* vtcm_cur = (uint8_t*)qshimGetBaseVtcmAddr();
  HmxPagedAttentionScratch scratch;
  hmx_attention_scratch_alloc(&scratch, &vtcm_cur, &tile.layout);
  if (is_core_master_hvx_thread(thread_id, hmx_thread_id)) {
  }
  qaicSyncHVXAndHMXThreads(thread_id);
  QShimUDmaHandle k_dma_handles[2][kMaxDmaHandles];
  int32_t k_dma_count[2] = {0, 0};
  QShimUDmaHandle v_dma_handles[2][kMaxDmaHandles];
  int32_t v_dma_count[2] = {0, 0};

  if (reqs_per_task <= 0) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  int32_t task_id = 0;
  for (int32_t kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
    for (int32_t req_base = 0; req_base < num_reqs;
         req_base += reqs_per_task) {
      const bool owns_task = (task_id % (int32_t)cfg->numCores) ==
                             (int32_t)cfg->coreID;
      ++task_id;
      if (!owns_task) {
        continue;
      }

      const int32_t req_tile = (num_reqs - req_base) < reqs_per_task
                                   ? (num_reqs - req_base)
                                   : reqs_per_task;
      const int32_t m = gqa * req_tile;
      const int32_t n_cols = req_tile * kv_block_tile;

      int32_t max_seq_len = 0;
      for (int32_t rb = 0; rb < req_tile; ++rb) {
        const int32_t seq_len = seq_lens[req_base + rb];
        max_seq_len = seq_len > max_seq_len ? seq_len : max_seq_len;
      }

      if (!is_hmx_thread) {
        for (int32_t rb = local_hvx_thread; rb < req_tile;
             rb += hvx_threads) {
          const int32_t req = req_base + rb;
          const int32_t token = query_start_loc[req];
          const int32_t slot = slot_mapping != NULL ? slot_mapping[token] : -1;
          if (slot >= 0 && key != NULL && value != NULL) {
            const int32_t block_id = slot / block_size;
            const int32_t block_offset = slot - block_id * block_size;
            const int32_t src =
                flat_kv_offset(token, kv_head, 0, num_kv_heads, head_dim);
            const int32_t dst =
                cache_offset(block_id, kv_head, block_offset, 0,
                             num_kv_heads, block_size, head_dim);
            const uint32_t row_bytes =
                (uint32_t)head_dim * sizeof(float16);
            memcpy(key_cache + dst, key + src, row_bytes);
            memcpy(value_cache + dst, value + src, row_bytes);
          }
        }

        for (int32_t r = local_hvx_thread; r < m; r += hvx_threads) {
          const int32_t h = r / req_tile;
          const int32_t rb = r - h * req_tile;
          const int32_t req = req_base + rb;
          const int32_t token = query_start_loc[req];
          const int32_t head = kv_head * gqa + h;
          const float16* q_src =
              query + flat_q_offset(token, head, 0, num_heads, head_dim);
          memcpy(scratch.q_rm + r * head_dim, q_src,
                 head_dim * sizeof(float16));
          scratch.row_max[r] = -FLT_MAX;
          scratch.row_sum[r] = 0.0f;
        }

        for (int32_t r = local_hvx_thread; r < m; r += hvx_threads) {
          zero_accumulator(scratch.out_acc + r * head_dim, head_dim);
        }
      }

      // Each task-owning NSP stores exactly the cache rows it will read.
      qaicSyncHVXAndHMXThreads(thread_id);

      if (is_hmx_thread) {
        *scratch.hmx_status = JIT_DEV_STATUS_SUCCESS;
        if (max_seq_len > 0) {
          *scratch.hmx_status = submit_decode_v_dma(
              thread_id, key_cache, scratch.k_rhs_rm[0], block_table,
              req_base, req_tile, kv_head, 0, kv_block_tile, n_cols,
              seq_lens, num_kv_heads, block_size, head_dim,
              max_blocks_per_seq, k_dma_handles[0], &k_dma_count[0]);
          if (*scratch.hmx_status == JIT_DEV_STATUS_SUCCESS) {
            *scratch.hmx_status = submit_decode_v_dma(
                thread_id, value_cache, scratch.v_rhs_rm[0], block_table,
                req_base, req_tile, kv_head, 0, kv_block_tile, n_cols,
                seq_lens, num_kv_heads, block_size, head_dim,
                max_blocks_per_seq, v_dma_handles[0], &v_dma_count[0]);
          }
        }
      }

      qaicSyncHVXAndHMXThreads(thread_id);
      if (*scratch.hmx_status != JIT_DEV_STATUS_SUCCESS) {
        return *scratch.hmx_status;
      }

      int32_t ret = hmx_prepare_lhs_crouton_16b(
          thread_id, hmx_thread_id, scratch.q_rm, scratch.q_lhs_crouton,
          scratch.hmx_status, m, head_dim);
      if (ret != JIT_DEV_STATUS_SUCCESS) {
        return ret;
      }

      for (int32_t kv_start = 0; kv_start < max_seq_len;
           kv_start += kv_block_tile) {
        const int32_t kv_buf = ((kv_start / kv_block_tile) & 1);

        if (is_hmx_thread) {
          int32_t dma_status =
              wait_dma_handles(k_dma_handles[kv_buf], k_dma_count[kv_buf]);
          if (dma_status == JIT_DEV_STATUS_SUCCESS) {
            const int32_t next_kv_start = kv_start + kv_block_tile;
            if (next_kv_start < max_seq_len) {
              const int32_t next_kv_buf = 1 - kv_buf;
              dma_status = submit_decode_v_dma(
                  thread_id, key_cache, scratch.k_rhs_rm[next_kv_buf],
                  block_table, req_base, req_tile, kv_head, next_kv_start,
                  kv_block_tile, n_cols, seq_lens, num_kv_heads, block_size,
                  head_dim, max_blocks_per_seq, k_dma_handles[next_kv_buf],
                  &k_dma_count[next_kv_buf]);
            }
          }
          if (dma_status != JIT_DEV_STATUS_SUCCESS) {
            *scratch.hmx_status = dma_status;
          }
        }
        qaicSyncHVXAndHMXThreads(thread_id);
        if (*scratch.hmx_status != JIT_DEV_STATUS_SUCCESS) {
          return *scratch.hmx_status;
        }

        ret = hmx_matmul_prepared_lhs_rm_16b(
            thread_id, hmx_thread_id, scratch.q_lhs_crouton,
            scratch.k_rhs_rm[kv_buf], scratch.s_rm, scratch.rhs_crouton,
            scratch.out_crouton, scratch.hmx_status, m, head_dim, n_cols,
            true, false, NULL);
        if (ret != JIT_DEV_STATUS_SUCCESS) {
          return ret;
        }

        if (!is_hmx_thread) {
          for (int32_t r = local_hvx_thread; r < m; r += hvx_threads) {
            const int32_t req_lane = r % req_tile;
            const int32_t req = req_base + req_lane;
            const int32_t seq_len = seq_lens[req];

            int32_t valid_count = seq_len - kv_start;
            if (valid_count > kv_block_tile) {
              valid_count = kv_block_tile;
            }
            const int32_t valid_begin = req_lane * kv_block_tile;
            if (valid_count <= 0) {
              zero_f16_row_hvx(scratch.p_rm + r * n_cols, n_cols);
              continue;
            }

            const float old_max = scratch.row_max[r];
            float block_max = -FLT_MAX;
            float block_sum = 0.0f;
            float16* p_row = scratch.p_rm + r * n_cols;
          online_softmax_block_hvx(
              scratch.s_rm + r * n_cols, p_row, n_cols, valid_begin,
              valid_count, scale, &block_max, &block_sum);
            const float new_max =
                block_max > old_max ? block_max : old_max;
            float prev_rescale = 1.0f;
            float curr_rescale = 1.0f;
            if (block_max > old_max) {
              prev_rescale =
                  old_max == -FLT_MAX
                      ? 0.0f
                      : online_rescale_exp_hvx(old_max - block_max);
            } else if (block_max < old_max) {
              curr_rescale = online_rescale_exp_hvx(block_max - old_max);
            }
            if (prev_rescale != 1.0f) {
              scale_accumulator_hvx(scratch.out_acc + r * head_dim,
                                    head_dim, prev_rescale);
            }
            if (curr_rescale != 1.0f) {
              scale_f16_row_hvx(p_row + valid_begin, valid_count,
                                 curr_rescale);
            }

            scratch.row_max[r] = new_max;
            scratch.row_sum[r] =
                scratch.row_sum[r] * prev_rescale +
                block_sum * curr_rescale;
          }
        }

        if (is_hmx_thread) {
          int32_t wait_status =
              wait_dma_handles(v_dma_handles[kv_buf], v_dma_count[kv_buf]);
          if (wait_status == JIT_DEV_STATUS_SUCCESS) {
            const int32_t next_kv_start = kv_start + kv_block_tile;
            if (next_kv_start < max_seq_len) {
              const int32_t next_kv_buf = 1 - kv_buf;
              wait_status = submit_decode_v_dma(
                  thread_id, value_cache, scratch.v_rhs_rm[next_kv_buf],
                  block_table, req_base, req_tile, kv_head, next_kv_start,
                  kv_block_tile, n_cols, seq_lens, num_kv_heads, block_size,
                  head_dim, max_blocks_per_seq, v_dma_handles[next_kv_buf],
                  &v_dma_count[next_kv_buf]);
            }
          }
          if (wait_status != JIT_DEV_STATUS_SUCCESS) {
            *scratch.hmx_status = wait_status;
          }
        }

        qaicSyncHVXAndHMXThreads(thread_id);
        if (*scratch.hmx_status != JIT_DEV_STATUS_SUCCESS) {
          return *scratch.hmx_status;
        }

        ret = hmx_matmul_rm_16b(
            thread_id, hmx_thread_id, scratch.p_rm, scratch.v_rhs_rm[kv_buf],
            scratch.pv_rm, scratch.lhs_crouton, scratch.rhs_crouton,
            scratch.out_crouton, scratch.hmx_status, m, n_cols, head_dim,
            false, false, NULL);
        if (ret != JIT_DEV_STATUS_SUCCESS) {
          return ret;
        }

        if (!is_hmx_thread) {
          for (int32_t r = local_hvx_thread; r < m; r += hvx_threads) {
            add_value_hvx(scratch.out_acc + r * head_dim,
                          scratch.pv_rm + r * head_dim, head_dim);
          }
        }

        qaicSyncHVXAndHMXThreads(thread_id);
      }

      if (!is_hmx_thread) {
        for (int32_t r = local_hvx_thread; r < m; r += hvx_threads) {
          const int32_t h = r / req_tile;
          const int32_t rb = r - h * req_tile;
          const int32_t req = req_base + rb;
          const int32_t token = query_start_loc[req];
          const int32_t head = kv_head * gqa + h;
          float16* out_row =
              output + flat_q_offset(token, head, 0, num_heads, head_dim);
          const float denom = scratch.row_sum[r];
          const float inv_denom = denom == 0.0f ? 0.0f : 1.0f / denom;
          store_normalized_hvx(out_row, scratch.out_acc + r * head_dim,
                               head_dim, inv_denom);
        }
      }

      qaicSyncHVXAndHMXThreads(thread_id);
    }
  }

  return JIT_DEV_STATUS_SUCCESS;
}

}  // namespace

QAIC_KERNEL_API int32_t multinsp_multithreaded_paged_attention_store(
    const AicJitEntryPointConfig* cfg, const AicJitPointerArray* ptrs) {
  const float16* key = (const float16*)ptrs->pointers[0];
  const float16* value = (const float16*)ptrs->pointers[1];
  float16* key_cache = (float16*)ptrs->pointers[2];
  float16* value_cache = (float16*)ptrs->pointers[3];
  const int32_t* slot_mapping = (const int32_t*)ptrs->pointers[4];

  const int32_t num_tokens = *(const int32_t*)ptrs->pointers[5];
  const int32_t num_kv_heads = *(const int32_t*)ptrs->pointers[6];
  const int32_t head_dim = *(const int32_t*)ptrs->pointers[7];
  const int32_t block_size = *(const int32_t*)ptrs->pointers[8];

  const int32_t total = num_tokens * num_kv_heads;
  const int32_t wid = worker_id(cfg);
  const int32_t workers = num_workers(cfg);
  const int32_t chunk = ceil_div_i32(total, workers);
  const int32_t start = wid * chunk;
  const int32_t end = start + chunk < total ? start + chunk : total;

  for (int32_t row = start; row < end; ++row) {
    const int32_t kv_head = row % num_kv_heads;
    const int32_t token = row / num_kv_heads;

    const int32_t slot = slot_mapping[token];
    if (slot < 0) {
      continue;
    }
    const int32_t block_id = slot / block_size;
    const int32_t block_offset = slot - block_id * block_size;

    const int32_t src =
        flat_kv_offset(token, kv_head, 0, num_kv_heads, head_dim);
    const int32_t dst = cache_offset(block_id, kv_head, block_offset, 0,
                                     num_kv_heads, block_size, head_dim);
    const uint32_t row_bytes = (uint32_t)head_dim * sizeof(float16);
    memcpy(key_cache + dst, key + src, row_bytes);
    memcpy(value_cache + dst, value + src, row_bytes);
  }

  return JIT_DEV_STATUS_SUCCESS;
}

QAIC_KERNEL_API int32_t multinsp_multithreaded_paged_attention_hmx_prefill(
    const AicJitEntryPointConfig* cfg, const AicJitPointerArray* ptrs) {
  const float16* query = (const float16*)ptrs->pointers[0];
  float16* key_cache = (float16*)ptrs->pointers[1];
  float16* value_cache = (float16*)ptrs->pointers[2];
  float16* output = (float16*)ptrs->pointers[3];
  const int32_t* block_table = (const int32_t*)ptrs->pointers[4];
  const int32_t* query_start_loc = (const int32_t*)ptrs->pointers[5];
  const int32_t* seq_lens = (const int32_t*)ptrs->pointers[6];
  const int32_t num_reqs = *(const int32_t*)ptrs->pointers[7];
  const int32_t num_tokens = *(const int32_t*)ptrs->pointers[8];
  const int32_t num_heads = *(const int32_t*)ptrs->pointers[9];
  const int32_t num_kv_heads = *(const int32_t*)ptrs->pointers[10];
  const int32_t head_dim = *(const int32_t*)ptrs->pointers[11];
  const int32_t block_size = *(const int32_t*)ptrs->pointers[12];
  const int32_t max_blocks_per_seq = *(const int32_t*)ptrs->pointers[13];
  const int32_t causal = *(const int32_t*)ptrs->pointers[14];
  const float scale = *(const float*)ptrs->pointers[15];

  if (all_requests_are_decode(query_start_loc, num_reqs) ||
      num_tokens <= num_reqs) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  return paged_attention_hmx_prefill(
      cfg, query, NULL, NULL, key_cache, value_cache, NULL, output,
      block_table, query_start_loc, seq_lens, num_reqs, num_tokens, num_heads,
      num_kv_heads, head_dim, block_size, max_blocks_per_seq, causal != 0,
      scale);
}

QAIC_KERNEL_API int32_t
multinsp_multithreaded_paged_attention_hmx_prefill_direct(
    const AicJitEntryPointConfig* cfg, const AicJitPointerArray* ptrs) {
  const float16* query = (const float16*)ptrs->pointers[0];
  const float16* key = (const float16*)ptrs->pointers[1];
  const float16* value = (const float16*)ptrs->pointers[2];
  float16* key_cache = (float16*)ptrs->pointers[3];
  float16* value_cache = (float16*)ptrs->pointers[4];
  const int32_t* slot_mapping = (const int32_t*)ptrs->pointers[5];
  float16* output = (float16*)ptrs->pointers[6];
  const int32_t* block_table = (const int32_t*)ptrs->pointers[7];
  const int32_t* query_start_loc = (const int32_t*)ptrs->pointers[8];
  const int32_t* seq_lens = (const int32_t*)ptrs->pointers[9];
  const int32_t num_reqs = *(const int32_t*)ptrs->pointers[10];
  const int32_t num_tokens = *(const int32_t*)ptrs->pointers[11];
  const int32_t num_heads = *(const int32_t*)ptrs->pointers[12];
  const int32_t num_kv_heads = *(const int32_t*)ptrs->pointers[13];
  const int32_t head_dim = *(const int32_t*)ptrs->pointers[14];
  const int32_t block_size = *(const int32_t*)ptrs->pointers[15];
  const int32_t max_blocks_per_seq = *(const int32_t*)ptrs->pointers[16];
  const int32_t causal = *(const int32_t*)ptrs->pointers[17];
  const float scale = *(const float*)ptrs->pointers[18];

  if (all_requests_are_decode(query_start_loc, num_reqs) ||
      num_tokens <= num_reqs) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  return paged_attention_hmx_prefill(
      cfg, query, key, value, key_cache, value_cache, slot_mapping, output,
      block_table, query_start_loc, seq_lens, num_reqs, num_tokens, num_heads,
      num_kv_heads, head_dim, block_size, max_blocks_per_seq, causal != 0,
      scale);
}

QAIC_KERNEL_API int32_t multinsp_multithreaded_paged_attention_hmx_decode(
    const AicJitEntryPointConfig* cfg, const AicJitPointerArray* ptrs) {
  const float16* query = (const float16*)ptrs->pointers[0];
  const float16* key = (const float16*)ptrs->pointers[1];
  const float16* value = (const float16*)ptrs->pointers[2];
  float16* key_cache = (float16*)ptrs->pointers[3];
  float16* value_cache = (float16*)ptrs->pointers[4];
  const int32_t* slot_mapping = (const int32_t*)ptrs->pointers[5];
  float16* output = (float16*)ptrs->pointers[6];
  const int32_t* block_table = (const int32_t*)ptrs->pointers[7];
  const int32_t* query_start_loc = (const int32_t*)ptrs->pointers[8];
  const int32_t* seq_lens = (const int32_t*)ptrs->pointers[9];
  const int32_t num_reqs = *(const int32_t*)ptrs->pointers[10];
  const int32_t num_tokens = *(const int32_t*)ptrs->pointers[11];
  const int32_t num_heads = *(const int32_t*)ptrs->pointers[12];
  const int32_t num_kv_heads = *(const int32_t*)ptrs->pointers[13];
  const int32_t head_dim = *(const int32_t*)ptrs->pointers[14];
  const int32_t block_size = *(const int32_t*)ptrs->pointers[15];
  const int32_t max_blocks_per_seq = *(const int32_t*)ptrs->pointers[16];
  const int32_t causal = *(const int32_t*)ptrs->pointers[17];
  const float scale = *(const float*)ptrs->pointers[18];

  if (!all_requests_are_decode(query_start_loc, num_reqs) ||
      num_tokens != num_reqs) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  return paged_attention_hmx_decode(
      cfg, query, key, value, key_cache, value_cache, slot_mapping, output,
      block_table, query_start_loc, seq_lens, num_reqs, num_heads,
      num_kv_heads, head_dim, block_size, max_blocks_per_seq, causal != 0,
      scale);
}

QAIC_KERNEL_API int32_t multinsp_multithreaded_paged_attention_hmx_decode_cache(
    const AicJitEntryPointConfig* cfg, const AicJitPointerArray* ptrs) {
  const float16* query = (const float16*)ptrs->pointers[0];
  float16* key_cache = (float16*)ptrs->pointers[1];
  float16* value_cache = (float16*)ptrs->pointers[2];
  float16* output = (float16*)ptrs->pointers[3];
  const int32_t* block_table = (const int32_t*)ptrs->pointers[4];
  const int32_t* query_start_loc = (const int32_t*)ptrs->pointers[5];
  const int32_t* seq_lens = (const int32_t*)ptrs->pointers[6];
  const int32_t num_reqs = *(const int32_t*)ptrs->pointers[7];
  const int32_t num_tokens = *(const int32_t*)ptrs->pointers[8];
  const int32_t num_heads = *(const int32_t*)ptrs->pointers[9];
  const int32_t num_kv_heads = *(const int32_t*)ptrs->pointers[10];
  const int32_t head_dim = *(const int32_t*)ptrs->pointers[11];
  const int32_t block_size = *(const int32_t*)ptrs->pointers[12];
  const int32_t max_blocks_per_seq = *(const int32_t*)ptrs->pointers[13];
  const int32_t causal = *(const int32_t*)ptrs->pointers[14];
  const float scale = *(const float*)ptrs->pointers[15];

  if (!all_requests_are_decode(query_start_loc, num_reqs) ||
      num_tokens != num_reqs) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  return paged_attention_hmx_decode(
      cfg, query, NULL, NULL, key_cache, value_cache, NULL, output,
      block_table, query_start_loc, seq_lens, num_reqs, num_heads,
      num_kv_heads, head_dim, block_size, max_blocks_per_seq, causal != 0,
      scale);
}

QAIC_KERNEL_API int32_t multinsp_multithreaded_paged_attention(
    const AicJitEntryPointConfig* cfg, const AicJitPointerArray* ptrs) {
  const float16* query = (const float16*)ptrs->pointers[0];
  const float16* key_cache = (const float16*)ptrs->pointers[1];
  const float16* value_cache = (const float16*)ptrs->pointers[2];
  float16* output = (float16*)ptrs->pointers[3];
  const int32_t* block_table = (const int32_t*)ptrs->pointers[4];
  const int32_t* query_start_loc = (const int32_t*)ptrs->pointers[5];
  const int32_t* seq_lens = (const int32_t*)ptrs->pointers[6];
  const int32_t num_reqs = *(const int32_t*)ptrs->pointers[7];
  const int32_t num_tokens = *(const int32_t*)ptrs->pointers[8];
  const int32_t num_heads = *(const int32_t*)ptrs->pointers[9];
  const int32_t num_kv_heads = *(const int32_t*)ptrs->pointers[10];
  const int32_t head_dim = *(const int32_t*)ptrs->pointers[11];
  const int32_t block_size = *(const int32_t*)ptrs->pointers[12];
  const int32_t max_blocks_per_seq = *(const int32_t*)ptrs->pointers[13];
  const int32_t causal = *(const int32_t*)ptrs->pointers[14];
  const float scale = *(const float*)ptrs->pointers[15];

  if (head_dim > kMaxHeadDim || num_kv_heads <= 0 ||
      num_heads % num_kv_heads != 0) {
    return JIT_DEV_STATUS_SUCCESS;
  }

  const int32_t gqa = num_heads / num_kv_heads;
  const int32_t total_rows = num_tokens * num_heads;
  const bool decode_order = all_requests_are_decode(query_start_loc, num_reqs);
  int64_t hmx_thread_id = QAIC_V68_HVX_THREADS_COUNT;
  qshimQuery(DEV_ATTR_QSHIM_HMX_THREAD_ID, &hmx_thread_id);

  const bool hmx_thread_is_launched =
      hmx_thread_id >= 0 && hmx_thread_id < (int64_t)cfg->numThreads;
  if (hmx_thread_is_launched && cfg->threadID == (uint32_t)hmx_thread_id) {
    return JIT_DEV_STATUS_SUCCESS;
  }

  const int32_t hvx_threads_per_core =
      cfg->numThreads - (hmx_thread_is_launched ? 1 : 0);
  int32_t local_hvx_thread = cfg->threadID;
  if (hmx_thread_is_launched && cfg->threadID > (uint32_t)hmx_thread_id) {
    --local_hvx_thread;
  }
  const int32_t wid = cfg->coreID * hvx_threads_per_core + local_hvx_thread;
  const int32_t workers = cfg->numCores * hvx_threads_per_core;
  const int32_t chunk = ceil_div_i32(total_rows, workers);
  const int32_t start = wid * chunk;
  const int32_t end = start + chunk < total_rows ? start + chunk : total_rows;

  float acc[kMaxHeadDim] __attribute__((aligned(HVX_VectorSize)));

  for (int32_t row = start; row < end; ++row) {
    int32_t req = -1;
    int32_t token = 0;
    int32_t head = 0;
    int32_t kv_head = 0;
    int32_t local_q = 0;
    const bool mapped = decode_order
                            ? map_decode_ordered_row(row, query_start_loc,
                                                     num_reqs, gqa, &req,
                                                     &token, &head, &kv_head,
                                                     &local_q)
                            : map_prefill_ordered_row(row, query_start_loc,
                                                      num_reqs, num_kv_heads,
                                                      gqa, &req, &token,
                                                      &head, &kv_head,
                                                      &local_q);
    if (!mapped) {
      continue;
    }

    const int32_t req_token_start = query_start_loc[req];
    const int32_t req_token_end = query_start_loc[req + 1];
    const int32_t q_len = req_token_end - req_token_start;
    const int32_t seq_len = seq_lens[req];
    const int32_t query_pos = seq_len - q_len + local_q;
    int32_t kv_limit = causal ? (query_pos + 1) : seq_len;
    if (kv_limit > seq_len) {
      kv_limit = seq_len;
    }
    if (kv_limit < 0) {
      kv_limit = 0;
    }

    const float16* q_row =
        query + flat_q_offset(token, head, 0, num_heads, head_dim);
    zero_accumulator(acc, head_dim);
    float max_score = -FLT_MAX;
    float denom = 0.0f;

    for (int32_t kv_pos = 0; kv_pos < kv_limit; ++kv_pos) {
      const int32_t logical_block = kv_pos / block_size;
      const int32_t block_offset = kv_pos - logical_block * block_size;
      const int32_t block_id =
          block_table[req * max_blocks_per_seq + logical_block];

      const float16* k_row =
          key_cache + cache_offset(block_id, kv_head, block_offset, 0,
                                   num_kv_heads, block_size, head_dim);
      const float dot = dot_f16_f32_hvx(q_row, k_row, head_dim);
      const float score = dot * scale;
      const float new_max = score > max_score ? score : max_score;
      const float rescale = expf(max_score - new_max);
      const float p = expf(score - new_max);

      if (rescale != 1.0f) {
        scale_accumulator_hvx(acc, head_dim, rescale);
      }
      denom = denom * rescale + p;
      max_score = new_max;

      const float16* v_row =
          value_cache + cache_offset(block_id, kv_head, block_offset, 0,
                                     num_kv_heads, block_size, head_dim);
      accumulate_value_hvx(acc, v_row, head_dim, p);
    }

    const float inv_denom = denom == 0.0f ? 0.0f : 1.0f / denom;
    float16* out_row =
        output + flat_q_offset(token, head, 0, num_heads, head_dim);
    store_normalized_hvx(out_row, acc, head_dim, inv_denom);
  }

  return JIT_DEV_STATUS_SUCCESS;
}