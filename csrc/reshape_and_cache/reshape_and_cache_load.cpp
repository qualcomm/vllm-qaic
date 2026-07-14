// ---------------------------------------------------------------------------------------
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
// ---------------------------------------------------------------------------------------

#include "QAicHexagonHVX.h"
#include "QAicHexagonTypes.h"
#include "QAicHexagonUtils.h"
#include "hexagon_protos.h"
#include "hexagon_types.h"
#include "jit_qshim_api.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>

void kv_cache_blocked_to_flat_hvx(
    float16* key,          // [num_tokens, num_heads_k, head_dim]
    float16* value,        // [num_tokens, num_heads_v, head_dim]
    float16* key_cache,    // [num_blocks, num_heads_k, block_size, head_dim]
    float16* value_cache,  // [num_blocks, num_heads_v, block_size, head_dim]
    int32_t* buff, int32_t* buffer,
    const int64_t* slot_mapping,  // [num_tokens]
    int32_t seqlen, int32_t block_size, int32_t nheads_k, int32_t headdim,
    int32_t element_bytes) {
  const int token_bytes = nheads_k * headdim * element_bytes;

  float16 const* k_in = (float16 const*)key;
  float16 const* v_in = (float16 const*)value;
  float16* k_base = (float16*)key_cache;
  float16* v_base = (float16*)value_cache;
  int32_t* buff_base = (int32_t*)buff;
  int64_t* buff_base_a;  // = (int64_t*)buffer; // Assuming buff has enough
                         // space for this

  const int64_t* block_table = (const int64_t*)slot_mapping;

  int t = 0;
  int32_t slot_idx = 0;
  int32_t block_id = slot_idx / block_size;
  int32_t block_offset = slot_idx % block_size;
  uint32_t headdim_size = headdim;

  int32_t src_head_size = headdim_size;
  int32_t src_kv_head_stride = headdim;
  int32_t src_token_stride = nheads_k * headdim;

  int32_t dst_head_stride = headdim * block_size;
  int32_t dst_within_block_stride = block_size;

  uint32_t block_stride = nheads_k * block_size * headdim;
  uint32_t src_key_offset = 0;
  uint32_t dest_key_offset = 0;
  uint32_t src_value_offset = 0;
  uint32_t dest_value_offset = 0;
  int counter = 0;
  int counter_a = 0;
  uint32_t inheaddim = 0;
  uint32_t tok = 0;
  uint32_t hid = 0;
  int32_t dst_kv_head_stride = headdim;
  int32_t dst_token_stride = nheads_k * headdim;
  int32_t src_head_stride = headdim * block_size;
  int32_t src_within_block_stride = block_size;
  while (tok < (uint32_t)seqlen) {
    slot_idx = (int32_t)(*((int64_t*)slot_mapping + (int32_t)tok));
    buff_base_a = (int64_t*)slot_mapping;
    slot_idx = *((int32_t*)buff_base_a + tok);
    block_id = slot_idx / block_size;
    block_offset = slot_idx % block_size;
    while (hid < (uint32_t)nheads_k) {
      while (inheaddim < headdim_size) {
        dest_value_offset =
            dst_token_stride * tok + dst_kv_head_stride * hid + inheaddim;
        dest_key_offset =
            dst_token_stride * tok + dst_kv_head_stride * hid + inheaddim;

        src_value_offset = block_stride * block_id + src_head_stride * hid +
                           block_offset * dst_kv_head_stride + inheaddim;

        src_key_offset = block_stride * block_id + src_head_stride * hid +
                         src_within_block_stride * inheaddim + block_offset;

        key[dest_key_offset] = key_cache[src_key_offset];
        value[dest_value_offset] = value_cache[src_value_offset];

        buff_base[counter] = dest_value_offset;
        if (value_cache == key_cache) {
          buffer[counter] = 1;
        } else {
          buffer[counter] = src_value_offset;
        }
        // buffer[counter] = dest_value_offset;
        counter++;
        // inheaddim = (int32_t)inheaddim + 1;
        inheaddim++;
      }
      inheaddim = 0;
      hid = (int32_t)hid + 1;
    }
    hid = 0;
    tok = (int32_t)tok + 1;
  }
  tok = 0;
}

QAIC_KERNEL_API int32_t multinsp_multithread_kv_cache_blocked_to_flat_hvx_f(
    const AicJitEntryPointConfig* entryConfig,
    const AicJitPointerArray* pointerArray) {
  int32_t const numel = (int32_t)*(int32_t*)pointerArray->pointers[11];
  int32_t const numelPerCore =
      (numel + entryConfig->numCores - 1) / entryConfig->numCores;
  int32_t const numelPerThread =
      (numelPerCore + entryConfig->numThreads - 1) / entryConfig->numThreads;
  int32_t const offset = entryConfig->threadID * numelPerThread +
                         entryConfig->coreID * numelPerCore;

  float16* key = (float16*)pointerArray->pointers[0];
  float16* value = (float16*)pointerArray->pointers[1];
  float16* key_cache = (float16*)pointerArray->pointers[2];
  float16* value_cache = (float16*)pointerArray->pointers[3];
  int32_t* buff = (int32_t*)pointerArray->pointers[4];
  int32_t* buffer = (int32_t*)pointerArray->pointers[5];
  int64_t* slot_mapping = (int64_t*)pointerArray->pointers[6];
  int32_t seqlen = *(const int32_t*)pointerArray->pointers[7];
  int32_t block_size = *(const int32_t*)pointerArray->pointers[8];
  int32_t nheads_k = *(const int32_t*)pointerArray->pointers[9];
  int32_t headdim = *(const int32_t*)pointerArray->pointers[10];
  int32_t element_bytes = *(const int32_t*)pointerArray->pointers[11];

  kv_cache_blocked_to_flat_hvx(key, value, key_cache, value_cache, buff, buffer,
                               slot_mapping, seqlen, block_size, nheads_k,
                               headdim, element_bytes);

  return JIT_DEV_STATUS_SUCCESS;
}
