// ---------------------------------------------------------------------------------------
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
// ---------------------------------------------------------------------------------------

#pragma once

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "QAicHexagonActivations.h"
#include "QAicHexagonHVX.h"
#include "QAicHexagonMath.h"
#include "QAicHexagonReducer.h"
#include "QAicHexagonTypes.h"
#include "QAicHexagonUtils.h"
#include "hexagon_protos.h"
#include "hexagon_types.h"
#include "jit_dev_status_codes.h"
#include "jit_qshim_api.h"

namespace grouped_topk_router {

constexpr int32_t kMaxGroups = 128;
constexpr int32_t kMaxTopK = 64;
constexpr int32_t kMaxExperts = 1024;
constexpr float kNegInf = -3.4028234663852886e38F;

enum class ScoreMode : int32_t {
  kSoftmax = 0,
  kSoftmaxF32 = 1,
  kSigmoid = 2,
  kSigmoidF32 = 3,
};

// Convert the Python-facing score mode id into the kernel enum.
inline bool score_mode_from_id(int32_t score_mode_id, ScoreMode* score_mode) {
  switch (score_mode_id) {
    case 0:
      *score_mode = ScoreMode::kSoftmax;
      return true;
    case 1:
      *score_mode = ScoreMode::kSoftmaxF32;
      return true;
    case 2:
      *score_mode = ScoreMode::kSigmoid;
      return true;
    case 3:
      *score_mode = ScoreMode::kSigmoidF32;
      return true;
    default:
      return false;
  }
}

struct RouterTileParams {
  const float16* input;
  const float16* bias;
  float* topk_weights;
  int32_t* topk_ids;
  int32_t num_tokens;
  int32_t num_experts;
  int32_t num_groups;
  int32_t topk_group;
  int32_t topk;
  bool renormalize;
  float routed_scaling_factor;
  bool use_bias;
  ScoreMode score_mode;
  int32_t token_begin;
  int32_t token_end;
  float16* vtcm_scores;
  float16* vtcm_selection_scores;
};

// Integer ceiling division used for work partitioning.
inline int32_t ceil_div_i32(int32_t a, int32_t b) { return (a + b - 1) / b; }

// Round a byte count up to the requested power-of-two alignment.
inline uint64_t align_up_u64(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

// Compare value/index pairs with lower index as the deterministic tie-breaker.
inline bool better_pair(float value, int32_t index, float best_value,
                        int32_t best_index) {
  return (value > best_value) ||
         ((value == best_value) && (index < best_index));
}

inline uint32_t query_vtcm_size_by_two() {
  int64_t vtcm_size = 0;
  const uint32_t status = qshimQuery(DEV_ATTR_QSHIM_VTCM_SIZE, &vtcm_size);
  if (status == JIT_DEV_STATUS_SUCCESS && vtcm_size > 0) {
    return (uint32_t)vtcm_size / 2U;
  }
  return 4U * 1024U * 1024U;
}

// Scalar fp32 sigmoid helper.
inline float sigmoid_f32(float x) { return 0.5F * tanhf(0.5F * x) + 0.5F; }

// Vectorized fp16 sigmoid helper using HVX tanh.
inline HVX_Vector sigmoid_hf(HVX_Vector input_vhf) {
  static constexpr float16 one_hf = 1.0F;
  static constexpr float16 half_hf = 0.5F;
  const HVX_Vector one_vhf = Q6_Vh_vsplat_R(*(const uint16_t*)&one_hf);
  const HVX_Vector half_vhf = Q6_Vh_vsplat_R(*(const uint16_t*)&half_hf);
  const HVX_Vector half_input_vhf = Q6_Vhf_vmpy_VhfVhf(input_vhf, half_vhf);
  const HVX_Vector tanh_vhf = qaic_tanh_hf(half_input_vhf);
  const HVX_Vector shifted_vhf = Q6_Vhf_vadd_VhfVhf(tanh_vhf, one_vhf);
  return Q6_Vhf_vmpy_VhfVhf(shifted_vhf, half_vhf);
}

// Materialize fp16 softmax scores with HVX reducers and fp16 intermediates.
inline void materialize_softmax_scores_hvx_hf(const float16* token_input,
                                              float16* token_scores,
                                              int32_t num_experts) {
  constexpr int32_t kVecElts = HVX_VectorSize / sizeof(float16);
  const int32_t full_vec_elems = (num_experts / kVecElts) * kVecElts;
  const int32_t rem_elems = num_experts - full_vec_elems;

  MaxReducerFloat16 max_reducer;
  for (int32_t expert = 0; expert < full_vec_elems; expert += kVecElts) {
    max_reducer.reduce(
        LoadUnaligned<HVX_Vector>((const int8_t*)&token_input[expert]));
  }
  if (rem_elems > 0) {
    max_reducer.reduce(
        LoadUnaligned<HVX_Vector>((const int8_t*)&token_input[full_vec_elems],
                                  rem_elems * sizeof(float16)),
        rem_elems);
  }
  const HVX_Vector max_vec = max_reducer.finishSplat();

  DiffExpFloat16 diff_exp;
  SumReducerFloat16 sum_reducer;
  for (int32_t expert = 0; expert < full_vec_elems; expert += kVecElts) {
    const HVX_Vector in_vec =
        LoadUnaligned<HVX_Vector>((const int8_t*)&token_input[expert]);
    const HVX_Vector exp_vec = diff_exp(in_vec, max_vec);
    sum_reducer.reduce(exp_vec);
    StoreUnalignedHVX((int8_t*)&token_scores[expert], exp_vec);
  }
  if (rem_elems > 0) {
    const HVX_Vector in_vec =
        LoadUnaligned<HVX_Vector>((const int8_t*)&token_input[full_vec_elems],
                                  rem_elems * sizeof(float16));
    const HVX_Vector exp_vec = diff_exp(in_vec, max_vec);
    sum_reducer.reduce(exp_vec, rem_elems);
    StoreUnalignedHVX((int8_t*)&token_scores[full_vec_elems], exp_vec,
                      rem_elems * sizeof(float16));
  }

  float16 exp_sum_h = (float16)0.0F;
  sum_reducer.finish(&exp_sum_h);
  const float exp_sum = (float)exp_sum_h;
  if (exp_sum == 0.0F) {
    for (int32_t expert = 0; expert < num_experts; ++expert) {
      token_scores[expert] = (float16)0.0F;
    }
    return;
  }

  float16 inv_sum_h = (float16)(1.0F / exp_sum);
  NormalizerFloat16 normalizer(&inv_sum_h);
  for (int32_t expert = 0; expert < full_vec_elems; expert += kVecElts) {
    const HVX_Vector exp_vec =
        LoadUnaligned<HVX_Vector>((const int8_t*)&token_scores[expert]);
    const HVX_Vector norm_vec = normalizer.normalize(exp_vec);
    StoreUnalignedHVX((int8_t*)&token_scores[expert], norm_vec);
  }
  if (rem_elems > 0) {
    const HVX_Vector exp_vec =
        LoadUnaligned<HVX_Vector>((const int8_t*)&token_scores[full_vec_elems],
                                  rem_elems * sizeof(float16));
    const HVX_Vector norm_vec = normalizer.normalize(exp_vec);
    StoreUnalignedHVX((int8_t*)&token_scores[full_vec_elems], norm_vec,
                      rem_elems * sizeof(float16));
  }
}

// Materialize fp16 sigmoid scores with HVX vector math.
inline void materialize_sigmoid_scores_hvx_hf(const float16* token_input,
                                              float16* token_scores,
                                              int32_t num_experts) {
  constexpr int32_t kVecElts = HVX_VectorSize / sizeof(float16);
  const int32_t full_vec_elems = (num_experts / kVecElts) * kVecElts;
  const int32_t rem_elems = num_experts - full_vec_elems;

  for (int32_t expert = 0; expert < full_vec_elems; expert += kVecElts) {
    const HVX_Vector in_vec =
        LoadUnaligned<HVX_Vector>((const int8_t*)&token_input[expert]);
    const HVX_Vector sigmoid_vec = sigmoid_hf(in_vec);
    StoreUnalignedHVX((int8_t*)&token_scores[expert], sigmoid_vec);
  }
  if (rem_elems > 0) {
    const HVX_Vector in_vec =
        LoadUnaligned<HVX_Vector>((const int8_t*)&token_input[full_vec_elems],
                                  rem_elems * sizeof(float16));
    const HVX_Vector sigmoid_vec = sigmoid_hf(in_vec);
    StoreUnalignedHVX((int8_t*)&token_scores[full_vec_elems], sigmoid_vec,
                      rem_elems * sizeof(float16));
  }
}

// Scalar score materialization fallback for softmax/sigmoid modes.
inline void materialize_scores_scalar_hf(const float16* token_input,
                                         float16* token_scores,
                                         int32_t num_experts,
                                         ScoreMode score_mode) {
  if (score_mode == ScoreMode::kSoftmax) {
    float row_max = kNegInf;
    for (int32_t expert = 0; expert < num_experts; ++expert) {
      const float value = (float)token_input[expert];
      if (value > row_max) {
        row_max = value;
      }
    }

    float row_sum = 0.0F;
    for (int32_t expert = 0; expert < num_experts; ++expert) {
      const float score = expf((float)token_input[expert] - row_max);
      token_scores[expert] = (float16)score;
      row_sum += score;
    }
    if (row_sum == 0.0F) {
      row_sum = 1.0F;
    }
    const float inv_sum = 1.0F / row_sum;
    for (int32_t expert = 0; expert < num_experts; ++expert) {
      token_scores[expert] = (float16)((float)token_scores[expert] * inv_sum);
    }
  } else {
    for (int32_t expert = 0; expert < num_experts; ++expert) {
      token_scores[expert] = (float16)sigmoid_f32((float)token_input[expert]);
    }
  }
}

// Softmax computed entirely in fp32, result stored back as fp16.
inline void materialize_softmax_scores_f32_hf(const float16* token_input,
                                              float16* token_scores,
                                              int32_t num_experts) {
  constexpr int32_t kVecEltsF16 = HVX_VectorSize / sizeof(float16);
  constexpr int32_t kVecEltsSF = HVX_VectorSize / sizeof(float);
  const int32_t full_vec_elems = (num_experts / kVecEltsF16) * kVecEltsF16;
  const int32_t rem_elems = num_experts - full_vec_elems;

  // ---- pass 1: find max in fp32 ----
  MaxReducerFloat max_reducer;
  for (int32_t e = 0; e < full_vec_elems; e += kVecEltsF16) {
    const HVX_Vector in_hf =
        LoadUnaligned<HVX_Vector>((const int8_t*)&token_input[e]);
    const HVX_VectorPair in_sf = Q6_Wsf_vcvt_Vhf(in_hf);
    max_reducer.reduce(Q6_V_lo_W(in_sf));
    max_reducer.reduce(Q6_V_hi_W(in_sf));
  }
  if (rem_elems > 0) {
    const HVX_Vector in_hf =
        LoadUnaligned<HVX_Vector>((const int8_t*)&token_input[full_vec_elems],
                                  rem_elems * sizeof(float16));
    const HVX_VectorPair in_sf = Q6_Wsf_vcvt_Vhf(in_hf);
    const int32_t rem_lo = rem_elems < kVecEltsSF ? rem_elems : kVecEltsSF;
    const int32_t rem_hi = rem_elems > kVecEltsSF ? rem_elems - kVecEltsSF : 0;
    max_reducer.reduce(Q6_V_lo_W(in_sf), rem_lo);
    if (rem_hi > 0) max_reducer.reduce(Q6_V_hi_W(in_sf), rem_hi);
  }
  const HVX_Vector max_vsf = max_reducer.finishSplat();

  // ---- pass 2: exp(x - max) in fp32, accumulate sum ----
  DiffExpFloat diff_exp;
  SumReducerFloat sum_reducer;
  // We need to store fp32 temporarily; use a scratch array on stack.
  // max experts = 1024 => 1024 * 4 = 4KB, acceptable for NSP stack.
  float exp_scratch[kMaxExperts] __attribute__((aligned(HVX_VectorSize)));

  for (int32_t e = 0; e < full_vec_elems; e += kVecEltsF16) {
    const HVX_Vector in_hf =
        LoadUnaligned<HVX_Vector>((const int8_t*)&token_input[e]);
    const HVX_VectorPair in_sf = Q6_Wsf_vcvt_Vhf(in_hf);
    const HVX_Vector exp_lo = diff_exp(Q6_V_lo_W(in_sf), max_vsf);
    const HVX_Vector exp_hi = diff_exp(Q6_V_hi_W(in_sf), max_vsf);
    sum_reducer.reduce(exp_lo);
    sum_reducer.reduce(exp_hi);
    StoreUnalignedHVX((int8_t*)&exp_scratch[e], exp_lo);
    StoreUnalignedHVX((int8_t*)&exp_scratch[e + kVecEltsSF], exp_hi);
  }
  if (rem_elems > 0) {
    const HVX_Vector in_hf =
        LoadUnaligned<HVX_Vector>((const int8_t*)&token_input[full_vec_elems],
                                  rem_elems * sizeof(float16));
    const HVX_VectorPair in_sf = Q6_Wsf_vcvt_Vhf(in_hf);
    const int32_t rem_lo = rem_elems < kVecEltsSF ? rem_elems : kVecEltsSF;
    const int32_t rem_hi = rem_elems > kVecEltsSF ? rem_elems - kVecEltsSF : 0;
    const HVX_Vector exp_lo = diff_exp(Q6_V_lo_W(in_sf), max_vsf);
    sum_reducer.reduce(exp_lo, rem_lo);
    StoreUnalignedHVX((int8_t*)&exp_scratch[full_vec_elems], exp_lo,
                      rem_lo * sizeof(float));
    if (rem_hi > 0) {
      const HVX_Vector exp_hi = diff_exp(Q6_V_hi_W(in_sf), max_vsf);
      sum_reducer.reduce(exp_hi, rem_hi);
      StoreUnalignedHVX((int8_t*)&exp_scratch[full_vec_elems + kVecEltsSF],
                        exp_hi, rem_hi * sizeof(float));
    }
  }

  const HVX_Vector sum_vsf = sum_reducer.finishSplat();
  // finishSplat splats the reduced sum to all lanes; extract lane 0.
  float sum_arr[kVecEltsSF] __attribute__((aligned(HVX_VectorSize)));
  StoreHVX(&sum_arr[0], sum_vsf);
  const float exp_sum = sum_arr[0];
  if (exp_sum == 0.0F) {
    for (int32_t e = 0; e < num_experts; ++e) token_scores[e] = (float16)0.0F;
    return;
  }
  const float inv_sum = 1.0F / exp_sum;
  float inv_sum_f = inv_sum;
  NormalizerFloat normalizer(&inv_sum_f);

  // ---- pass 3: normalize fp32 exp values, demote to fp16 ----
  for (int32_t e = 0; e < full_vec_elems; e += kVecEltsF16) {
    const HVX_Vector exp_lo =
        LoadUnaligned<HVX_Vector>((const int8_t*)&exp_scratch[e]);
    const HVX_Vector exp_hi =
        LoadUnaligned<HVX_Vector>((const int8_t*)&exp_scratch[e + kVecEltsSF]);
    const HVX_Vector norm_lo = normalizer.normalize(exp_lo);
    const HVX_Vector norm_hi = normalizer.normalize(exp_hi);
    const HVX_Vector result_hf = Q6_Vhf_vcvt_VsfVsf(norm_lo, norm_hi);
    StoreUnalignedHVX((int8_t*)&token_scores[e], result_hf);
  }
  if (rem_elems > 0) {
    const int32_t rem_lo = rem_elems < kVecEltsSF ? rem_elems : kVecEltsSF;
    const int32_t rem_hi = rem_elems > kVecEltsSF ? rem_elems - kVecEltsSF : 0;
    const HVX_Vector exp_lo = LoadUnaligned<HVX_Vector>(
        (const int8_t*)&exp_scratch[full_vec_elems], rem_lo * sizeof(float));
    HVX_Vector exp_hi = Q6_V_vzero();
    if (rem_hi > 0) {
      exp_hi = LoadUnaligned<HVX_Vector>(
          (const int8_t*)&exp_scratch[full_vec_elems + kVecEltsSF],
          rem_hi * sizeof(float));
    }
    const HVX_Vector norm_lo = normalizer.normalize(exp_lo);
    const HVX_Vector norm_hi = normalizer.normalize(exp_hi);
    const HVX_Vector result_hf = Q6_Vhf_vcvt_VsfVsf(norm_lo, norm_hi);
    StoreUnalignedHVX((int8_t*)&token_scores[full_vec_elems], result_hf,
                      rem_elems * sizeof(float16));
  }
}

inline void materialize_sigmoid_scores_f32_hf(const float16* token_input,
                                              float16* token_scores,
                                              int32_t num_experts) {
  constexpr int32_t kVecEltsF16 = HVX_VectorSize / sizeof(float16);
  constexpr int32_t kVecEltsSF = HVX_VectorSize / sizeof(float);
  const int32_t full_vec_elems = (num_experts / kVecEltsF16) * kVecEltsF16;
  const int32_t rem_elems = num_experts - full_vec_elems;

  // half_sf and one_sf constants
  static constexpr float kHalfF = 0.5F;
  static constexpr float kOneF = 1.0F;
  const HVX_Vector half_vsf = Q6_V_vsplat_R(*(const uint32_t*)&kHalfF);
  const HVX_Vector one_vsf = Q6_V_vsplat_R(*(const uint32_t*)&kOneF);

  for (int32_t e = 0; e < full_vec_elems; e += kVecEltsF16) {
    const HVX_Vector in_hf =
        LoadUnaligned<HVX_Vector>((const int8_t*)&token_input[e]);
    const HVX_VectorPair in_sf = Q6_Wsf_vcvt_Vhf(in_hf);
    // sigmoid(x) = 0.5 * tanh(0.5 * x) + 0.5
    const HVX_Vector half_x_lo = Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(in_sf), half_vsf);
    const HVX_Vector half_x_hi = Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(in_sf), half_vsf);
    const HVX_Vector tanh_lo = qaic_tanh_sf(half_x_lo);
    const HVX_Vector tanh_hi = qaic_tanh_sf(half_x_hi);
    const HVX_Vector sig_lo =
        Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(tanh_lo, half_vsf), half_vsf);
    const HVX_Vector sig_hi =
        Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(tanh_hi, half_vsf), half_vsf);
    const HVX_Vector result_hf = Q6_Vhf_vcvt_VsfVsf(sig_lo, sig_hi);
    StoreUnalignedHVX((int8_t*)&token_scores[e], result_hf);
  }
  if (rem_elems > 0) {
    const HVX_Vector in_hf =
        LoadUnaligned<HVX_Vector>((const int8_t*)&token_input[full_vec_elems],
                                  rem_elems * sizeof(float16));
    const HVX_VectorPair in_sf = Q6_Wsf_vcvt_Vhf(in_hf);
    const HVX_Vector half_x_lo = Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(in_sf), half_vsf);
    const HVX_Vector half_x_hi = Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(in_sf), half_vsf);
    const HVX_Vector tanh_lo = qaic_tanh_sf(half_x_lo);
    const HVX_Vector tanh_hi = qaic_tanh_sf(half_x_hi);
    const HVX_Vector sig_lo =
        Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(tanh_lo, half_vsf), half_vsf);
    const HVX_Vector sig_hi =
        Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(tanh_hi, half_vsf), half_vsf);
    const HVX_Vector result_hf = Q6_Vhf_vcvt_VsfVsf(sig_lo, sig_hi);
    StoreUnalignedHVX((int8_t*)&token_scores[full_vec_elems], result_hf,
                      rem_elems * sizeof(float16));
  }
}

// Dispatch score materialization to the requested score mode implementation.
inline void materialize_scores_hf(const float16* token_input,
                                  float16* token_scores, int32_t num_experts,
                                  ScoreMode score_mode) {
  if (score_mode == ScoreMode::kSoftmax) {
    materialize_softmax_scores_hvx_hf(token_input, token_scores, num_experts);
  } else if (score_mode == ScoreMode::kSigmoid) {
    materialize_sigmoid_scores_hvx_hf(token_input, token_scores, num_experts);
  } else if (score_mode == ScoreMode::kSoftmaxF32) {
    materialize_softmax_scores_f32_hf(token_input, token_scores, num_experts);
  } else {
    materialize_sigmoid_scores_f32_hf(token_input, token_scores, num_experts);
  }
}

// Build biased/unbiased selection scores in fp16 scratch.
inline void materialize_selection_scores_hvx_hf(const float16* token_scores,
                                                const float16* bias,
                                                float16* selection_scores,
                                                int32_t num_experts,
                                                bool use_bias) {
  constexpr int32_t kVecElts = HVX_VectorSize / sizeof(float16);
  const int32_t full_vec_elems = (num_experts / kVecElts) * kVecElts;
  const int32_t rem_elems = num_experts - full_vec_elems;

  for (int32_t expert = 0; expert < full_vec_elems; expert += kVecElts) {
    const HVX_Vector score_vec =
        LoadUnaligned<HVX_Vector>((const int8_t*)&token_scores[expert]);
    HVX_Vector selection_vec = score_vec;
    if (use_bias) {
      const HVX_Vector bias_vec =
          LoadUnaligned<HVX_Vector>((const int8_t*)&bias[expert]);
      selection_vec = Q6_Vhf_vadd_VhfVhf(score_vec, bias_vec);
    }
    StoreUnalignedHVX((int8_t*)&selection_scores[expert], selection_vec);
  }
  if (rem_elems > 0) {
    const HVX_Vector score_vec =
        LoadUnaligned<HVX_Vector>((const int8_t*)&token_scores[full_vec_elems],
                                  rem_elems * sizeof(float16));
    HVX_Vector selection_vec = score_vec;
    if (use_bias) {
      const HVX_Vector bias_vec = LoadUnaligned<HVX_Vector>(
          (const int8_t*)&bias[full_vec_elems], rem_elems * sizeof(float16));
      selection_vec = Q6_Vhf_vadd_VhfVhf(score_vec, bias_vec);
    }
    StoreUnalignedHVX((int8_t*)&selection_scores[full_vec_elems], selection_vec,
                      rem_elems * sizeof(float16));
  }
}

// Read a pre-materialized selection score as fp32.
inline float selection_score_at(const float16* selection_scores,
                                int32_t expert) {
  return (float)selection_scores[expert];
}

// Read either biased or unbiased selection score for one expert.
inline float selection_score_at(const float16* token_scores,
                                const float16* bias, int32_t expert,
                                bool use_bias) {
  float value = (float)token_scores[expert];
  if (use_bias) {
    value = (float)((float16)(value + (float)bias[expert]));
  }
  return value;
}

// Clear the small bitset used to track selected experts.
inline void clear_selected_experts(uint64_t* selected_words) {
  for (int32_t i = 0; i < (kMaxExperts + 63) / 64; ++i) {
    selected_words[i] = 0ULL;
  }
}

// Test whether an expert id is already selected in the bitset.
inline bool expert_selected(const uint64_t* selected_words, int32_t expert) {
  const uint64_t bit = 1ULL << (expert & 63);
  return (selected_words[expert >> 6] & bit) != 0ULL;
}

// Mark an expert id as selected in the bitset.
inline void mark_expert_selected(uint64_t* selected_words, int32_t expert) {
  selected_words[expert >> 6] |= 1ULL << (expert & 63);
}

// Compute max fp16 selection score over a contiguous expert span with HVX.
inline float max_score_hvx_hf(const float16* selection_scores, int32_t begin,
                              int32_t count) {
  if (count <= 0) {
    return kNegInf;
  }

  constexpr int32_t kVecElts = HVX_VectorSize / sizeof(float16);
  const int32_t full_vec_elems = (count / kVecElts) * kVecElts;
  const int32_t rem_elems = count - full_vec_elems;

  MaxReducerFloat16 max_reducer;
  for (int32_t i = 0; i < full_vec_elems; i += kVecElts) {
    max_reducer.reduce(
        LoadUnaligned<HVX_Vector>((const int8_t*)&selection_scores[begin + i]));
  }
  if (rem_elems > 0) {
    max_reducer.reduce(
        LoadUnaligned<HVX_Vector>(
            (const int8_t*)&selection_scores[begin + full_vec_elems],
            rem_elems * sizeof(float16)),
        rem_elems);
  }

  float16 max_vec_h[HVX_VectorSize / sizeof(float16)]
      __attribute__((aligned(HVX_VectorSize)));
  StoreHVX(&max_vec_h[0], max_reducer.finishSplat());
  return (float)max_vec_h[0];
}

// Compute max fp32 score over a contiguous span with HVX.
inline float max_score_hvx_f32(const float* scores, int32_t count) {
  if (count <= 0) {
    return kNegInf;
  }

  constexpr int32_t kVecElts = HVX_VectorSize / sizeof(float);
  const int32_t full_vec_elems = (count / kVecElts) * kVecElts;
  const int32_t rem_elems = count - full_vec_elems;

  MaxReducerFloat max_reducer;
  for (int32_t i = 0; i < full_vec_elems; i += kVecElts) {
    max_reducer.reduce(LoadUnaligned<HVX_Vector>((const int8_t*)&scores[i]));
  }
  if (rem_elems > 0) {
    max_reducer.reduce(
        LoadUnaligned<HVX_Vector>((const int8_t*)&scores[full_vec_elems],
                                  rem_elems * sizeof(float)),
        rem_elems);
  }

  float max_vec_f[HVX_VectorSize / sizeof(float)]
      __attribute__((aligned(HVX_VectorSize)));
  StoreHVX(&max_vec_f[0], max_reducer.finishSplat());
  return max_vec_f[0];
}

// Return an HVX vector splatted with negative fp16 max value.
inline HVX_Vector neg_max_hf_vec() {
  const float16 neg_max_h = (float16)-65504.0F;
  return Q6_Vh_vsplat_R(*(const uint16_t*)&neg_max_h);
}

// Merge two candidate vectors into top-1 and top-2 vectors lane-wise.
inline void merge_top2_vectors_hf(HVX_Vector candidate1, HVX_Vector candidate2,
                                  HVX_Vector& best1, HVX_Vector& best2) {
  const HVX_Vector new_best1 = Q6_Vhf_vfmax_VhfVhf(best1, candidate1);
  const HVX_Vector lower_best1 = Q6_Vhf_vfmin_VhfVhf(best1, candidate1);
  const HVX_Vector best2_candidates = Q6_Vhf_vfmax_VhfVhf(best2, candidate2);
  best2 = Q6_Vhf_vfmax_VhfVhf(best2_candidates, lower_best1);
  best1 = new_best1;
}

// Compute the sum of top-2 fp16 scores in an expert span using HVX.
inline float top2_sum_score_hvx_hf(const float16* selection_scores,
                                   int32_t begin, int32_t end) {
  const int32_t count = end - begin;
  if (count <= 0) {
    return kNegInf;
  }

  constexpr int32_t kVecElts = HVX_VectorSize / sizeof(float16);
  const HVX_Vector neg_vec = neg_max_hf_vec();
  HVX_Vector best1 = neg_vec;
  HVX_Vector best2 = neg_vec;

  int32_t offset = 0;
  for (; offset + kVecElts <= count; offset += kVecElts) {
    const HVX_Vector values = LoadUnaligned<HVX_Vector>(
        (const int8_t*)&selection_scores[begin + offset]);
    merge_top2_vectors_hf(values, neg_vec, best1, best2);
  }
  if (offset < count) {
    const int32_t rem_elems = count - offset;
    const HVX_Vector values = LoadUnaligned<HVX_Vector>(
        (const int8_t*)&selection_scores[begin + offset],
        rem_elems * sizeof(float16));
    const HVX_VectorPred pred = Q6_Q_vsetq2_R(rem_elems * sizeof(float16));
    const HVX_Vector padded_values = Q6_V_vmux_QVV(pred, values, neg_vec);
    merge_top2_vectors_hf(padded_values, neg_vec, best1, best2);
  }

  for (int32_t shift = kVecElts / 2; shift > 0; shift >>= 1) {
    const HVX_Vector other1 = Q6_V_vror_VR(best1, shift * sizeof(float16));
    const HVX_Vector other2 = Q6_V_vror_VR(best2, shift * sizeof(float16));
    merge_top2_vectors_hf(other1, other2, best1, best2);
  }

  float16 best1_h[HVX_VectorSize / sizeof(float16)]
      __attribute__((aligned(HVX_VectorSize)));
  float16 best2_h[HVX_VectorSize / sizeof(float16)]
      __attribute__((aligned(HVX_VectorSize)));
  StoreHVX(&best1_h[0], best1);
  StoreHVX(&best2_h[0], best2);
  const float top1 = (float)best1_h[0];
  const float top2 = count <= 1 ? top1 : (float)best2_h[0];
  return top1 + top2;
}

// Compute max of token score plus bias over a span using HVX.
inline float max_score_with_bias_hvx_hf(const float16* token_scores,
                                        const float16* bias, int32_t begin,
                                        int32_t count) {
  if (count <= 0) {
    return kNegInf;
  }

  constexpr int32_t kVecElts = HVX_VectorSize / sizeof(float16);
  const int32_t full_vec_elems = (count / kVecElts) * kVecElts;
  const int32_t rem_elems = count - full_vec_elems;

  MaxReducerFloat16 max_reducer;
  for (int32_t i = 0; i < full_vec_elems; i += kVecElts) {
    const HVX_Vector score_vec =
        LoadUnaligned<HVX_Vector>((const int8_t*)&token_scores[begin + i]);
    const HVX_Vector bias_vec =
        LoadUnaligned<HVX_Vector>((const int8_t*)&bias[begin + i]);
    max_reducer.reduce(Q6_Vhf_vadd_VhfVhf(score_vec, bias_vec));
  }
  if (rem_elems > 0) {
    const HVX_Vector score_vec = LoadUnaligned<HVX_Vector>(
        (const int8_t*)&token_scores[begin + full_vec_elems],
        rem_elems * sizeof(float16));
    const HVX_Vector bias_vec =
        LoadUnaligned<HVX_Vector>((const int8_t*)&bias[begin + full_vec_elems],
                                  rem_elems * sizeof(float16));
    max_reducer.reduce(Q6_Vhf_vadd_VhfVhf(score_vec, bias_vec), rem_elems);
  }

  float16 max_vec_h[HVX_VectorSize / sizeof(float16)]
      __attribute__((aligned(HVX_VectorSize)));
  StoreHVX(&max_vec_h[0], max_reducer.finishSplat());
  return (float)max_vec_h[0];
}

// Scalar top-2 sum over precomputed selection scores.
inline float top2_sum_score_hf(const float16* selection_scores, int32_t begin,
                               int32_t end) {
  const int32_t count = end - begin;
  if (count <= 0) {
    return kNegInf;
  }

  float best1 = kNegInf;
  float best2 = kNegInf;
  int32_t best1_id = begin;
  int32_t best2_id = begin;
  for (int32_t expert = begin; expert < end; ++expert) {
    const float value = selection_score_at(selection_scores, expert);
    if (better_pair(value, expert, best1, best1_id)) {
      best2 = best1;
      best2_id = best1_id;
      best1 = value;
      best1_id = expert;
    } else if (better_pair(value, expert, best2, best2_id)) {
      best2 = value;
      best2_id = expert;
    }
  }
  if (count <= 1) {
    return best1 + best1;
  }
  return best1 + best2;
}

// Scalar top-2 sum over token scores plus bias.
inline float top2_sum_score_with_bias_hf(const float16* token_scores,
                                         const float16* bias, int32_t begin,
                                         int32_t end) {
  const int32_t count = end - begin;
  if (count <= 0) {
    return kNegInf;
  }

  float best1 = kNegInf;
  float best2 = kNegInf;
  int32_t best1_id = begin;
  int32_t best2_id = begin;
  for (int32_t expert = begin; expert < end; ++expert) {
    const float value = selection_score_at(token_scores, bias, expert, true);
    if (better_pair(value, expert, best1, best1_id)) {
      best2 = best1;
      best2_id = best1_id;
      best1 = value;
      best1_id = expert;
    } else if (better_pair(value, expert, best2, best2_id)) {
      best2 = value;
      best2_id = expert;
    }
  }
  if (count <= 1) {
    return best1 + best1;
  }
  return best1 + best2;
}

// Scalar select of top groups by group score.
inline void select_top_groups(float* group_scores, int32_t num_groups,
                              int32_t topk_group, int32_t* selected_groups) {
  for (int32_t rank = 0; rank < topk_group; ++rank) {
    float best_value = kNegInf;
    int32_t best_group = 0;

    for (int32_t group = 0; group < num_groups; ++group) {
      const float value = group_scores[group];
      if (better_pair(value, group, best_value, best_group)) {
        best_value = value;
        best_group = group;
      }
    }

    selected_groups[rank] = best_group;
    group_scores[best_group] = kNegInf;
  }
}

// Find the first unmasked group matching a target score.
inline int32_t first_group_with_score(const float* group_scores,
                                      int32_t num_groups, float target_score) {
  for (int32_t group = 0; group < num_groups; ++group) {
    if (group_scores[group] == target_score) {
      return group;
    }
  }
  return 0;
}

// HVX-assisted top-group selection.
inline void select_top_groups_hvx(float* group_scores, int32_t num_groups,
                                  int32_t topk_group,
                                  int32_t* selected_groups) {
  for (int32_t rank = 0; rank < topk_group; ++rank) {
    const float best_value = max_score_hvx_f32(group_scores, num_groups);
    const int32_t best_group =
        first_group_with_score(group_scores, num_groups, best_value);
    selected_groups[rank] = best_group;
    group_scores[best_group] = kNegInf;
  }
}

// Find the first expert in a span matching a target fp16 score.
inline int32_t first_expert_with_score_hf(const float16* selection_scores,
                                          int32_t begin, int32_t end,
                                          float target_score) {
  for (int32_t expert = begin; expert < end; ++expert) {
    if (selection_score_at(selection_scores, expert) == target_score) {
      return expert;
    }
  }
  return begin;
}

// Produce shuffled even/odd vectors for one bitonic step.
inline void bitonic_sort_step_shuffle_hvx(unsigned step_idx,
                                          int32_t element_bytes,
                                          const HVX_Vector& values,
                                          HVX_Vector& shuffled_lo,
                                          HVX_Vector& shuffled_hi) {
  HVX_VectorPair pair =
      Q6_W_vdeal_VVR(values, values, element_bytes << step_idx);
  shuffled_lo = Q6_V_lo_W(pair);
  shuffled_hi = Q6_V_hi_W(pair);
}

inline HVX_VectorPred vcmp_gt_xacc_hf(const HVX_VectorPred& predicate,
                                      const HVX_Vector& v1,
                                      const HVX_Vector& v2) {
  const HVX_VectorPred gt_pred = Q6_Q_vcmp_gt_VhfVhf(v1, v2);
  return Q6_Q_xor_QQ(predicate, gt_pred);
}

inline HVX_VectorPred vcmp_gt_xacc_w(const HVX_VectorPred& predicate,
                                     const HVX_Vector& v1,
                                     const HVX_Vector& v2) {
  return Q6_Q_vcmp_gtxacc_QVwVw(predicate, v1, v2);
}

inline HVX_VectorPred bitonic_sort_step_swap_mask_hf(
    unsigned step_idx, const HVX_Vector& step_masks,
    const HVX_Vector& shuffled_lo, const HVX_Vector& shuffled_hi) {
  static constexpr uint32_t mask_bit_select[7] = {
      0x01010101, 0x02020202, 0x04040404, 0x08080808,
      0x10101010, 0x20202020, 0x40404040};
  const HVX_VectorPred comparison_reversal =
      Q6_Q_vand_VR(step_masks, mask_bit_select[step_idx]);
  return vcmp_gt_xacc_hf(comparison_reversal, shuffled_lo, shuffled_hi);
}

inline HVX_VectorPred bitonic_sort_step_swap_mask_w(
    unsigned step_idx, const HVX_Vector& step_masks,
    const HVX_Vector& shuffled_lo, const HVX_Vector& shuffled_hi) {
  static constexpr uint32_t mask_bit_select[7] = {
      0x01010101, 0x02020202, 0x04040404, 0x08080808,
      0x10101010, 0x20202020, 0x40404040};
  const HVX_VectorPred comparison_reversal =
      Q6_Q_vand_VR(step_masks, mask_bit_select[step_idx]);
  return vcmp_gt_xacc_w(comparison_reversal, shuffled_lo, shuffled_hi);
}

// Apply one bitonic sort step to fp16 values and int32 ids.
inline void bitonic_sort_step_hf_i32(unsigned step_idx,
                                     const HVX_Vector& step_masks_hf,
                                     const HVX_Vector step_masks_i32[2],
                                     HVX_Vector& vals, HVX_Vector idx[2]) {
  HVX_Vector idx_shuffle_even[2];
  HVX_Vector idx_shuffle_odd[2];
  if (step_idx < 5) {
    bitonic_sort_step_shuffle_hvx(step_idx, sizeof(int32_t), idx[0],
                                  idx_shuffle_even[0], idx_shuffle_odd[0]);
    bitonic_sort_step_shuffle_hvx(step_idx, sizeof(int32_t), idx[1],
                                  idx_shuffle_even[1], idx_shuffle_odd[1]);
  } else {
    idx_shuffle_even[0] = Q6_V_equals_V(idx[0]);
    idx_shuffle_odd[0] = Q6_V_equals_V(idx[1]);
    idx_shuffle_even[1] = Q6_V_equals_V(idx[0]);
    idx_shuffle_odd[1] = Q6_V_equals_V(idx[1]);
  }

  HVX_VectorPred half_idx_swap = bitonic_sort_step_swap_mask_w(
      step_idx, step_masks_i32[0], idx_shuffle_even[0], idx_shuffle_odd[0]);
  const HVX_Vector lower_half_idx_cmp = Q6_V_vand_QR(half_idx_swap, 0x01010101);
  half_idx_swap = bitonic_sort_step_swap_mask_w(
      step_idx, step_masks_i32[1], idx_shuffle_even[1], idx_shuffle_odd[1]);
  const HVX_Vector upper_half_idx_cmp = Q6_V_vand_QR(half_idx_swap, 0x01010101);
  const HVX_Vector idx_cmp_result =
      Q6_V_lo_W(Q6_W_vdeal_VVR(upper_half_idx_cmp, lower_half_idx_cmp, -2));

  HVX_Vector even_dup;
  HVX_Vector odd_dup;
  bitonic_sort_step_shuffle_hvx(step_idx, sizeof(float16), vals, even_dup,
                                odd_dup);
  const HVX_VectorPred val_swap_mask_pred = bitonic_sort_step_swap_mask_hf(
      step_idx, step_masks_hf, even_dup, odd_dup);
  HVX_Vector val_swap_mask = Q6_V_vand_QR(val_swap_mask_pred, 0x01010101);

  const HVX_VectorPred swap_mask_rev_pred = bitonic_sort_step_swap_mask_hf(
      step_idx, step_masks_hf, odd_dup, even_dup);
  const HVX_Vector vals_are_different = Q6_V_vand_QR(
      Q6_Q_xor_QQ(val_swap_mask_pred, swap_mask_rev_pred), 0x01010101);
  val_swap_mask = Q6_V_vand_VV(val_swap_mask, vals_are_different);
  const HVX_Vector swap_because_of_idx =
      Q6_V_vand_VV(Q6_V_vnot_V(vals_are_different), idx_cmp_result);
  const HVX_Vector swap_mask = Q6_V_vor_VV(val_swap_mask, swap_because_of_idx);

  const HVX_VectorPred swap_mask_pred = Q6_Q_vand_VR(swap_mask, 0x01010101);
  vals = Q6_V_vmux_QVV(swap_mask_pred, even_dup, odd_dup);

  const HVX_VectorPair expanded_pred =
      Q6_W_vshuff_VVR(swap_mask, swap_mask, -2);
  const HVX_VectorPred left_idx_mask =
      Q6_Q_vand_VR(Q6_V_lo_W(expanded_pred), 0x01010101);
  idx[0] =
      Q6_V_vmux_QVV(left_idx_mask, idx_shuffle_even[0], idx_shuffle_odd[0]);
  const HVX_VectorPred right_idx_mask =
      Q6_Q_vand_VR(Q6_V_hi_W(expanded_pred), 0x01010101);
  idx[1] =
      Q6_V_vmux_QVV(right_idx_mask, idx_shuffle_even[1], idx_shuffle_odd[1]);
}

template <bool DataAscending, bool IdxAscending>
// Sort one HVX candidate vector by fp16 value and int32 id.
inline void bitonic_sort_hf_i32(const HVX_Vector& step_masks_hf,
                                const HVX_Vector step_masks_i32[2],
                                HVX_Vector& vals, HVX_Vector idx[2]) {
  HVX_Vector empty = Q6_V_vzero();
  HVX_Vector full = Q6_V_vsplat_R(0xffffffff);
  HVX_VectorPair pair;
  if (DataAscending) {
    pair = Q6_W_vshuff_VVR(empty, full, 4);
  } else {
    pair = Q6_W_vshuff_VVR(full, empty, 4);
  }
  HVX_Vector sort_order_mask_hf = Q6_V_lo_W(pair);

  if (IdxAscending) {
    pair = Q6_W_vshuff_VVR(empty, full, 8);
  } else {
    pair = Q6_W_vshuff_VVR(full, empty, 8);
  }
  HVX_Vector sort_order_mask_i32[2];
  sort_order_mask_i32[0] = Q6_V_lo_W(pair);
  sort_order_mask_i32[1] = Q6_V_hi_W(pair);

  for (unsigned phase = 0; phase < 6; ++phase) {
    const HVX_Vector vals_step_mask =
        Q6_V_vxor_VV(sort_order_mask_hf, step_masks_hf);
    HVX_Vector idx_step_mask[2];
    idx_step_mask[0] = Q6_V_vxor_VV(sort_order_mask_i32[0], step_masks_i32[0]);
    idx_step_mask[1] = Q6_V_vxor_VV(sort_order_mask_i32[1], step_masks_i32[1]);

    for (unsigned step = phase; step < phase + 1; --step) {
      bitonic_sort_step_hf_i32(step, vals_step_mask, idx_step_mask, vals, idx);
    }

    pair = Q6_W_vshuff_VVR(sort_order_mask_hf, sort_order_mask_hf, -4);
    sort_order_mask_hf = Q6_V_lo_W(pair);
    pair = Q6_W_vshuff_VVR(sort_order_mask_i32[0], sort_order_mask_i32[0], -8);
    sort_order_mask_i32[0] = Q6_V_lo_W(pair);
    sort_order_mask_i32[1] = Q6_V_hi_W(pair);
  }
}

template <bool DataAscending, bool IdxAscending>
// Merge two sorted HVX vectors and keep the first vector's worth of candidates.
inline void bitonic_sort_merge_keep_first_hf_i32(
    const HVX_Vector& step_masks_hf, const HVX_Vector step_masks_i32[2],
    HVX_Vector& v1, HVX_Vector v1_idx[2], const HVX_Vector& v2,
    const HVX_Vector v2_idx[2]) {
  HVX_VectorPred swap_mask_pred;
  if (DataAscending) {
    if (IdxAscending) {
      swap_mask_pred = Q6_Q_not_Q(Q6_Q_vcmp_gt_VhfVhf(v1, v2));
    } else {
      swap_mask_pred = Q6_Q_vcmp_gt_VhfVhf(v2, v1);
    }
  } else {
    if (IdxAscending) {
      swap_mask_pred = Q6_Q_not_Q(Q6_Q_vcmp_gt_VhfVhf(v2, v1));
    } else {
      swap_mask_pred = Q6_Q_vcmp_gt_VhfVhf(v1, v2);
    }
  }

  v1 = Q6_V_vmux_QVV(swap_mask_pred, v1, v2);
  const HVX_Vector swap_mask = Q6_V_vand_QR(swap_mask_pred, 0x01010101);
  const HVX_VectorPair expanded_pred =
      Q6_W_vshuff_VVR(swap_mask, swap_mask, -2);
  const HVX_VectorPred left_idx_mask =
      Q6_Q_vand_VR(Q6_V_lo_W(expanded_pred), 0x01010101);
  v1_idx[0] = Q6_V_vmux_QVV(left_idx_mask, v1_idx[0], v2_idx[0]);
  const HVX_VectorPred right_idx_mask =
      Q6_Q_vand_VR(Q6_V_hi_W(expanded_pred), 0x01010101);
  v1_idx[1] = Q6_V_vmux_QVV(right_idx_mask, v1_idx[1], v2_idx[1]);

  HVX_Vector vals_step_mask;
  if (DataAscending) {
    vals_step_mask = Q6_V_vnot_V(step_masks_hf);
  } else {
    vals_step_mask = Q6_V_equals_V(step_masks_hf);
  }

  HVX_Vector idx_step_mask[2];
  if (IdxAscending) {
    idx_step_mask[0] = Q6_V_vnot_V(step_masks_i32[0]);
    idx_step_mask[1] = Q6_V_vnot_V(step_masks_i32[1]);
  } else {
    idx_step_mask[0] = Q6_V_equals_V(step_masks_i32[0]);
    idx_step_mask[1] = Q6_V_equals_V(step_masks_i32[1]);
  }

  for (unsigned step_idx = 5; step_idx < 6; --step_idx) {
    bitonic_sort_step_hf_i32(step_idx, vals_step_mask, idx_step_mask, v1,
                             v1_idx);
  }
}

// Initialize reusable masks for fp16/int32 bitonic sorting.
inline void bitonic_step_masks_hf_i32(HVX_Vector& step_masks_hf,
                                      HVX_Vector step_masks_i32[2]) {
  alignas(HVX_VectorSize) static constexpr uint8_t bitonic_step_mask[128] = {
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,
      0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
      0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x21, 0x22, 0x23,
      0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
      0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b,
      0x3c, 0x3d, 0x3e, 0x3f, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
      0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f, 0x50, 0x51, 0x52, 0x53,
      0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f,
      0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x6b,
      0x6c, 0x6d, 0x6e, 0x6f, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77,
      0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f};
  const HVX_Vector step_masks_byte = LoadHVX(bitonic_step_mask);
  HVX_VectorPair pair = Q6_W_vshuff_VVR(step_masks_byte, step_masks_byte, -1);
  step_masks_hf = Q6_V_lo_W(pair);
  pair = Q6_W_vshuff_VVR(step_masks_hf, step_masks_hf, -2);
  step_masks_i32[0] = Q6_V_lo_W(pair);
  step_masks_i32[1] = Q6_V_hi_W(pair);
}

// Load one candidate chunk into HVX vectors, padding missing lanes.
inline void load_bitonic_candidate_chunk_hf_i32(const float16* candidate_scores,
                                                const int32_t* candidate_ids,
                                                int32_t offset, int32_t count,
                                                HVX_Vector& vals,
                                                HVX_Vector idx[2]) {
  constexpr int32_t kValsInVec = HVX_VectorSize / sizeof(float16);
  constexpr int32_t kIdxInVec = HVX_VectorSize / sizeof(int32_t);
  const float16 neg_max_h = (float16)-65504.0F;
  const HVX_Vector neg_max_vec = Q6_Vh_vsplat_R(*(const uint16_t*)&neg_max_h);
  const HVX_Vector int_max_vec = Q6_V_vsplat_R(INT32_MAX);

  const int32_t vals_to_load = count < kValsInVec ? count : kValsInVec;
  vals = LoadUnaligned<HVX_Vector>((const int8_t*)&candidate_scores[offset],
                                   vals_to_load * sizeof(float16));
  if (vals_to_load < kValsInVec) {
    const HVX_VectorPred pred = Q6_Q_vsetq2_R(vals_to_load * sizeof(float16));
    vals = Q6_V_vmux_QVV(pred, vals, neg_max_vec);
  }

  const int32_t lo_to_load =
      vals_to_load < kIdxInVec ? vals_to_load : kIdxInVec;
  idx[0] = LoadUnaligned<HVX_Vector>((const int8_t*)&candidate_ids[offset],
                                     lo_to_load * sizeof(int32_t));
  if (lo_to_load < kIdxInVec) {
    const HVX_VectorPred pred = Q6_Q_vsetq2_R(lo_to_load * sizeof(int32_t));
    idx[0] = Q6_V_vmux_QVV(pred, idx[0], int_max_vec);
  }

  const int32_t hi_to_load =
      vals_to_load > kIdxInVec ? vals_to_load - kIdxInVec : 0;
  if (hi_to_load > 0) {
    idx[1] = LoadUnaligned<HVX_Vector>(
        (const int8_t*)&candidate_ids[offset + kIdxInVec],
        hi_to_load * sizeof(int32_t));
    if (hi_to_load < kIdxInVec) {
      const HVX_VectorPred pred = Q6_Q_vsetq2_R(hi_to_load * sizeof(int32_t));
      idx[1] = Q6_V_vmux_QVV(pred, idx[1], int_max_vec);
    }
  } else {
    idx[1] = int_max_vec;
  }
}

// Select top-k from a candidate list using HVX bitonic merge steps.
inline void select_topk_candidates_bitonic_hf(
    const float16* token_scores, const float16* candidate_scores,
    const int32_t* candidate_ids, int32_t candidate_count,
    int32_t selected_count, float* selected_weights, int32_t* selected_ids) {
  constexpr int32_t kValsInVec = HVX_VectorSize / sizeof(float16);
  const float16 neg_max_h = (float16)-65504.0F;
  HVX_Vector best_vals = Q6_Vh_vsplat_R(*(const uint16_t*)&neg_max_h);
  HVX_Vector best_idx[2] = {Q6_V_vsplat_R(INT32_MAX), Q6_V_vsplat_R(INT32_MAX)};
  HVX_Vector step_masks_hf;
  HVX_Vector step_masks_i32[2];
  bitonic_step_masks_hf_i32(step_masks_hf, step_masks_i32);

  for (int32_t offset = 0; offset < candidate_count; offset += kValsInVec) {
    HVX_Vector candidate_vals;
    HVX_Vector candidate_idx[2];
    load_bitonic_candidate_chunk_hf_i32(candidate_scores, candidate_ids, offset,
                                        candidate_count - offset,
                                        candidate_vals, candidate_idx);
    bitonic_sort_hf_i32<true, false>(step_masks_hf, step_masks_i32,
                                     candidate_vals, candidate_idx);
    bitonic_sort_merge_keep_first_hf_i32<false, true>(
        step_masks_hf, step_masks_i32, best_vals, best_idx, candidate_vals,
        candidate_idx);
  }

  int32_t sorted_ids[kValsInVec] __attribute__((aligned(HVX_VectorSize)));
  StoreHVX(&sorted_ids[0], best_idx[0]);
  StoreHVX(&sorted_ids[HVX_VectorSize / sizeof(int32_t)], best_idx[1]);
  for (int32_t rank = 0; rank < selected_count; ++rank) {
    const int32_t expert = sorted_ids[rank];
    selected_ids[rank] = expert;
    selected_weights[rank] = (float)token_scores[expert];
  }
}

// Select top-k experts from selected groups using compacted bitonic candidates.
inline void select_topk_from_groups_bitonic_hf(
    const float16* token_scores, const float16* selection_scores,
    const int32_t* selected_groups, int32_t topk_group,
    int32_t experts_per_group, int32_t selected_count, float* selected_weights,
    int32_t* selected_ids) {
  float16 candidate_scores[kMaxExperts]
      __attribute__((aligned(HVX_VectorSize)));
  int32_t candidate_ids[kMaxExperts] __attribute__((aligned(HVX_VectorSize)));
  int32_t sorted_groups[kMaxGroups] __attribute__((aligned(HVX_VectorSize)));

  for (int32_t i = 0; i < topk_group; ++i) {
    sorted_groups[i] = selected_groups[i];
  }
  for (int32_t i = 1; i < topk_group; ++i) {
    const int32_t group = sorted_groups[i];
    int32_t j = i - 1;
    while (j >= 0 && sorted_groups[j] > group) {
      sorted_groups[j + 1] = sorted_groups[j];
      --j;
    }
    sorted_groups[j + 1] = group;
  }

  int32_t candidate_count = 0;

  for (int32_t group_rank = 0; group_rank < topk_group; ++group_rank) {
    const int32_t group = sorted_groups[group_rank];
    const int32_t begin = group * experts_per_group;
    const int32_t end = begin + experts_per_group;
    for (int32_t expert = begin; expert < end; ++expert) {
      candidate_scores[candidate_count] = selection_scores[expert];
      candidate_ids[candidate_count] = expert;
      ++candidate_count;
    }
  }

  select_topk_candidates_bitonic_hf(
      token_scores, candidate_scores, candidate_ids, candidate_count,
      selected_count, selected_weights, selected_ids);
}

// Select regular top-k experts with bitonic candidates over all experts.
inline void select_topk_regular_bitonic_hf(const float16* token_scores,
                                           const float16* selection_scores,
                                           int32_t num_experts,
                                           int32_t selected_count,
                                           float* selected_weights,
                                           int32_t* selected_ids) {
  int32_t candidate_ids[kMaxExperts] __attribute__((aligned(HVX_VectorSize)));
  for (int32_t expert = 0; expert < num_experts; ++expert) {
    candidate_ids[expert] = expert;
  }

  select_topk_candidates_bitonic_hf(token_scores, selection_scores,
                                    candidate_ids, num_experts, selected_count,
                                    selected_weights, selected_ids);
}

// Scalar top-k selection from selected groups.
inline void select_topk_from_groups_scalar_hf(
    const float16* token_scores, float16* selection_scores,
    const int32_t* selected_groups, int32_t topk_group,
    int32_t experts_per_group, int32_t selected_count, float* selected_weights,
    int32_t* selected_ids) {
  for (int32_t rank = 0; rank < selected_count; ++rank) {
    float best_value = kNegInf;
    int32_t best_expert = 0;

    for (int32_t group_rank = 0; group_rank < topk_group; ++group_rank) {
      const int32_t group = selected_groups[group_rank];
      const int32_t begin = group * experts_per_group;
      const int32_t end = begin + experts_per_group;
      for (int32_t expert = begin; expert < end; ++expert) {
        const float value = selection_score_at(selection_scores, expert);
        if (better_pair(value, expert, best_value, best_expert)) {
          best_value = value;
          best_expert = expert;
        }
      }
    }

    selected_ids[rank] = best_expert;
    selected_weights[rank] = (float)token_scores[best_expert];
    selection_scores[best_expert] = (float16)-65504.0F;
  }
}

// HVX-assisted repeated top-k selection from selected groups.
inline void select_topk_from_groups_hvx_hf(
    const float16* token_scores, float16* selection_scores,
    const int32_t* selected_groups, int32_t topk_group,
    int32_t experts_per_group, int32_t selected_count, float* selected_weights,
    int32_t* selected_ids) {
  for (int32_t rank = 0; rank < selected_count; ++rank) {
    float best_value = kNegInf;
    int32_t best_expert = 0;

    for (int32_t group_rank = 0; group_rank < topk_group; ++group_rank) {
      const int32_t group = selected_groups[group_rank];
      const int32_t begin = group * experts_per_group;
      const float group_best_value =
          max_score_hvx_hf(selection_scores, begin, experts_per_group);
      if (group_best_value < best_value) {
        continue;
      }

      const int32_t end = begin + experts_per_group;
      const int32_t group_best_expert = first_expert_with_score_hf(
          selection_scores, begin, end, group_best_value);
      if (better_pair(group_best_value, group_best_expert, best_value,
                      best_expert)) {
        best_value = group_best_value;
        best_expert = group_best_expert;
      }
    }

    selected_ids[rank] = best_expert;
    selected_weights[rank] = (float)token_scores[best_expert];
    selection_scores[best_expert] = (float16)-65504.0F;
  }
}

// Normalize selected weights, apply routing scale, and store ids/weights.
inline void normalize_and_store(const float* selected_weights,
                                const int32_t* selected_ids,
                                int32_t selected_count, float* out_weights,
                                int32_t* out_ids, int32_t topk,
                                bool renormalize, float routed_scaling_factor) {
  float denom = 0.0F;
  if (renormalize) {
    for (int32_t i = 0; i < selected_count; ++i) {
      denom += selected_weights[i];
    }
    if (denom == 0.0F) {
      denom = 1.0F;
    }
  }

  float scale = routed_scaling_factor;
  if (renormalize) {
    scale /= denom;
  }

  for (int32_t i = 0; i < topk; ++i) {
    if (i < selected_count) {
      out_ids[i] = selected_ids[i];
      out_weights[i] = selected_weights[i] * scale;
    } else {
      out_ids[i] = 0;
      out_weights[i] = 0.0F;
    }
  }
}

// Zero-fill output slots when a token cannot be routed.
inline void zero_topk(float* out_weights, int32_t* out_ids, int32_t topk) {
  for (int32_t i = 0; i < topk; ++i) {
    out_ids[i] = 0;
    out_weights[i] = 0.0F;
  }
}

// Route one token through grouped top-k selection.
inline void route_grouped_token_hf(
    const float16* token_input, const float16* bias, float16* token_scores,
    float16* selection_scores, float* out_weights, int32_t* out_ids,
    int32_t num_experts, int32_t num_groups, int32_t topk_group, int32_t topk,
    bool renormalize, float routed_scaling_factor, bool use_bias,
    ScoreMode score_mode) {
  if (num_groups <= 0 || topk_group <= 0 || topk <= 0 ||
      num_experts <= num_groups || num_experts % num_groups != 0 ||
      num_experts > kMaxExperts || num_groups > kMaxGroups ||
      topk_group > kMaxGroups || topk > kMaxTopK) {
    zero_topk(out_weights, out_ids, topk);
    return;
  }

  materialize_scores_hf(token_input, token_scores, num_experts, score_mode);
  materialize_selection_scores_hvx_hf(token_scores, bias, selection_scores,
                                      num_experts, use_bias);

  const int32_t experts_per_group = num_experts / num_groups;
  float group_scores[kMaxGroups];
  int32_t selected_groups[kMaxGroups];

  for (int32_t group = 0; group < num_groups; ++group) {
    const int32_t begin = group * experts_per_group;
    const int32_t end = begin + experts_per_group;
    if (use_bias) {
      group_scores[group] =
          experts_per_group >= 16
              ? top2_sum_score_hvx_hf(selection_scores, begin, end)
              : top2_sum_score_hf(selection_scores, begin, end);
    } else {
      group_scores[group] =
          max_score_hvx_hf(selection_scores, begin, experts_per_group);
    }
  }

  if (topk_group > num_groups) {
    topk_group = num_groups;
  }
  if (num_groups >= 16) {
    select_top_groups_hvx(group_scores, num_groups, topk_group,
                          selected_groups);
  } else {
    select_top_groups(group_scores, num_groups, topk_group, selected_groups);
  }

  int32_t selected_ids[kMaxTopK];
  float selected_weights[kMaxTopK];
  int32_t selected_count = topk;
  const int32_t max_candidates = topk_group * experts_per_group;
  if (selected_count > max_candidates) {
    selected_count = max_candidates;
  }

  if (topk_group * experts_per_group >= 64 && selected_count >= 4) {
    select_topk_from_groups_bitonic_hf(
        token_scores, selection_scores, selected_groups, topk_group,
        experts_per_group, selected_count, selected_weights, selected_ids);
  } else if (experts_per_group >= 16) {
    select_topk_from_groups_hvx_hf(
        token_scores, selection_scores, selected_groups, topk_group,
        experts_per_group, selected_count, selected_weights, selected_ids);
  } else {
    select_topk_from_groups_scalar_hf(
        token_scores, selection_scores, selected_groups, topk_group,
        experts_per_group, selected_count, selected_weights, selected_ids);
  }

  normalize_and_store(selected_weights, selected_ids, selected_count,
                      out_weights, out_ids, topk, renormalize,
                      routed_scaling_factor);
}

// Route one token through regular top-k selection.
inline void route_regular_token_hf(
    const float16* token_input, const float16* bias, float16* token_scores,
    float16* selection_scores, float* out_weights, int32_t* out_ids,
    int32_t num_experts, int32_t topk, bool renormalize,
    float routed_scaling_factor, bool use_bias, ScoreMode score_mode) {
  if (topk <= 0 || topk > kMaxTopK || num_experts <= 0 ||
      num_experts > kMaxExperts) {
    zero_topk(out_weights, out_ids, topk);
    return;
  }

  materialize_scores_hf(token_input, token_scores, num_experts, score_mode);
  materialize_selection_scores_hvx_hf(token_scores, bias, selection_scores,
                                      num_experts, use_bias);

  int32_t selected_count = topk;
  if (selected_count > num_experts) {
    selected_count = num_experts;
  }

  int32_t selected_ids[kMaxTopK];
  float selected_weights[kMaxTopK];

  if (num_experts >= 64 && selected_count >= 4) {
    select_topk_regular_bitonic_hf(token_scores, selection_scores, num_experts,
                                   selected_count, selected_weights,
                                   selected_ids);
  } else {
    for (int32_t rank = 0; rank < selected_count; ++rank) {
      float best_value = kNegInf;
      int32_t best_expert = 0;

      for (int32_t expert = 0; expert < num_experts; ++expert) {
        const float value = selection_score_at(selection_scores, expert);
        if (better_pair(value, expert, best_value, best_expert)) {
          best_value = value;
          best_expert = expert;
        }
      }

      selected_ids[rank] = best_expert;
      selected_weights[rank] = (float)token_scores[best_expert];
      selection_scores[best_expert] = (float16)-65504.0F;
    }
  }

  normalize_and_store(selected_weights, selected_ids, selected_count,
                      out_weights, out_ids, topk, renormalize,
                      routed_scaling_factor);
}

// Assign per-thread VTCM score scratch slices.
inline void init_vtcm_scratch_buffers(RouterTileParams* params,
                                      int32_t thread_id, int32_t num_threads) {
  int8_t* vtcm_base = (int8_t*)qshimGetBaseVtcmAddr();
  const uint32_t vtcm_bytes = query_vtcm_size_by_two();
  const uint32_t thread_vtcm_bytes = vtcm_bytes / num_threads;
  uint64_t ptr = (uint64_t)(vtcm_base + thread_id * thread_vtcm_bytes);
  ptr = align_up_u64(ptr, HVX_VectorSize);

  params->vtcm_scores = (float16*)ptr;
  ptr += params->num_experts * sizeof(float16);
  ptr = align_up_u64(ptr, HVX_VectorSize);
  params->vtcm_selection_scores = (float16*)ptr;
}

// Parse kernel arguments and assign each compute worker its token range.
inline uint32_t init_common_params(const AicJitEntryPointConfig* entryConfig,
                                   const AicJitPointerArray* pointerArray,
                                   RouterTileParams* params, bool grouped) {
  params->input = (const float16*)pointerArray->pointers[0];
  params->bias = (const float16*)pointerArray->pointers[1];
  params->topk_weights = (float*)pointerArray->pointers[2];
  params->topk_ids = (int32_t*)pointerArray->pointers[3];

  params->num_tokens = *(const int32_t*)pointerArray->pointers[4];
  params->num_experts = *(const int32_t*)pointerArray->pointers[5];
  if (grouped) {
    params->num_groups = *(const int32_t*)pointerArray->pointers[6];
    params->topk_group = *(const int32_t*)pointerArray->pointers[7];
    params->topk = *(const int32_t*)pointerArray->pointers[8];
    params->renormalize = *(const int32_t*)pointerArray->pointers[9] != 0;
    params->routed_scaling_factor = *(const float*)pointerArray->pointers[10];
    params->use_bias = *(const int32_t*)pointerArray->pointers[11] != 0;
    if (!score_mode_from_id(*(const int32_t*)pointerArray->pointers[12],
                            &params->score_mode)) {
      return JIT_DEV_ERROR_INVALID_PARAMETER;
    }
  } else {
    params->num_groups = 0;
    params->topk_group = 0;
    params->topk = *(const int32_t*)pointerArray->pointers[6];
    params->renormalize = *(const int32_t*)pointerArray->pointers[7] != 0;
    params->routed_scaling_factor = *(const float*)pointerArray->pointers[8];
    params->use_bias = *(const int32_t*)pointerArray->pointers[9] != 0;
    if (!score_mode_from_id(*(const int32_t*)pointerArray->pointers[10],
                            &params->score_mode)) {
      return JIT_DEV_ERROR_INVALID_PARAMETER;
    }
  }

  const int32_t local_thread_id =
      entryConfig->threadID % entryConfig->numThreads;
  const int32_t compute_threads_per_core = entryConfig->numThreads;
  const int32_t compute_thread_id = local_thread_id;

  const int32_t workers = entryConfig->numCores * compute_threads_per_core;
  const int32_t worker_id =
      entryConfig->coreID * compute_threads_per_core + compute_thread_id;
  const int32_t tokens_per_worker = ceil_div_i32(params->num_tokens, workers);
  params->token_begin = worker_id * tokens_per_worker;
  params->token_end = params->token_begin + tokens_per_worker;
  if (params->token_end > params->num_tokens) {
    params->token_end = params->num_tokens;
  }

  init_vtcm_scratch_buffers(params, compute_thread_id,
                            compute_threads_per_core);

  return JIT_DEV_STATUS_SUCCESS;
}

// Process assigned tokens through grouped routing.
inline void process_grouped_direct(const RouterTileParams& params,
                                   ScoreMode score_mode) {
  const float16* bias = params.use_bias ? params.bias : nullptr;
  for (int32_t token = params.token_begin; token < params.token_end; ++token) {
    route_grouped_token_hf(
        params.input + token * params.num_experts, bias, params.vtcm_scores,
        params.vtcm_selection_scores, params.topk_weights + token * params.topk,
        params.topk_ids + token * params.topk, params.num_experts,
        params.num_groups, params.topk_group, params.topk, params.renormalize,
        params.routed_scaling_factor, params.use_bias, score_mode);
  }
}

// Process assigned tokens through regular routing.
inline void process_regular_direct(const RouterTileParams& params,
                                   ScoreMode score_mode) {
  const float16* bias = params.use_bias ? params.bias : nullptr;
  for (int32_t token = params.token_begin; token < params.token_end; ++token) {
    route_regular_token_hf(
        params.input + token * params.num_experts, bias, params.vtcm_scores,
        params.vtcm_selection_scores, params.topk_weights + token * params.topk,
        params.topk_ids + token * params.topk, params.num_experts, params.topk,
        params.renormalize, params.routed_scaling_factor, params.use_bias,
        score_mode);
  }
}

// Entry point for the grouped top-k router kernel.
inline uint32_t grouped_kernel_main(const AicJitEntryPointConfig* entryConfig,
                                    const AicJitPointerArray* pointerArray) {
  RouterTileParams params;
  uint32_t status =
      init_common_params(entryConfig, pointerArray, &params, true);
  if (status != JIT_DEV_STATUS_SUCCESS) {
    return status;
  }
  if (params.token_begin >= params.token_end) {
    return JIT_DEV_STATUS_SUCCESS;
  }

  process_grouped_direct(params, params.score_mode);
  return JIT_DEV_STATUS_SUCCESS;
}

// Entry point for the regular top-k router kernel.
inline uint32_t regular_kernel_main(const AicJitEntryPointConfig* entryConfig,
                                    const AicJitPointerArray* pointerArray) {
  RouterTileParams params;
  uint32_t status =
      init_common_params(entryConfig, pointerArray, &params, false);
  if (status != JIT_DEV_STATUS_SUCCESS) {
    return status;
  }
  if (params.token_begin >= params.token_end) {
    return JIT_DEV_STATUS_SUCCESS;
  }

  process_regular_direct(params, params.score_mode);
  return JIT_DEV_STATUS_SUCCESS;
}

}  // namespace grouped_topk_router
