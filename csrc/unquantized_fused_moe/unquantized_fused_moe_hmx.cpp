// ---------------------------------------------------------------------------------------
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
// ---------------------------------------------------------------------------------------

#include <stdint.h>

#include "QAicHexagonHMX.h"
#include "QAicHexagonHMXDefs.h"
#include "QAicHexagonHVX.h"
#include "QAicHexagonMath.h"
#include "QAicHexagonPlatformIntf.h"
#include "QAicHexagonTypes.h"
#include "QAicHexagonUtils.h"
#include "jit_dev_exe_function.h"
#include "jit_dev_status_codes.h"
#include "jit_qshim_api.h"

namespace unquantized_fused_moe_hmx {

enum ActivationId : int32_t {
  kSilu = 0,
  kGelu = 1,
  kGeluTanh = 2,
  kSwigluOAI = 3,
  kSwigluStep = 4,
  kSiluNoMul = 5,
  kGeluNoMul = 6,
  kGeluTanhNoMul = 7,
  kRelu2NoMul = 8,
};

enum PostprocessOrder : int32_t {
  kScaleThenBias = 0,
  kBiasThenScale = 1,
};

#define QAIC_MOE_SHUFFLE_EVEN_HF                                               \
  0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37,    \
      40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60, 61, 64, 65, 68, 69, 72, 73,  \
      76, 77, 80, 81, 84, 85, 88, 89, 92, 93, 96, 97, 100, 101, 104, 105, 108, \
      109, 112, 113, 116, 117, 120, 121, 124, 125, 128, 129, 132, 133, 136,    \
      137, 140, 141, 144, 145, 148, 149, 152, 153, 156, 157, 160, 161, 164,    \
      165, 168, 169, 172, 173, 176, 177, 180, 181, 184, 185, 188, 189, 192,    \
      193, 196, 197, 200, 201, 204, 205, 208, 209, 212, 213, 216, 217, 220,    \
      221, 224, 225, 228, 229, 232, 233, 236, 237, 240, 241, 244, 245, 248,    \
      249, 252, 253

#define QAIC_MOE_SHUFFLE_ODD_HF                                                \
  2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31, 34, 35, 38, 39,  \
      42, 43, 46, 47, 50, 51, 54, 55, 58, 59, 62, 63, 66, 67, 70, 71, 74, 75,  \
      78, 79, 82, 83, 86, 87, 90, 91, 94, 95, 98, 99, 102, 103, 106, 107, 110, \
      111, 114, 115, 118, 119, 122, 123, 126, 127, 130, 131, 134, 135, 138,    \
      139, 142, 143, 146, 147, 150, 151, 154, 155, 158, 159, 162, 163, 166,    \
      167, 170, 171, 174, 175, 178, 179, 182, 183, 186, 187, 190, 191, 194,    \
      195, 198, 199, 202, 203, 206, 207, 210, 211, 214, 215, 218, 219, 222,    \
      223, 226, 227, 230, 231, 234, 235, 238, 239, 242, 243, 246, 247, 250,    \
      251, 254, 255

struct HmxScratch {
  uint32_t* status_words;
  float16* lhs_rm;
  float16* rhs_cm;
  float16* lhs_crouton;
  float16* rhs_crouton;
  float16* out_crouton;
  float16* out_rm;
  float16* gate_up_batch;
  float16* hidden_batch;
  int32_t route_tile_rows;
  int32_t column_tile_cols;
};

static constexpr uint32_t kMaxKernelThreads = 8;

inline uint64_t align_up_u64(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1ULL) & ~(alignment - 1ULL);
}

inline uint8_t* align_up_ptr(uint8_t* ptr, uintptr_t alignment) {
  const uintptr_t value = (uintptr_t)ptr;
  return (uint8_t*)((value + alignment - 1U) & ~(alignment - 1U));
}

inline bool activation_is_no_mul(int32_t activation_id) {
  return activation_id == kSiluNoMul || activation_id == kGeluNoMul ||
         activation_id == kGeluTanhNoMul || activation_id == kRelu2NoMul;
}

inline bool valid_activation(int32_t activation_id) {
  return activation_id >= kSilu && activation_id <= kRelu2NoMul;
}

inline bool valid_jit_args(const AicJitEntryPointConfig* cfg,
                           const AicJitPointerArray* ptrs,
                           uint32_t min_pointers) {
  return cfg != nullptr && ptrs != nullptr &&
         ptrs->numPointers >= min_pointers && cfg->numCores > 0 &&
         cfg->numThreads > 0;
}

template <typename T>
inline T pointer_arg(const AicJitPointerArray* ptrs, uint32_t index) {
  return reinterpret_cast<T>(ptrs->pointers[index]);
}

inline int32_t param_i32(const float* params, uint32_t index) {
  return (int32_t)params[index];
}

inline bool param_bool(const float* params, uint32_t index) {
  return param_i32(params, index) != 0;
}

inline int32_t map_global_to_local_expert(int32_t global_expert,
                                          const float* expert_map,
                                          int32_t local_num_experts,
                                          int32_t global_num_experts,
                                          bool has_expert_map) {
  if (has_expert_map) {
    if (global_expert < 0 || global_expert >= global_num_experts) {
      return -1;
    }
    const int32_t local_expert = (int32_t)expert_map[global_expert];
    return (local_expert >= 0 && local_expert < local_num_experts)
               ? local_expert
               : -1;
  }
  return (global_expert >= 0 && global_expert < local_num_experts)
             ? global_expert
             : -1;
}

inline HVX_Vector splat_hf(float value) {
  float16 value_hf = (float16)value;
  return Q6_Vh_vsplat_R(*(uint16_t*)&value_hf);
}

inline HVX_Vector splat_sf(float value) {
  return Q6_V_vsplat_R(*(uint32_t*)&value);
}

inline HVX_Vector load_partial_hf_zero(const float16* ptr, int32_t elements) {
  const int32_t bytes = elements * (int32_t)sizeof(float16);
  const HVX_Vector value =
      LoadUnaligned<HVX_Vector>((const int8_t*)ptr, (uint32_t)bytes);
  const HVX_VectorPred valid = Q6_Q_vsetq2_R(bytes);
  return Q6_V_vmux_QVV(valid, value, Q6_V_vzero());
}

inline HVX_Vector silu_vec_hf(HVX_Vector value_vhf) {
  return Q6_Vhf_vmpy_VhfVhf(value_vhf, qaic_sigmoid_hf(value_vhf));
}

inline HVX_Vector gelu_exact_vec_hf(HVX_Vector value_vhf) {
  static constexpr float kInvSqrt2 = 0.7071067811865475F;

  HVX_VectorPair value_pair = Q6_Wsf_vcvt_Vhf(value_vhf);
  HVX_Vector value_lo = Q6_V_lo_W(value_pair);
  HVX_Vector value_hi = Q6_V_hi_W(value_pair);
  HVX_Vector inv_sqrt2 = splat_sf(kInvSqrt2);

  HVX_Vector erf_lo = qaic_erf_sf(Q6_Vsf_vmpy_VsfVsf(value_lo, inv_sqrt2));
  HVX_Vector erf_hi = qaic_erf_sf(Q6_Vsf_vmpy_VsfVsf(value_hi, inv_sqrt2));
  HVX_Vector erf_vhf = Q6_Vhf_vcvt_VsfVsf(erf_lo, erf_hi);
  HVX_Vector one_vhf = splat_hf(1.0F);
  HVX_Vector half_vhf = splat_hf(0.5F);
  HVX_Vector scaled = Q6_Vhf_vmpy_VhfVhf(half_vhf, value_vhf);
  return Q6_Vhf_vmpy_VhfVhf(Q6_Vhf_vadd_VhfVhf(erf_vhf, one_vhf), scaled);
}

inline HVX_Vector gelu_tanh_approx_vec_hf(HVX_Vector value_vhf) {
  static constexpr float kSqrt2OverPi = 0.7978845608028654F;
  static constexpr float kGeluCoeff = 0.044715F;

  HVX_VectorPair value_pair = Q6_Wsf_vcvt_Vhf(value_vhf);
  HVX_Vector value_lo = Q6_V_lo_W(value_pair);
  HVX_Vector value_hi = Q6_V_hi_W(value_pair);
  HVX_Vector coeff = splat_sf(kGeluCoeff);
  HVX_Vector scale = splat_sf(kSqrt2OverPi);

  HVX_Vector arg_lo = Q6_Vsf_vmpy_VsfVsf(value_lo, value_lo);
  arg_lo = Q6_Vsf_vmpy_VsfVsf(arg_lo, value_lo);
  arg_lo = Q6_Vsf_vmpy_VsfVsf(arg_lo, coeff);
  arg_lo = Q6_Vsf_vadd_VsfVsf(arg_lo, value_lo);
  arg_lo = Q6_Vsf_vmpy_VsfVsf(arg_lo, scale);

  HVX_Vector arg_hi = Q6_Vsf_vmpy_VsfVsf(value_hi, value_hi);
  arg_hi = Q6_Vsf_vmpy_VsfVsf(arg_hi, value_hi);
  arg_hi = Q6_Vsf_vmpy_VsfVsf(arg_hi, coeff);
  arg_hi = Q6_Vsf_vadd_VsfVsf(arg_hi, value_hi);
  arg_hi = Q6_Vsf_vmpy_VsfVsf(arg_hi, scale);

  HVX_Vector tanh_vhf = qaic_tanh_hf(Q6_Vhf_vcvt_VsfVsf(arg_lo, arg_hi));
  HVX_Vector one_vhf = splat_hf(1.0F);
  HVX_Vector half_vhf = splat_hf(0.5F);
  HVX_Vector scaled = Q6_Vhf_vmpy_VhfVhf(half_vhf, value_vhf);
  return Q6_Vhf_vmpy_VhfVhf(Q6_Vhf_vadd_VhfVhf(tanh_vhf, one_vhf), scaled);
}

inline HVX_Vector relu2_vec_hf(HVX_Vector value_vhf) {
  HVX_Vector zero = Q6_V_vzero();
  HVX_VectorPred positive = Q6_Q_vcmp_gt_VhfVhf(value_vhf, zero);
  HVX_Vector relu = Q6_V_vmux_QVV(positive, value_vhf, zero);
  return Q6_Vhf_vmpy_VhfVhf(relu, relu);
}

inline HVX_Vector clamp_max_vec_hf(HVX_Vector value_vhf, float max_value) {
  HVX_Vector max_vhf = splat_hf(max_value);
  HVX_VectorPred too_high = Q6_Q_vcmp_gt_VhfVhf(value_vhf, max_vhf);
  return Q6_V_vmux_QVV(too_high, max_vhf, value_vhf);
}

inline HVX_Vector clamp_vec_hf(HVX_Vector value_vhf, float min_value,
                               float max_value) {
  HVX_Vector min_vhf = splat_hf(min_value);
  HVX_Vector max_vhf = splat_hf(max_value);
  HVX_VectorPred too_low = Q6_Q_vcmp_gt_VhfVhf(min_vhf, value_vhf);
  HVX_VectorPred too_high = Q6_Q_vcmp_gt_VhfVhf(value_vhf, max_vhf);
  value_vhf = Q6_V_vmux_QVV(too_low, min_vhf, value_vhf);
  return Q6_V_vmux_QVV(too_high, max_vhf, value_vhf);
}

inline __attribute__((always_inline)) HVX_Vector apply_activation_hvx(
    HVX_Vector gate_vhf, HVX_Vector up_vhf, int32_t activation_id) {
  if (activation_id == kSiluNoMul) {
    return silu_vec_hf(gate_vhf);
  }
  if (activation_id == kGeluNoMul) {
    return gelu_exact_vec_hf(gate_vhf);
  }
  if (activation_id == kGeluTanhNoMul) {
    return gelu_tanh_approx_vec_hf(gate_vhf);
  }
  if (activation_id == kRelu2NoMul) {
    return relu2_vec_hf(gate_vhf);
  }
  if (activation_id == kSwigluOAI) {
    gate_vhf = clamp_max_vec_hf(gate_vhf, 7.0F);
    up_vhf = clamp_vec_hf(up_vhf, -7.0F, 7.0F);
    const HVX_Vector sigmoid_arg =
        Q6_Vhf_vmpy_VhfVhf(gate_vhf, splat_hf(1.702F));
    const HVX_Vector swish_gate =
        Q6_Vhf_vmpy_VhfVhf(gate_vhf, qaic_sigmoid_hf(sigmoid_arg));
    return Q6_Vhf_vmpy_VhfVhf(Q6_Vhf_vadd_VhfVhf(up_vhf, splat_hf(1.0F)),
                              swish_gate);
  }
  if (activation_id == kSwigluStep) {
    gate_vhf = clamp_max_vec_hf(silu_vec_hf(gate_vhf), 7.0F);
    up_vhf = clamp_vec_hf(up_vhf, -7.0F, 7.0F);
    return Q6_Vhf_vmpy_VhfVhf(gate_vhf, up_vhf);
  }

  const HVX_Vector activated = activation_id == kSilu ? silu_vec_hf(gate_vhf)
                               : activation_id == kGelu
                                   ? gelu_exact_vec_hf(gate_vhf)
                                   : gelu_tanh_approx_vec_hf(gate_vhf);
  return Q6_Vhf_vmpy_VhfVhf(activated, up_vhf);
}

inline void apply_activation_vec(const float16* gate_up, float16* hidden,
                                 int32_t intermediate_size,
                                 int32_t activation_id) {
  constexpr int32_t kElemsPerHalfVector = sizeof(HVX_Vector) / sizeof(float16);
  const int32_t full_vectors = intermediate_size / kElemsPerHalfVector;
  const int32_t vector_elems = full_vectors * kElemsPerHalfVector;

  for (int32_t offset = 0; offset < vector_elems;
       offset += kElemsPerHalfVector) {
    HVX_Vector gate_vhf =
        LoadUnaligned<HVX_Vector>((const int8_t*)(gate_up + offset));
    HVX_Vector out_vhf;
    if (activation_id == kSiluNoMul) {
      out_vhf = silu_vec_hf(gate_vhf);
    } else if (activation_id == kGeluNoMul) {
      out_vhf = gelu_exact_vec_hf(gate_vhf);
    } else if (activation_id == kGeluTanhNoMul) {
      out_vhf = gelu_tanh_approx_vec_hf(gate_vhf);
    } else if (activation_id == kRelu2NoMul) {
      out_vhf = relu2_vec_hf(gate_vhf);
    } else {
      HVX_Vector up_vhf = LoadUnaligned<HVX_Vector>(
          (const int8_t*)(gate_up + intermediate_size + offset));
      if (activation_id == kSwigluOAI) {
        gate_vhf = clamp_max_vec_hf(gate_vhf, 7.0F);
        up_vhf = clamp_vec_hf(up_vhf, -7.0F, 7.0F);
        HVX_Vector sigmoid_arg = Q6_Vhf_vmpy_VhfVhf(gate_vhf, splat_hf(1.702F));
        HVX_Vector swish_gate =
            Q6_Vhf_vmpy_VhfVhf(gate_vhf, qaic_sigmoid_hf(sigmoid_arg));
        out_vhf = Q6_Vhf_vmpy_VhfVhf(Q6_Vhf_vadd_VhfVhf(up_vhf, splat_hf(1.0F)),
                                     swish_gate);
      } else if (activation_id == kSwigluStep) {
        gate_vhf = clamp_max_vec_hf(silu_vec_hf(gate_vhf), 7.0F);
        up_vhf = clamp_vec_hf(up_vhf, -7.0F, 7.0F);
        out_vhf = Q6_Vhf_vmpy_VhfVhf(gate_vhf, up_vhf);
      } else {
        HVX_Vector activated;
        if (activation_id == kSilu) {
          activated = silu_vec_hf(gate_vhf);
        } else if (activation_id == kGelu) {
          activated = gelu_exact_vec_hf(gate_vhf);
        } else {
          activated = gelu_tanh_approx_vec_hf(gate_vhf);
        }
        out_vhf = Q6_Vhf_vmpy_VhfVhf(activated, up_vhf);
      }
    }
    StoreUnalignedHVX((int8_t*)(hidden + offset), out_vhf);
  }

  const int32_t remaining = intermediate_size - vector_elems;
  if (remaining > 0) {
    const HVX_Vector gate_vhf =
        load_partial_hf_zero(gate_up + vector_elems, remaining);
    const HVX_Vector up_vhf =
        activation_is_no_mul(activation_id)
            ? Q6_V_vzero()
            : load_partial_hf_zero(gate_up + intermediate_size + vector_elems,
                                   remaining);
    const HVX_Vector out_vhf =
        apply_activation_hvx(gate_vhf, up_vhf, activation_id);
    StoreUnalignedHVX((int8_t*)(hidden + vector_elems), out_vhf,
                      remaining * (int32_t)sizeof(float16));
  }
}

inline void zero_route_out(float16* route_out, int32_t hidden_size) {
  constexpr int32_t kElemsPerHalfVector = sizeof(HVX_Vector) / sizeof(float16);
  const int32_t full_vectors = hidden_size / kElemsPerHalfVector;
  const int32_t vector_elems = full_vectors * kElemsPerHalfVector;
  HVX_Vector zero = Q6_V_vzero();
  for (int32_t offset = 0; offset < vector_elems;
       offset += kElemsPerHalfVector) {
    StoreUnalignedHVX((int8_t*)(route_out + offset), zero);
  }
  const int32_t remaining = hidden_size - vector_elems;
  if (remaining > 0) {
    StoreUnalignedHVX((int8_t*)(route_out + vector_elems), zero,
                      remaining * (int32_t)sizeof(float16));
  }
}

inline int32_t w13_dest_index(int32_t row, int32_t intermediate_size,
                              int32_t activation_id) {
  if (activation_id != kSwigluOAI) {
    return row;
  }
  const int32_t pair_idx = row / 2;
  return (row & 1) ? intermediate_size + pair_idx : pair_idx;
}

inline uint32_t hmx_lhs_crouton_bytes(int32_t rows, int32_t common_dim) {
  int shape[2] = {rows, common_dim};
  const QAicHMXDims<QAIC_HMX_FlatNXYD> flat_dims(shape, sizeof(float16),
                                                 HMXShapeKind::YD);
  const QAicHMXDims<QAIC_HMX_MATMUL_CROUTON_16B> crouton_dims(flat_dims);
  return (uint32_t)crouton_dims.sizeInBytes();
}

inline uint32_t hmx_out_crouton_bytes(int32_t rows, int32_t cols) {
  int shape[2] = {rows, cols};
  const QAicHMXDims<QAIC_HMX_FlatNXYD> flat_dims(shape, sizeof(float16),
                                                 HMXShapeKind::YD);
  const QAicHMXDims<QAIC_HMX_MATMUL_CROUTON_16B> crouton_dims(flat_dims);
  return (uint32_t)crouton_dims.sizeInBytes();
}

inline float16* alloc_aligned_hf(uint8_t* base, uint64_t* offset,
                                 uint32_t alignment, uint32_t bytes) {
  const uint64_t aligned_offset = align_up_u64(*offset, alignment);
  *offset = aligned_offset + bytes;
  return (float16*)(base + aligned_offset);
}

inline uint64_t hmx_required_vtcm_bytes(int32_t route_tile_rows,
                                        int32_t column_tile_cols,
                                        int32_t max_common_dim, int32_t w13_dim,
                                        int32_t intermediate_size) {
  uint64_t offset = 0;
  offset = align_up_u64(offset, sizeof(uint32_t));
  offset += kMaxKernelThreads * sizeof(uint32_t);
  offset = align_up_u64(offset, QAIC_HMX_INPUT_ALIGNMENT);
  offset += (uint64_t)route_tile_rows * max_common_dim * sizeof(float16);
  offset = align_up_u64(offset, QAIC_HMX_INPUT_ALIGNMENT);
  offset += (uint64_t)column_tile_cols * max_common_dim * sizeof(float16);
  offset = align_up_u64(offset, QAIC_HMX_ALIGNMENT);
  offset += hmx_lhs_crouton_bytes(route_tile_rows, max_common_dim);
  offset = align_up_u64(offset, QAIC_HMX_WEIGHT_ALIGNMENT);
  offset += qaic_compute_rhs_croutons_size_in_bytes(
      1, column_tile_cols, max_common_dim, sizeof(float16));
  offset = align_up_u64(offset, QAIC_HMX_ALIGNMENT);
  offset += hmx_out_crouton_bytes(route_tile_rows, column_tile_cols);
  offset = align_up_u64(offset, QAIC_HMX_INPUT_ALIGNMENT);
  offset += (uint64_t)route_tile_rows * column_tile_cols * sizeof(float16);
  offset = align_up_u64(offset, QAIC_HMX_INPUT_ALIGNMENT);
  offset += (uint64_t)route_tile_rows * w13_dim * sizeof(float16);
  offset = align_up_u64(offset, QAIC_HMX_INPUT_ALIGNMENT);
  offset += (uint64_t)route_tile_rows * intermediate_size * sizeof(float16);
  return offset;
}

inline bool init_hmx_scratch(uint8_t* vtcm_base_raw, uint32_t vtcm_size,
                             int32_t total_routes, int32_t hidden_size,
                             int32_t w13_dim, int32_t intermediate_size,
                             HmxScratch* scratch) {
  static constexpr int32_t kRowCandidates[] = {64, 32, 16, 8, 4, 2, 1};
  static constexpr int32_t kColCandidates[] = {64, 32, 16, 8, 4, 2, 1};
  const int32_t max_common_dim =
      hidden_size > intermediate_size ? hidden_size : intermediate_size;
  const int32_t max_output_dim = w13_dim > hidden_size ? w13_dim : hidden_size;
  uint8_t* vtcm_base = align_up_ptr(vtcm_base_raw, QAIC_HMX_ALIGNMENT);
  const uint64_t base_slop = (uint64_t)(vtcm_base - vtcm_base_raw);

  for (uint32_t row_i = 0;
       row_i < sizeof(kRowCandidates) / sizeof(kRowCandidates[0]); ++row_i) {
    int32_t route_rows = kRowCandidates[row_i];
    if (route_rows > total_routes) {
      route_rows = total_routes;
    }
    if (route_rows <= 0) {
      continue;
    }
    for (uint32_t col_i = 0;
         col_i < sizeof(kColCandidates) / sizeof(kColCandidates[0]); ++col_i) {
      int32_t column_cols = kColCandidates[col_i];
      if (column_cols > max_output_dim) {
        column_cols = max_output_dim;
      }
      if (column_cols <= 0) {
        continue;
      }
      const uint64_t required = hmx_required_vtcm_bytes(
          route_rows, column_cols, max_common_dim, w13_dim, intermediate_size);
      if (required + base_slop <= vtcm_size) {
        uint64_t offset = 0;
        offset = align_up_u64(offset, sizeof(uint32_t));
        scratch->status_words = (uint32_t*)(vtcm_base + offset);
        offset += kMaxKernelThreads * sizeof(uint32_t);
        scratch->lhs_rm =
            alloc_aligned_hf(vtcm_base, &offset, QAIC_HMX_INPUT_ALIGNMENT,
                             (uint32_t)((uint64_t)route_rows * max_common_dim *
                                        sizeof(float16)));
        scratch->rhs_cm =
            alloc_aligned_hf(vtcm_base, &offset, QAIC_HMX_INPUT_ALIGNMENT,
                             (uint32_t)((uint64_t)column_cols * max_common_dim *
                                        sizeof(float16)));
        scratch->lhs_crouton =
            alloc_aligned_hf(vtcm_base, &offset, QAIC_HMX_ALIGNMENT,
                             hmx_lhs_crouton_bytes(route_rows, max_common_dim));
        scratch->rhs_crouton = alloc_aligned_hf(
            vtcm_base, &offset, QAIC_HMX_WEIGHT_ALIGNMENT,
            qaic_compute_rhs_croutons_size_in_bytes(
                1, column_cols, max_common_dim, sizeof(float16)));
        scratch->out_crouton =
            alloc_aligned_hf(vtcm_base, &offset, QAIC_HMX_ALIGNMENT,
                             hmx_out_crouton_bytes(route_rows, column_cols));
        scratch->out_rm = alloc_aligned_hf(
            vtcm_base, &offset, QAIC_HMX_INPUT_ALIGNMENT,
            (uint32_t)((uint64_t)route_rows * column_cols * sizeof(float16)));
        scratch->gate_up_batch = alloc_aligned_hf(
            vtcm_base, &offset, QAIC_HMX_INPUT_ALIGNMENT,
            (uint32_t)((uint64_t)route_rows * w13_dim * sizeof(float16)));
        scratch->hidden_batch =
            alloc_aligned_hf(vtcm_base, &offset, QAIC_HMX_INPUT_ALIGNMENT,
                             (uint32_t)((uint64_t)route_rows *
                                        intermediate_size * sizeof(float16)));
        scratch->route_tile_rows = route_rows;
        scratch->column_tile_cols = column_cols;
        return true;
      }
    }
  }
  return false;
}

inline bool is_core_master_hvx_thread(uint32_t thread_id, uint32_t hmx_thread) {
  return thread_id == (hmx_thread == 0U ? 1U : 0U);
}

inline bool is_hmx_thread(uint32_t thread_id, uint32_t hmx_thread) {
  return thread_id == hmx_thread;
}

inline uint32_t local_hvx_thread_id(uint32_t thread_id, uint32_t hmx_thread) {
  return thread_id > hmx_thread ? thread_id - 1U : thread_id;
}

inline int32_t* shared_status_ptr(const HmxScratch& scratch) {
  return (int32_t*)scratch.status_words;
}

inline void copy_linear_hf_hvx(const float16* src, float16* dst,
                               int32_t elements, uint32_t hvx_thread_id,
                               uint32_t hvx_threads) {
  if (hvx_thread_id >= hvx_threads) {
    return;
  }
  constexpr int32_t kElemsPerHalfVector = sizeof(HVX_Vector) / sizeof(float16);
  const int32_t vector_elems =
      (elements / kElemsPerHalfVector) * kElemsPerHalfVector;
  for (int32_t offset = (int32_t)hvx_thread_id * kElemsPerHalfVector;
       offset < vector_elems;
       offset += (int32_t)hvx_threads * kElemsPerHalfVector) {
    HVX_Vector value = LoadUnaligned<HVX_Vector>((const int8_t*)(src + offset));
    StoreUnalignedHVX((int8_t*)(dst + offset), value);
  }
  const int32_t tail = elements - vector_elems;
  if (tail > 0 && hvx_thread_id == 0) {
    HVX_Vector value = LoadUnaligned<HVX_Vector>(
        (const int8_t*)(src + vector_elems), tail * (int32_t)sizeof(float16));
    StoreUnalignedHVX((int8_t*)(dst + vector_elems), value,
                      tail * (int32_t)sizeof(float16));
  }
}

inline void gather_route_lhs_hvx(const float16* x,
                                 const float* expert_route_indices,
                                 int32_t route_offset, int32_t actual_routes,
                                 int32_t topk, int32_t hidden_size,
                                 float16* lhs_rm, uint32_t hvx_thread_id,
                                 uint32_t hvx_threads) {
  if (hvx_thread_id >= hvx_threads) {
    return;
  }
  constexpr int32_t kElemsPerHalfVector = sizeof(HVX_Vector) / sizeof(float16);
  const int32_t vector_elems =
      (hidden_size / kElemsPerHalfVector) * kElemsPerHalfVector;
  const int32_t tail = hidden_size - vector_elems;

  for (int32_t row = (int32_t)hvx_thread_id; row < actual_routes;
       row += (int32_t)hvx_threads) {
    const int32_t route_idx = (int32_t)expert_route_indices[route_offset + row];
    const int32_t token = route_idx / topk;
    const float16* src_row = x + (int64_t)token * hidden_size;
    float16* dst_row = lhs_rm + (int64_t)row * hidden_size;
    for (int32_t h = 0; h < vector_elems; h += kElemsPerHalfVector) {
      HVX_Vector value =
          LoadUnaligned<HVX_Vector>((const int8_t*)(src_row + h));
      StoreUnalignedHVX((int8_t*)(dst_row + h), value);
    }
    if (tail > 0) {
      HVX_Vector value =
          LoadUnaligned<HVX_Vector>((const int8_t*)(src_row + vector_elems),
                                    tail * (int32_t)sizeof(float16));
      StoreUnalignedHVX((int8_t*)(dst_row + vector_elems), value,
                        tail * (int32_t)sizeof(float16));
    }
  }
}

inline HVX_Vector postprocess_hf_vector(HVX_Vector value_hf, HVX_Vector bias_hf,
                                        bool has_bias,
                                        HVX_Vector route_weight_sf,
                                        bool apply_route_weight,
                                        PostprocessOrder order) {
  HVX_VectorPair value_pair = Q6_Wsf_vcvt_Vhf(value_hf);
  HVX_Vector value_lo = Q6_V_lo_W(value_pair);
  HVX_Vector value_hi = Q6_V_hi_W(value_pair);

  if (apply_route_weight && order == kScaleThenBias) {
    value_lo = Q6_Vsf_vmpy_VsfVsf(value_lo, route_weight_sf);
    value_hi = Q6_Vsf_vmpy_VsfVsf(value_hi, route_weight_sf);
  }
  if (has_bias) {
    HVX_VectorPair bias_pair = Q6_Wsf_vcvt_Vhf(bias_hf);
    value_lo = Q6_Vsf_vadd_VsfVsf(value_lo, Q6_V_lo_W(bias_pair));
    value_hi = Q6_Vsf_vadd_VsfVsf(value_hi, Q6_V_hi_W(bias_pair));
  }
  if (apply_route_weight && order == kBiasThenScale) {
    value_lo = Q6_Vsf_vmpy_VsfVsf(value_lo, route_weight_sf);
    value_hi = Q6_Vsf_vmpy_VsfVsf(value_hi, route_weight_sf);
  }
  return Q6_Vhf_vcvt_VsfVsf(value_lo, value_hi);
}

inline void store_w13_tile_hvx(const float16* out_rm, float16* gate_up_batch,
                               const float16* w13_bias,
                               const float16* topk_weights,
                               const float* expert_route_indices,
                               int32_t route_offset, int32_t actual_routes,
                               int32_t col_start, int32_t cols, int32_t w13_dim,
                               int32_t intermediate_size, int32_t activation_id,
                               bool has_bias, bool apply_router_weight_on_input,
                               uint32_t hvx_thread_id, uint32_t hvx_threads) {
  if (hvx_thread_id >= hvx_threads) {
    return;
  }

  if (activation_id == kSwigluOAI) {
    constexpr int32_t kElemsPerHalfVector =
        sizeof(HVX_Vector) / sizeof(float16);
    for (int32_t row = (int32_t)hvx_thread_id; row < actual_routes;
         row += (int32_t)hvx_threads) {
      const int32_t route_idx =
          (int32_t)expert_route_indices[route_offset + row];
      const float route_weight = (float)topk_weights[route_idx];
      HVX_Vector route_weight_sf = splat_sf(route_weight);
      const float16* src_row = out_rm + (int64_t)row * cols;
      float16* dst_row = gate_up_batch + (int64_t)row * w13_dim;
      const float16* bias_row = has_bias ? w13_bias + col_start : nullptr;

      int32_t col = 0;
      for (; col + 1 < cols;) {
        int32_t chunk_cols = cols - col;
        if (chunk_cols > kElemsPerHalfVector) {
          chunk_cols = kElemsPerHalfVector;
        }
        chunk_cols &= ~1;
        const int32_t pairs = chunk_cols / 2;
        const int32_t bytes = chunk_cols * (int32_t)sizeof(float16);
        const int32_t store_bytes = pairs * (int32_t)sizeof(float16);

        HVX_Vector value_hf =
            LoadUnaligned<HVX_Vector>((const int8_t*)(src_row + col), bytes);
        HVX_Vector bias_hf = has_bias
                                 ? LoadUnaligned<HVX_Vector>(
                                       (const int8_t*)(bias_row + col), bytes)
                                 : Q6_V_vzero();
        HVX_Vector out_hf =
            postprocess_hf_vector(value_hf, bias_hf, has_bias, route_weight_sf,
                                  apply_router_weight_on_input, kScaleThenBias);

        HVX_VectorB out_vb = out_hf;
        HVX_VectorB even_hf =
            qaic_shufflevectorB(out_vb, out_vb, QAIC_MOE_SHUFFLE_EVEN_HF);
        HVX_VectorB odd_hf =
            qaic_shufflevectorB(out_vb, out_vb, QAIC_MOE_SHUFFLE_ODD_HF);

        const int32_t w13_col = col_start + col;
        const int32_t pair_start = w13_col / 2;
        if ((w13_col & 1) == 0) {
          StoreUnalignedHVX((int8_t*)(dst_row + pair_start), even_hf,
                            store_bytes);
          StoreUnalignedHVX((int8_t*)(dst_row + intermediate_size + pair_start),
                            odd_hf, store_bytes);
        } else {
          StoreUnalignedHVX((int8_t*)(dst_row + intermediate_size + pair_start),
                            even_hf, store_bytes);
          StoreUnalignedHVX((int8_t*)(dst_row + pair_start + 1), odd_hf,
                            store_bytes);
        }
        col += chunk_cols;
      }

      if (col < cols) {
        const int32_t w13_col = col_start + col;
        HVX_Vector value_hf = LoadUnaligned<HVX_Vector>(
            (const int8_t*)(src_row + col), (int32_t)sizeof(float16));
        HVX_Vector bias_hf = has_bias ? LoadUnaligned<HVX_Vector>(
                                            (const int8_t*)(w13_bias + w13_col),
                                            (int32_t)sizeof(float16))
                                      : Q6_V_vzero();
        HVX_Vector out_hf =
            postprocess_hf_vector(value_hf, bias_hf, has_bias, route_weight_sf,
                                  apply_router_weight_on_input, kScaleThenBias);
        StoreUnalignedHVX(
            (int8_t*)(dst_row +
                      w13_dest_index(w13_col, intermediate_size, kSwigluOAI)),
            out_hf, (int32_t)sizeof(float16));
      }
    }
    return;
  }

  constexpr int32_t kElemsPerHalfVector = sizeof(HVX_Vector) / sizeof(float16);
  const int32_t vector_elems =
      (cols / kElemsPerHalfVector) * kElemsPerHalfVector;
  const int32_t tail = cols - vector_elems;
  for (int32_t row = (int32_t)hvx_thread_id; row < actual_routes;
       row += (int32_t)hvx_threads) {
    const int32_t route_idx = (int32_t)expert_route_indices[route_offset + row];
    const float route_weight = (float)topk_weights[route_idx];
    HVX_Vector route_weight_sf = splat_sf(route_weight);
    const float16* src_row = out_rm + (int64_t)row * cols;
    float16* dst_row = gate_up_batch + (int64_t)row * w13_dim + col_start;
    const float16* bias_row = has_bias ? w13_bias + col_start : nullptr;

    for (int32_t col = 0; col < vector_elems; col += kElemsPerHalfVector) {
      HVX_Vector value_hf =
          LoadUnaligned<HVX_Vector>((const int8_t*)(src_row + col));
      HVX_Vector bias_hf =
          has_bias ? LoadUnaligned<HVX_Vector>((const int8_t*)(bias_row + col))
                   : Q6_V_vzero();
      HVX_Vector out_hf =
          postprocess_hf_vector(value_hf, bias_hf, has_bias, route_weight_sf,
                                apply_router_weight_on_input, kScaleThenBias);
      StoreUnalignedHVX((int8_t*)(dst_row + col), out_hf);
    }
    if (tail > 0) {
      HVX_Vector value_hf = load_partial_hf_zero(src_row + vector_elems, tail);
      HVX_Vector bias_hf =
          has_bias ? load_partial_hf_zero(bias_row + vector_elems, tail)
                   : Q6_V_vzero();
      HVX_Vector out_hf =
          postprocess_hf_vector(value_hf, bias_hf, has_bias, route_weight_sf,
                                apply_router_weight_on_input, kScaleThenBias);
      StoreUnalignedHVX((int8_t*)(dst_row + vector_elems), out_hf,
                        tail * (int32_t)sizeof(float16));
    }
  }
}

inline void apply_activation_batch_hvx(
    const float16* gate_up_batch, float16* hidden_batch, int32_t actual_routes,
    int32_t w13_dim, int32_t intermediate_size, int32_t activation_id,
    uint32_t hvx_thread_id, uint32_t hvx_threads) {
  if (hvx_thread_id >= hvx_threads) {
    return;
  }
  for (int32_t row = (int32_t)hvx_thread_id; row < actual_routes;
       row += (int32_t)hvx_threads) {
    apply_activation_vec(gate_up_batch + (int64_t)row * w13_dim,
                         hidden_batch + (int64_t)row * intermediate_size,
                         intermediate_size, activation_id);
  }
}

inline void scatter_w2_tile_hvx(const float16* out_rm, float16* route_out,
                                const float16* w2_bias,
                                const float16* topk_weights,
                                const float* expert_route_indices,
                                int32_t route_offset, int32_t actual_routes,
                                int32_t col_start, int32_t cols,
                                int32_t hidden_size, bool has_bias,
                                bool apply_router_weight_on_input,
                                uint32_t hvx_thread_id, uint32_t hvx_threads) {
  if (hvx_thread_id >= hvx_threads) {
    return;
  }

  constexpr int32_t kElemsPerHalfVector = sizeof(HVX_Vector) / sizeof(float16);
  const int32_t vector_elems =
      (cols / kElemsPerHalfVector) * kElemsPerHalfVector;
  const int32_t tail = cols - vector_elems;
  for (int32_t row = (int32_t)hvx_thread_id; row < actual_routes;
       row += (int32_t)hvx_threads) {
    const int32_t route_idx = (int32_t)expert_route_indices[route_offset + row];
    const float route_weight = (float)topk_weights[route_idx];
    HVX_Vector route_weight_sf = splat_sf(route_weight);
    const float16* src_row = out_rm + (int64_t)row * cols;
    float16* dst_row = route_out + (int64_t)route_idx * hidden_size + col_start;
    const float16* bias_row = has_bias ? w2_bias + col_start : nullptr;

    for (int32_t col = 0; col < vector_elems; col += kElemsPerHalfVector) {
      HVX_Vector value_hf =
          LoadUnaligned<HVX_Vector>((const int8_t*)(src_row + col));
      HVX_Vector bias_hf =
          has_bias ? LoadUnaligned<HVX_Vector>((const int8_t*)(bias_row + col))
                   : Q6_V_vzero();
      HVX_Vector out_hf =
          postprocess_hf_vector(value_hf, bias_hf, has_bias, route_weight_sf,
                                !apply_router_weight_on_input, kBiasThenScale);
      StoreUnalignedHVX((int8_t*)(dst_row + col), out_hf);
    }
    if (tail > 0) {
      HVX_Vector value_hf = load_partial_hf_zero(src_row + vector_elems, tail);
      HVX_Vector bias_hf =
          has_bias ? load_partial_hf_zero(bias_row + vector_elems, tail)
                   : Q6_V_vzero();
      HVX_Vector out_hf =
          postprocess_hf_vector(value_hf, bias_hf, has_bias, route_weight_sf,
                                !apply_router_weight_on_input, kBiasThenScale);
      StoreUnalignedHVX((int8_t*)(dst_row + vector_elems), out_hf,
                        tail * (int32_t)sizeof(float16));
    }
  }
}

inline __attribute__((always_inline)) uint32_t run_hmx_matmul_rhs_cw_tile(
    const float16* lhs_rm_src, float16* lhs_crouton, const float16* rhs_cm_src,
    float16* rhs_cm, float16* rhs_crouton, float16* out_crouton,
    float16* out_rm, int32_t rows, int32_t common_dim, int32_t cols,
    const HmxScratch& scratch, uint32_t thread_id, uint32_t hmx_thread,
    uint32_t hvx_thread_id, uint32_t hvx_threads) {
  int weight_shape[3] = {1, common_dim, cols};
  int lhs_shape[2] = {rows, common_dim};
  const QAicHMXDims<QAIC_HMX_FlatNXYD> lhs_dims(lhs_shape, sizeof(float16),
                                                HMXShapeKind::YD);
  const QAicHMXDims<QAIC_HMX_MATMUL_CROUTON_16B> lhs_crouton_dims(lhs_dims);
  int out_shape[2] = {rows, cols};
  const QAicHMXDims<QAIC_HMX_FlatNXYD> out_dims(out_shape, sizeof(float16),
                                                HMXShapeKind::YD);
  const QAicHMXDims<QAIC_HMX_MATMUL_CROUTON_16B> out_crouton_dims(out_dims);

  int32_t* shared_status = shared_status_ptr(scratch);
  if (is_core_master_hvx_thread(thread_id, hmx_thread)) {
    *shared_status = JIT_DEV_STATUS_SUCCESS;
  }

  copy_linear_hf_hvx(rhs_cm_src, rhs_cm, common_dim * cols, hvx_thread_id,
                     hvx_threads);
  qaicSyncHVXAndHMXThreads(thread_id);

  int32_t local_status = JIT_DEV_STATUS_SUCCESS;
  if (thread_id != hmx_thread) {
    JitDevStatusCode_t ret = qaic_hmx_cm_matmul_rhs_to_crouton_16b(
        (int16_t*)rhs_crouton, (const int16_t*)rhs_cm, weight_shape, thread_id);
    if (ret != JIT_DEV_STATUS_SUCCESS) {
      local_status = ret;
    }

    if (local_status == JIT_DEV_STATUS_SUCCESS) {
      ret = qaic_hmx_rm_matmul_lhs_to_crouton_16b(
          (int16_t*)lhs_crouton, (const int16_t*)lhs_rm_src, &lhs_crouton_dims,
          &lhs_dims, thread_id);
      if (ret != JIT_DEV_STATUS_SUCCESS) {
        local_status = ret;
      }
    }

    if (local_status != JIT_DEV_STATUS_SUCCESS) {
      *shared_status = local_status;
    }
  }
  qaicSyncHVXAndHMXThreads(thread_id);

  if (thread_id == hmx_thread && *shared_status == JIT_DEV_STATUS_SUCCESS) {
    const float16* lhs_inputs[1] = {lhs_crouton};
    qaic_hmx_matmul_with_crouton_16b(
        out_crouton, lhs_inputs, rhs_crouton, &out_dims, &out_crouton_dims,
        &lhs_crouton_dims, weight_shape, true, true, false);
  }
  qaicSyncHVXAndHMXThreads(thread_id);

  if (thread_id != hmx_thread && *shared_status == JIT_DEV_STATUS_SUCCESS) {
    JitDevStatusCode_t ret = qaic_hmx_matmul_crouton_to_rm_16b(
        (int16_t*)out_rm, (const int16_t*)out_crouton, &out_dims,
        &out_crouton_dims, thread_id);
    if (ret != JIT_DEV_STATUS_SUCCESS) {
      *shared_status = ret;
    }
  }
  qaicSyncHVXAndHMXThreads(thread_id);

  return (uint32_t)(*shared_status);
}

inline __attribute__((always_inline)) uint32_t process_expert_routes_hmx(
    const float16* x, const float16* topk_weights, const float16* w13_weight,
    const float16* w2_weight, const float16* w13_bias, const float16* w2_bias,
    float16* route_out, const float* expert_route_indices, int32_t route_begin,
    int32_t route_count, int32_t topk, int32_t hidden_size, int32_t w13_dim,
    int32_t intermediate_size, int32_t activation_id, bool has_bias,
    bool apply_router_weight_on_input, const HmxScratch& scratch,
    uint32_t thread_id, uint32_t hmx_thread, uint32_t hvx_thread_id,
    uint32_t hvx_threads) {
  if (route_count <= 0) {
    return JIT_DEV_STATUS_SUCCESS;
  }

  const int32_t route_tile_rows = scratch.route_tile_rows;
  const int32_t column_tile_cols = scratch.column_tile_cols;

  for (int32_t route_offset = 0; route_offset < route_count;
       route_offset += route_tile_rows) {
    int32_t actual_routes = route_tile_rows;
    if (actual_routes > route_count - route_offset) {
      actual_routes = route_count - route_offset;
    }

    gather_route_lhs_hvx(x, expert_route_indices, route_begin + route_offset,
                         actual_routes, topk, hidden_size, scratch.lhs_rm,
                         hvx_thread_id, hvx_threads);
    qaicSyncHVXAndHMXThreads(thread_id);

    for (int32_t col_start = 0; col_start < w13_dim;
         col_start += column_tile_cols) {
      int32_t cols = column_tile_cols;
      if (cols > w13_dim - col_start) {
        cols = w13_dim - col_start;
      }
      uint32_t status = run_hmx_matmul_rhs_cw_tile(
          scratch.lhs_rm, scratch.lhs_crouton,
          w13_weight + (int64_t)col_start * hidden_size, scratch.rhs_cm,
          scratch.rhs_crouton, scratch.out_crouton, scratch.out_rm,
          actual_routes, hidden_size, cols, scratch, thread_id, hmx_thread,
          hvx_thread_id, hvx_threads);
      if (status != JIT_DEV_STATUS_SUCCESS) {
        return status;
      }
      store_w13_tile_hvx(
          scratch.out_rm, scratch.gate_up_batch, has_bias ? w13_bias : nullptr,
          topk_weights, expert_route_indices, route_begin + route_offset,
          actual_routes, col_start, cols, w13_dim, intermediate_size,
          activation_id, has_bias, apply_router_weight_on_input, hvx_thread_id,
          hvx_threads);
      qaicSyncHVXAndHMXThreads(thread_id);
    }

    apply_activation_batch_hvx(scratch.gate_up_batch, scratch.hidden_batch,
                               actual_routes, w13_dim, intermediate_size,
                               activation_id, hvx_thread_id, hvx_threads);
    qaicSyncHVXAndHMXThreads(thread_id);

    for (int32_t col_start = 0; col_start < hidden_size;
         col_start += column_tile_cols) {
      int32_t cols = column_tile_cols;
      if (cols > hidden_size - col_start) {
        cols = hidden_size - col_start;
      }
      uint32_t status = run_hmx_matmul_rhs_cw_tile(
          scratch.hidden_batch, scratch.lhs_crouton,
          w2_weight + (int64_t)col_start * intermediate_size, scratch.rhs_cm,
          scratch.rhs_crouton, scratch.out_crouton, scratch.out_rm,
          actual_routes, intermediate_size, cols, scratch, thread_id,
          hmx_thread, hvx_thread_id, hvx_threads);
      if (status != JIT_DEV_STATUS_SUCCESS) {
        return status;
      }
      scatter_w2_tile_hvx(
          scratch.out_rm, route_out, has_bias ? w2_bias : nullptr, topk_weights,
          expert_route_indices, route_begin + route_offset, actual_routes,
          col_start, cols, hidden_size, has_bias, apply_router_weight_on_input,
          hvx_thread_id, hvx_threads);
    }
  }
  return JIT_DEV_STATUS_SUCCESS;
}

inline uint32_t route_compute_hmx_kernel_main(const AicJitEntryPointConfig* cfg,
                                              const AicJitPointerArray* ptrs) {
  if (!valid_jit_args(cfg, ptrs, 11)) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  const float* params = pointer_arg<const float*>(ptrs, 10);
  if (params == nullptr) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  const int32_t num_tokens = param_i32(params, 0);
  const int32_t hidden_size = param_i32(params, 1);
  const int32_t w13_dim = param_i32(params, 2);
  const int32_t intermediate_size = param_i32(params, 3);
  const int32_t num_experts = param_i32(params, 4);
  const int32_t topk = param_i32(params, 5);
  const int32_t activation_id = param_i32(params, 6);
  const bool has_bias = param_bool(params, 7);
  const bool apply_router_weight_on_input = param_bool(params, 8);
  const int32_t global_num_experts = param_i32(params, 9);
  const bool has_expert_map = param_bool(params, 10);

  if (num_tokens < 0 || hidden_size <= 0 || intermediate_size <= 0 ||
      num_experts <= 0 || topk <= 0 || global_num_experts <= 0 ||
      !valid_activation(activation_id)) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  if (num_tokens == 0) {
    return JIT_DEV_STATUS_SUCCESS;
  }

  const float16* x = pointer_arg<const float16*>(ptrs, 0);
  const float16* topk_weights = pointer_arg<const float16*>(ptrs, 1);
  const float16* topk_ids = pointer_arg<const float16*>(ptrs, 2);
  const float16* w13_weight = pointer_arg<const float16*>(ptrs, 3);
  const float16* w2_weight = pointer_arg<const float16*>(ptrs, 4);
  const float16* bias = pointer_arg<const float16*>(ptrs, 5);
  float16* route_out = pointer_arg<float16*>(ptrs, 6);
  const float* expert_route_indices = pointer_arg<const float*>(ptrs, 7);
  const float* expert_offsets = pointer_arg<const float*>(ptrs, 8);
  const float* expert_map = pointer_arg<const float*>(ptrs, 9);

  const int32_t expected_w13_dim = activation_is_no_mul(activation_id)
                                       ? intermediate_size
                                       : 2 * intermediate_size;
  if (w13_dim != expected_w13_dim) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  int64_t vtcm_size_i64 = 0;
  uint32_t status = qshimQuery(DEV_ATTR_QSHIM_VTCM_SIZE, &vtcm_size_i64);
  if (status != JIT_DEV_STATUS_SUCCESS) {
    return status;
  }

  int64_t hmx_thread_i64 = 0;
  status = qshimQuery(DEV_ATTR_QSHIM_HMX_THREAD_ID, &hmx_thread_i64);
  if (status != JIT_DEV_STATUS_SUCCESS) {
    return status;
  }

  int64_t hvx_threads_i64 = 0;
  status = qshimQuery(DEV_ATTR_QSHIM_NUM_HVX_UNITS, &hvx_threads_i64);
  if (status != JIT_DEV_STATUS_SUCCESS) {
    return status;
  }

  const uint32_t thread_id = cfg->threadID;
  const uint32_t hmx_thread = (uint32_t)hmx_thread_i64;
  const uint32_t hvx_threads = (uint32_t)hvx_threads_i64;
  const uint32_t num_threads = cfg->numThreads;
  const int32_t total_routes = num_tokens * topk;

  if (hvx_threads == 0 || hmx_thread_i64 < 0 || num_threads <= hmx_thread ||
      num_threads < hvx_threads + 1U || num_threads > kMaxKernelThreads ||
      cfg->numCores == 0) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }
  const bool current_thread_is_hmx = is_hmx_thread(thread_id, hmx_thread);
  const uint32_t hvx_thread_id =
      current_thread_is_hmx ? hvx_threads
                            : local_hvx_thread_id(thread_id, hmx_thread);

  if (current_thread_is_hmx) {
    qaic_hmx_matmul_bias_init_16b(false);
  }
  qaicSyncHVXAndHMXThreads(thread_id);

  HmxScratch scratch = {};
  if (!init_hmx_scratch(qshimGetBaseVtcmAddr(), (uint32_t)vtcm_size_i64,
                        total_routes, hidden_size, w13_dim, intermediate_size,
                        &scratch)) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  if (!current_thread_is_hmx && hvx_thread_id < hvx_threads) {
    const int32_t workers = (int32_t)(cfg->numCores * hvx_threads);
    const int32_t worker_id =
        (int32_t)(cfg->coreID * hvx_threads + hvx_thread_id);
    for (int32_t route_idx = worker_id; route_idx < total_routes;
         route_idx += workers) {
      const int32_t global_expert = (int32_t)topk_ids[route_idx];
      const int32_t local_expert =
          map_global_to_local_expert(global_expert, expert_map, num_experts,
                                     global_num_experts, has_expert_map);
      if (local_expert < 0) {
        zero_route_out(route_out + (int64_t)route_idx * hidden_size,
                       hidden_size);
      }
    }
  }
  qaicSyncHVXAndHMXThreads(thread_id);

  for (int32_t expert = 0; expert < num_experts; ++expert) {
    const int32_t route_begin = (int32_t)expert_offsets[expert];
    const int32_t route_count =
        (int32_t)(expert_offsets[expert + 1] - expert_offsets[expert]);
    if (route_count <= 0) {
      continue;
    }

    const int32_t routes_per_core =
        (route_count + (int32_t)cfg->numCores - 1) / (int32_t)cfg->numCores;
    const int32_t core_route_offset = (int32_t)cfg->coreID * routes_per_core;
    if (core_route_offset >= route_count) {
      continue;
    }
    int32_t local_route_count = route_count - core_route_offset;
    if (local_route_count > routes_per_core) {
      local_route_count = routes_per_core;
    }

    const float16* expert_w13 =
        w13_weight + ((int64_t)expert * w13_dim * hidden_size);
    const float16* expert_w2 =
        w2_weight + ((int64_t)expert * hidden_size * intermediate_size);
    const float16* expert_bias =
        has_bias ? bias + ((int64_t)expert * w13_dim) : nullptr;
    const float16* expert_w2_bias =
        has_bias ? bias + ((int64_t)num_experts * w13_dim) +
                       ((int64_t)expert * hidden_size)
                 : nullptr;

    status = process_expert_routes_hmx(
        x, topk_weights, expert_w13, expert_w2, expert_bias, expert_w2_bias,
        route_out, expert_route_indices, route_begin + core_route_offset,
        local_route_count, topk, hidden_size, w13_dim, intermediate_size,
        activation_id, has_bias, apply_router_weight_on_input, scratch,
        thread_id, hmx_thread, hvx_thread_id, hvx_threads);
    if (status != JIT_DEV_STATUS_SUCCESS) {
      return status;
    }
  }

  return JIT_DEV_STATUS_SUCCESS;
}

}  // namespace unquantized_fused_moe_hmx

QAIC_KERNEL_API int32_t
multinsp_multithreaded_unquantized_fused_moe_route_compute_hmx(
    const AicJitEntryPointConfig* entryConfig,
    const AicJitPointerArray* pointerArray) {
  return unquantized_fused_moe_hmx::route_compute_hmx_kernel_main(entryConfig,
                                                                  pointerArray);
}
