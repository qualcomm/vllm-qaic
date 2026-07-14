// ---------------------------------------------------------------------------------------
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
// ---------------------------------------------------------------------------------------

#include <math.h>
#include <stdint.h>

#include "QAicHexagonHVX.h"
#include "QAicHexagonMath.h"
#include "QAicHexagonPlatformIntf.h"
#include "QAicHexagonReducer.h"
#include "QAicHexagonTypes.h"
#include "QAicHexagonUtils.h"
#include "jit_dev_exe_function.h"
#include "jit_dev_status_codes.h"
#include "jit_qshim_api.h"

namespace unquantized_fused_moe_route_reduce {

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

inline uint32_t align_up_u32(uint32_t value, uint32_t alignment) {
  return (value + alignment - 1U) & ~(alignment - 1U);
}

inline uint32_t align_down_u32(uint32_t value, uint32_t alignment) {
  return alignment == 0U ? value : (value / alignment) * alignment;
}

inline uint8_t* align_up_ptr(uint8_t* ptr, uintptr_t alignment) {
  const uintptr_t value = (uintptr_t)ptr;
  return (uint8_t*)((value + alignment - 1U) & ~(alignment - 1U));
}

inline uint32_t wait_dma_handle(QShimUDmaHandle handle, uint32_t submit_status) {
  if (submit_status != JIT_DEV_STATUS_SUCCESS) {
    return submit_status;
  }
  if (handle == INVALID_UDMA_HANDLE) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }
  return qshimUDmaWait(handle);
}

inline bool activation_is_no_mul(int32_t activation_id) {
  return activation_id == kSiluNoMul || activation_id == kGeluNoMul ||
         activation_id == kGeluTanhNoMul || activation_id == kRelu2NoMul;
}

inline bool valid_activation(int32_t activation_id) {
  return activation_id >= kSilu && activation_id <= kRelu2NoMul;
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
    return (local_expert >= 0 && local_expert < local_num_experts) ? local_expert : -1;
  }
  return (global_expert >= 0 && global_expert < local_num_experts) ? global_expert : -1;
}





inline HVX_Vector splat_hf(float value) {
  float16 value_hf = (float16)value;
  return Q6_Vh_vsplat_R(*(uint16_t*)&value_hf);
}

inline HVX_Vector splat_sf(float value) {
  return Q6_V_vsplat_R(*(uint32_t*)&value);
}

inline float reduce_sum_vsf_pair(HVX_VectorPair acc_pair) {
  float acc = 0.0F;
  SumReducerFloat reducer;
  reducer.reduce(Q6_V_lo_W(acc_pair));
  reducer.reduce(Q6_V_hi_W(acc_pair));
  reducer.finish(&acc);
  return acc;
}

inline float dot_hf_hf_to_float(const float16* lhs,
                                const float16* rhs,
                                int32_t size) {
  constexpr int32_t kElemsPerHalfVector = sizeof(HVX_Vector) / sizeof(float16);
  const int32_t full_vectors = size / kElemsPerHalfVector;
  const int32_t vector_elems = full_vectors * kElemsPerHalfVector;

  HVX_Vector acc_lo = Q6_V_vzero();
  HVX_Vector acc_hi = Q6_V_vzero();
  for (int32_t offset = 0; offset < vector_elems; offset += kElemsPerHalfVector) {
    HVX_Vector lhs_vhf = LoadUnaligned<HVX_Vector>((const int8_t*)(lhs + offset));
    HVX_Vector rhs_vhf = LoadUnaligned<HVX_Vector>((const int8_t*)(rhs + offset));
    HVX_VectorPair lhs_pair = Q6_Wsf_vcvt_Vhf(lhs_vhf);
    HVX_VectorPair rhs_pair = Q6_Wsf_vcvt_Vhf(rhs_vhf);
    acc_lo = Q6_Vsf_vadd_VsfVsf(
        acc_lo, Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(lhs_pair), Q6_V_lo_W(rhs_pair)));
    acc_hi = Q6_Vsf_vadd_VsfVsf(
        acc_hi, Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(lhs_pair), Q6_V_hi_W(rhs_pair)));
  }

  float acc = reduce_sum_vsf_pair(Q6_W_vcombine_VV(acc_hi, acc_lo));
  for (int32_t i = vector_elems; i < size; ++i) {
    acc += (float)lhs[i] * (float)rhs[i];
  }
  return acc;
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

inline HVX_Vector clamp_vec_hf(HVX_Vector value_vhf, float min_value, float max_value) {
  HVX_Vector min_vhf = splat_hf(min_value);
  HVX_Vector max_vhf = splat_hf(max_value);
  HVX_VectorPred too_low = Q6_Q_vcmp_gt_VhfVhf(min_vhf, value_vhf);
  HVX_VectorPred too_high = Q6_Q_vcmp_gt_VhfVhf(value_vhf, max_vhf);
  value_vhf = Q6_V_vmux_QVV(too_low, min_vhf, value_vhf);
  return Q6_V_vmux_QVV(too_high, max_vhf, value_vhf);
}


inline void apply_activation_vec(const float16* gate_up,
                                 float16* hidden,
                                 int32_t intermediate_size,
                                 int32_t activation_id) {
  constexpr int32_t kElemsPerHalfVector = sizeof(HVX_Vector) / sizeof(float16);
  const int32_t full_vectors = intermediate_size / kElemsPerHalfVector;
  const int32_t vector_elems = full_vectors * kElemsPerHalfVector;

  for (int32_t offset = 0; offset < vector_elems; offset += kElemsPerHalfVector) {
    HVX_Vector gate_vhf = LoadUnaligned<HVX_Vector>((const int8_t*)(gate_up + offset));
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
        HVX_Vector swish_gate = Q6_Vhf_vmpy_VhfVhf(gate_vhf, qaic_sigmoid_hf(sigmoid_arg));
        out_vhf = Q6_Vhf_vmpy_VhfVhf(
            Q6_Vhf_vadd_VhfVhf(up_vhf, splat_hf(1.0F)), swish_gate);
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

  for (int32_t i = vector_elems; i < intermediate_size; ++i) {
    float value;
    if (activation_id == kSiluNoMul) {
      const float gate = (float)gate_up[i];
      value = gate * (0.5F * tanhf(0.5F * gate) + 0.5F);
    } else if (activation_id == kGeluNoMul) {
      const float gate = (float)gate_up[i];
      value = 0.5F * gate * (1.0F + erff(gate * 0.7071067811865475F));
    } else if (activation_id == kGeluTanhNoMul) {
      const float gate = (float)gate_up[i];
      const float gate3 = gate * gate * gate;
      value = 0.5F * gate *
              (1.0F + tanhf(0.7978845608028654F *
                             (gate + 0.044715F * gate3)));
    } else if (activation_id == kRelu2NoMul) {
      value = (float)gate_up[i];
      value = value > 0.0F ? value : 0.0F;
      value *= value;
    } else {
      float gate = (float)gate_up[i];
      float up = (float)gate_up[intermediate_size + i];
      if (activation_id == kSwigluOAI) {
        gate = gate < 7.0F ? gate : 7.0F;
        if (up < -7.0F) {
          up = -7.0F;
        } else if (up > 7.0F) {
          up = 7.0F;
        }
        const float sigmoid_arg = 1.702F * gate;
        value = (up + 1.0F) * gate *
                (0.5F * tanhf(0.5F * sigmoid_arg) + 0.5F);
      } else if (activation_id == kSwigluStep) {
        gate = gate * (0.5F * tanhf(0.5F * gate) + 0.5F);
        gate = gate < 7.0F ? gate : 7.0F;
        if (up < -7.0F) {
          up = -7.0F;
        } else if (up > 7.0F) {
          up = 7.0F;
        }
        value = gate * up;
      } else {
        float activated;
        if (activation_id == kSilu) {
          activated = gate * (0.5F * tanhf(0.5F * gate) + 0.5F);
        } else if (activation_id == kGelu) {
          activated = 0.5F * gate * (1.0F + erff(gate * 0.7071067811865475F));
        } else {
          const float gate3 = gate * gate * gate;
          activated = 0.5F * gate *
                      (1.0F + tanhf(0.7978845608028654F *
                                     (gate + 0.044715F * gate3)));
        }
        value = activated * up;
      }
    }
    hidden[i] = (float16)value;
  }
}

inline void zero_route_out(float16* route_out, int32_t hidden_size) {
  constexpr int32_t kElemsPerHalfVector = sizeof(HVX_Vector) / sizeof(float16);
  const int32_t full_vectors = hidden_size / kElemsPerHalfVector;
  const int32_t vector_elems = full_vectors * kElemsPerHalfVector;
  HVX_Vector zero = Q6_V_vzero();
  for (int32_t offset = 0; offset < vector_elems; offset += kElemsPerHalfVector) {
    StoreUnalignedHVX((int8_t*)(route_out + offset), zero);
  }
  for (int32_t h = vector_elems; h < hidden_size; ++h) {
    route_out[h] = (float16)0.0F;
  }
}


inline int32_t w13_dest_index(int32_t row,
                              int32_t intermediate_size,
                              int32_t activation_id) {
  if (activation_id != kSwigluOAI) {
    return row;
  }
  const int32_t pair_idx = row / 2;
  return (row & 1) ? intermediate_size + pair_idx : pair_idx;
}

inline uint32_t compute_gate_up_batch_tiled(const float16* const* token_ptrs,
                                            const float* route_weights,
                                            int32_t batch_size,
                                            const float16* w13,
                                            const float16* w13_bias,
                                            float16* gate_up_batch,
                                            float16* weight_tile,
                                            uint32_t weight_tile_bytes,
                                            uint32_t thread_id,
                                            int32_t hidden_size,
                                            int32_t w13_dim,
                                            int32_t intermediate_size,
                                            int32_t activation_id,
                                            bool has_bias,
                                            bool apply_router_weight_on_input) {
  const uint32_t row_bytes = (uint32_t)hidden_size * sizeof(float16);
  const int32_t tile_rows = weight_tile_bytes >= row_bytes
                                ? (int32_t)(weight_tile_bytes / row_bytes)
                                : 0;

  if (tile_rows <= 0 || weight_tile == nullptr) {
    for (int32_t row = 0; row < w13_dim; ++row) {
      const float16* weight_row = w13 + (int64_t)row * hidden_size;
      const int32_t dst_row = w13_dest_index(row, intermediate_size, activation_id);
      for (int32_t batch = 0; batch < batch_size; ++batch) {
        float value = dot_hf_hf_to_float(token_ptrs[batch], weight_row, hidden_size);
        if (apply_router_weight_on_input) {
          value *= route_weights[batch];
        }
        if (has_bias) {
          value += (float)w13_bias[row];
        }
        gate_up_batch[(int64_t)batch * w13_dim + dst_row] = (float16)value;
      }
    }
    return JIT_DEV_STATUS_SUCCESS;
  }

  float16* tile_buffers[2] = {weight_tile, weight_tile + (int64_t)weight_tile_bytes / sizeof(float16)};
  const int32_t num_tiles = (w13_dim + tile_rows - 1) / tile_rows;
  QShimUDmaHandle handles[2] = {INVALID_UDMA_HANDLE, INVALID_UDMA_HANDLE};
  uint32_t submit_status[2] = {JIT_DEV_STATUS_SUCCESS, JIT_DEV_STATUS_SUCCESS};

  int32_t first_rows = tile_rows;
  if (first_rows > w13_dim) {
    first_rows = w13_dim;
  }
  {
    AicJitUdmaDescCommonAttrs attrs = {};
    attrs.order = 0;
    handles[0] = qshimLinearUdmaSubmit(thread_id,
                                       (AicJitPtr)w13,
                                       (uint32_t)first_rows * row_bytes,
                                       (AicJitPtr)tile_buffers[0],
                                       &attrs,
                                       true,
                                       &submit_status[0]);
  }

  for (int32_t tile = 0; tile < num_tiles; ++tile) {
    const int32_t cur = tile & 1;
    const int32_t next = cur ^ 1;
    const int32_t row_start = tile * tile_rows;
    int32_t rows = tile_rows;
    if (rows > w13_dim - row_start) {
      rows = w13_dim - row_start;
    }

    uint32_t status = wait_dma_handle(handles[cur], submit_status[cur]);
    if (status != JIT_DEV_STATUS_SUCCESS) {
      return status;
    }

    const int32_t next_tile = tile + 1;
    if (next_tile < num_tiles) {
      const int32_t next_row_start = next_tile * tile_rows;
      int32_t next_rows = tile_rows;
      if (next_rows > w13_dim - next_row_start) {
        next_rows = w13_dim - next_row_start;
      }
      const float16* next_src = w13 + (int64_t)next_row_start * hidden_size;
      {
        AicJitUdmaDescCommonAttrs attrs = {};
        attrs.order = 0;
        handles[next] = qshimLinearUdmaSubmit(thread_id,
                                             (AicJitPtr)next_src,
                                             (uint32_t)next_rows * row_bytes,
                                             (AicJitPtr)tile_buffers[next],
                                             &attrs,
                                             true,
                                             &submit_status[next]);
      }
    }

    for (int32_t local_row = 0; local_row < rows; ++local_row) {
      const int32_t row = row_start + local_row;
      const float16* weight_row = tile_buffers[cur] + (int64_t)local_row * hidden_size;
      const int32_t dst_row = w13_dest_index(row, intermediate_size, activation_id);
      for (int32_t batch = 0; batch < batch_size; ++batch) {
        float value = dot_hf_hf_to_float(token_ptrs[batch], weight_row, hidden_size);
        if (apply_router_weight_on_input) {
          value *= route_weights[batch];
        }
        if (has_bias) {
          value += (float)w13_bias[row];
        }
        gate_up_batch[(int64_t)batch * w13_dim + dst_row] = (float16)value;
      }
    }
  }
  return JIT_DEV_STATUS_SUCCESS;
}

inline uint32_t accumulate_w2_batch_tiled(const float16* hidden_batch,
                                          const int32_t* route_indices,
                                          const float* route_weights,
                                          int32_t batch_size,
                                          const float16* w2,
                                          const float16* w2_bias,
                                          float16* route_out,
                                          float16* weight_tile,
                                          uint32_t weight_tile_bytes,
                                          uint32_t thread_id,
                                          int32_t hidden_size,
                                          int32_t intermediate_size,
                                          bool has_bias,
                                          bool apply_router_weight_on_input) {
  const uint32_t row_bytes = (uint32_t)intermediate_size * sizeof(float16);
  const int32_t tile_rows = weight_tile_bytes >= row_bytes
                                ? (int32_t)(weight_tile_bytes / row_bytes)
                                : 0;

  if (tile_rows <= 0 || weight_tile == nullptr) {
    for (int32_t h = 0; h < hidden_size; ++h) {
      const float16* weight_row = w2 + (int64_t)h * intermediate_size;
      for (int32_t batch = 0; batch < batch_size; ++batch) {
        const float16* hidden = hidden_batch + (int64_t)batch * intermediate_size;
        float acc = dot_hf_hf_to_float(hidden, weight_row, intermediate_size);
        if (has_bias) {
          acc += (float)w2_bias[h];
        }
        if (!apply_router_weight_on_input) {
          acc *= route_weights[batch];
        }
        route_out[(int64_t)route_indices[batch] * hidden_size + h] = (float16)acc;
      }
    }
    return JIT_DEV_STATUS_SUCCESS;
  }

  float16* tile_buffers[2] = {weight_tile, weight_tile + (int64_t)weight_tile_bytes / sizeof(float16)};
  const int32_t num_tiles = (hidden_size + tile_rows - 1) / tile_rows;
  QShimUDmaHandle handles[2] = {INVALID_UDMA_HANDLE, INVALID_UDMA_HANDLE};
  uint32_t submit_status[2] = {JIT_DEV_STATUS_SUCCESS, JIT_DEV_STATUS_SUCCESS};

  int32_t first_rows = tile_rows;
  if (first_rows > hidden_size) {
    first_rows = hidden_size;
  }
  {
    AicJitUdmaDescCommonAttrs attrs = {};
    attrs.order = 0;
    handles[0] = qshimLinearUdmaSubmit(thread_id,
                                       (AicJitPtr)w2,
                                       (uint32_t)first_rows * row_bytes,
                                       (AicJitPtr)tile_buffers[0],
                                       &attrs,
                                       true,
                                       &submit_status[0]);
  }

  for (int32_t tile = 0; tile < num_tiles; ++tile) {
    const int32_t cur = tile & 1;
    const int32_t next = cur ^ 1;
    const int32_t row_start = tile * tile_rows;
    int32_t rows = tile_rows;
    if (rows > hidden_size - row_start) {
      rows = hidden_size - row_start;
    }

    uint32_t status = wait_dma_handle(handles[cur], submit_status[cur]);
    if (status != JIT_DEV_STATUS_SUCCESS) {
      return status;
    }

    const int32_t next_tile = tile + 1;
    if (next_tile < num_tiles) {
      const int32_t next_row_start = next_tile * tile_rows;
      int32_t next_rows = tile_rows;
      if (next_rows > hidden_size - next_row_start) {
        next_rows = hidden_size - next_row_start;
      }
      const float16* next_src = w2 + (int64_t)next_row_start * intermediate_size;
      {
        AicJitUdmaDescCommonAttrs attrs = {};
        attrs.order = 0;
        handles[next] = qshimLinearUdmaSubmit(thread_id,
                                             (AicJitPtr)next_src,
                                             (uint32_t)next_rows * row_bytes,
                                             (AicJitPtr)tile_buffers[next],
                                             &attrs,
                                             true,
                                             &submit_status[next]);
      }
    }

    for (int32_t local_row = 0; local_row < rows; ++local_row) {
      const int32_t h = row_start + local_row;
      const float16* weight_row = tile_buffers[cur] + (int64_t)local_row * intermediate_size;
      for (int32_t batch = 0; batch < batch_size; ++batch) {
        const float16* hidden = hidden_batch + (int64_t)batch * intermediate_size;
        float acc = dot_hf_hf_to_float(hidden, weight_row, intermediate_size);
        if (has_bias) {
          acc += (float)w2_bias[h];
        }
        if (!apply_router_weight_on_input) {
          acc *= route_weights[batch];
        }
        route_out[(int64_t)route_indices[batch] * hidden_size + h] = (float16)acc;
      }
    }
  }
  return JIT_DEV_STATUS_SUCCESS;
}

inline uint32_t route_group_count_kernel_main(const AicJitEntryPointConfig* cfg,
                                              const AicJitPointerArray* ptrs) {
  const float16* topk_ids = (const float16*)ptrs->pointers[0];
  const float* expert_map = (const float*)ptrs->pointers[1];
  float* worker_counts = (float*)ptrs->pointers[2];
  const float* params = (const float*)ptrs->pointers[3];

  const int32_t num_tokens = (int32_t)params[0];
  const int32_t num_experts = (int32_t)params[4];
  const int32_t topk = (int32_t)params[5];
  const int32_t global_num_experts = (int32_t)params[9];
  const bool has_expert_map = ((int32_t)params[10]) != 0;
  if (num_tokens < 0 || num_experts <= 0 || topk <= 0 || global_num_experts <= 0) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  const int32_t workers = (int32_t)(cfg->numCores * cfg->numThreads);
  const int32_t worker_id = (int32_t)(cfg->coreID * cfg->numThreads +
                                      (cfg->threadID % cfg->numThreads));
  float* counts = worker_counts + (int64_t)worker_id * num_experts;
  for (int32_t expert = 0; expert < num_experts; ++expert) {
    counts[expert] = 0.0F;
  }

  const int32_t total_routes = num_tokens * topk;
  for (int32_t route_idx = worker_id; route_idx < total_routes; route_idx += workers) {
    const int32_t global_expert = (int32_t)topk_ids[route_idx];
    const int32_t local_expert = map_global_to_local_expert(
        global_expert, expert_map, num_experts, global_num_experts, has_expert_map);
    if (local_expert >= 0) {
      counts[local_expert] = counts[local_expert] + 1.0F;
    }
  }

  return JIT_DEV_STATUS_SUCCESS;
}

inline uint32_t route_group_prefix_kernel_main(const AicJitEntryPointConfig* cfg,
                                               const AicJitPointerArray* ptrs) {
  const float* worker_counts = (const float*)ptrs->pointers[0];
  float* worker_offsets = (float*)ptrs->pointers[1];
  float* expert_offsets = (float*)ptrs->pointers[2];
  const float* params = (const float*)ptrs->pointers[3];

  const int32_t num_tokens = (int32_t)params[0];
  const int32_t num_experts = (int32_t)params[4];
  const int32_t topk = (int32_t)params[5];
  if (num_tokens < 0 || num_experts <= 0 || topk <= 0) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  const int32_t worker_id = (int32_t)(cfg->coreID * cfg->numThreads +
                                      (cfg->threadID % cfg->numThreads));
  if (worker_id != 0) {
    return JIT_DEV_STATUS_SUCCESS;
  }

  const int32_t workers = (int32_t)(cfg->numCores * cfg->numThreads);
  float running = 0.0F;
  for (int32_t expert = 0; expert < num_experts; ++expert) {
    expert_offsets[expert] = running;
    for (int32_t worker = 0; worker < workers; ++worker) {
      const int64_t idx = (int64_t)worker * num_experts + expert;
      worker_offsets[idx] = running;
      running += worker_counts[idx];
    }
  }
  expert_offsets[num_experts] = running;

  return JIT_DEV_STATUS_SUCCESS;
}

inline uint32_t route_group_fill_kernel_main(const AicJitEntryPointConfig* cfg,
                                             const AicJitPointerArray* ptrs) {
  const float16* topk_ids = (const float16*)ptrs->pointers[0];
  const float* expert_map = (const float*)ptrs->pointers[1];
  const float* worker_offsets = (const float*)ptrs->pointers[2];
  float* expert_route_indices = (float*)ptrs->pointers[3];
  const float* params = (const float*)ptrs->pointers[4];

  const int32_t num_tokens = (int32_t)params[0];
  const int32_t num_experts = (int32_t)params[4];
  const int32_t topk = (int32_t)params[5];
  const int32_t global_num_experts = (int32_t)params[9];
  const bool has_expert_map = ((int32_t)params[10]) != 0;
  if (num_tokens < 0 || num_experts <= 0 || topk <= 0 || global_num_experts <= 0) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  const uint32_t local_thread_id = cfg->threadID % cfg->numThreads;
  constexpr uint32_t kAlign = 128U;
  const uint32_t local_counts_bytes = align_up_u32(num_experts * sizeof(float), kAlign);
  const uint32_t required_vtcm_bytes = local_counts_bytes * cfg->numThreads + kAlign;

  int64_t vtcm_size = 0;
  uint32_t status = qshimQuery(DEV_ATTR_QSHIM_VTCM_SIZE, &vtcm_size);
  if (status != JIT_DEV_STATUS_SUCCESS) {
    return status;
  }
  if ((uint64_t)required_vtcm_bytes > (uint64_t)vtcm_size) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  float* local_counts = (float*)(align_up_ptr(qshimGetBaseVtcmAddr(), kAlign) +
                                 local_thread_id * local_counts_bytes);
  for (int32_t expert = 0; expert < num_experts; ++expert) {
    local_counts[expert] = 0.0F;
  }

  const int32_t workers = (int32_t)(cfg->numCores * cfg->numThreads);
  const int32_t worker_id = (int32_t)(cfg->coreID * cfg->numThreads + local_thread_id);
  const int32_t total_routes = num_tokens * topk;
  const float* offsets = worker_offsets + (int64_t)worker_id * num_experts;

  for (int32_t route_idx = worker_id; route_idx < total_routes; route_idx += workers) {
    const int32_t global_expert = (int32_t)topk_ids[route_idx];
    const int32_t local_expert = map_global_to_local_expert(
        global_expert, expert_map, num_experts, global_num_experts, has_expert_map);
    if (local_expert >= 0) {
      const int32_t out_idx = (int32_t)(offsets[local_expert] + local_counts[local_expert]);
      expert_route_indices[out_idx] = (float)route_idx;
      local_counts[local_expert] = local_counts[local_expert] + 1.0F;
    }
  }

  return JIT_DEV_STATUS_SUCCESS;
}


inline uint32_t route_compute_kernel_main(const AicJitEntryPointConfig* cfg,
                                          const AicJitPointerArray* ptrs) {
  const float16* x = (const float16*)ptrs->pointers[0];
  const float16* topk_weights = (const float16*)ptrs->pointers[1];
  const float16* topk_ids = (const float16*)ptrs->pointers[2];
  const float16* w13_weight = (const float16*)ptrs->pointers[3];
  const float16* w2_weight = (const float16*)ptrs->pointers[4];
  const float16* bias = (const float16*)ptrs->pointers[5];
  float16* route_out = (float16*)ptrs->pointers[6];
  const float* expert_route_indices = (const float*)ptrs->pointers[7];
  const float* expert_offsets = (const float*)ptrs->pointers[8];
  const float* expert_map = (const float*)ptrs->pointers[9];
  const float* params = (const float*)ptrs->pointers[10];

  const int32_t num_tokens = (int32_t)params[0];
  const int32_t hidden_size = (int32_t)params[1];
  const int32_t w13_dim = (int32_t)params[2];
  const int32_t intermediate_size = (int32_t)params[3];
  const int32_t num_experts = (int32_t)params[4];
  const int32_t topk = (int32_t)params[5];
  const int32_t activation_id = (int32_t)params[6];
  const bool has_bias = ((int32_t)params[7]) != 0;
  const bool apply_router_weight_on_input = ((int32_t)params[8]) != 0;
  const int32_t global_num_experts = (int32_t)params[9];
  const bool has_expert_map = ((int32_t)params[10]) != 0;

  if (num_tokens < 0 || hidden_size <= 0 || intermediate_size <= 0 ||
      num_experts <= 0 || topk <= 0 || global_num_experts <= 0 ||
      !valid_activation(activation_id)) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  const int32_t expected_w13_dim = activation_is_no_mul(activation_id)
                                       ? intermediate_size
                                       : 2 * intermediate_size;
  if (w13_dim != expected_w13_dim) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  const uint32_t thread_id = cfg->threadID;
  const uint32_t local_thread_id = thread_id % cfg->numThreads;
  constexpr uint32_t kAlign = 128U;
  constexpr uint32_t kTargetWeightTileBytes = 64U * 1024U;
  constexpr int32_t kMaxBatchRoutes = 4;

  const uint32_t w13_row_bytes = align_up_u32(hidden_size * sizeof(float16), kAlign);
  const uint32_t w2_row_bytes = align_up_u32(intermediate_size * sizeof(float16), kAlign);
  const uint32_t max_weight_row_bytes = w13_row_bytes > w2_row_bytes ? w13_row_bytes : w2_row_bytes;
  const uint32_t total_routes = (uint32_t)(num_tokens * topk);

  int64_t vtcm_size = 0;
  uint32_t status = qshimQuery(DEV_ATTR_QSHIM_VTCM_SIZE, &vtcm_size);
  if (status != JIT_DEV_STATUS_SUCCESS) {
    return status;
  }

  const uint32_t usable_vtcm = vtcm_size > (int64_t)kAlign ? (uint32_t)vtcm_size - kAlign : 0U;
  const uint32_t max_scratch_per_thread = usable_vtcm / cfg->numThreads;
  uint32_t batch_size_u32 = 0;
  uint32_t weight_tile_bytes = 0;
  uint32_t gate_up_batch_bytes = 0;
  uint32_t hidden_batch_bytes = 0;

  for (int32_t candidate_batch = kMaxBatchRoutes; candidate_batch >= 1; --candidate_batch) {
    const uint32_t candidate_gate_up_bytes =
        align_up_u32((uint32_t)candidate_batch * w13_dim * sizeof(float16), kAlign);
    const uint32_t candidate_hidden_bytes =
        align_up_u32((uint32_t)candidate_batch * intermediate_size * sizeof(float16), kAlign);
    const uint32_t batch_bytes = candidate_gate_up_bytes + candidate_hidden_bytes;
    if (batch_bytes >= max_scratch_per_thread) {
      continue;
    }
    const uint32_t available_for_tiles = max_scratch_per_thread - batch_bytes;
    uint32_t candidate_tile_bytes = 0;
    if (max_weight_row_bytes > 0) {
      candidate_tile_bytes = align_down_u32(
          ((kTargetWeightTileBytes < available_for_tiles / 2U) ? kTargetWeightTileBytes : (available_for_tiles / 2U)), max_weight_row_bytes);
    }
    if (candidate_tile_bytes >= max_weight_row_bytes || candidate_batch == 1) {
      batch_size_u32 = (uint32_t)candidate_batch;
      weight_tile_bytes = candidate_tile_bytes;
      gate_up_batch_bytes = candidate_gate_up_bytes;
      hidden_batch_bytes = candidate_hidden_bytes;
      break;
    }
  }

  if (batch_size_u32 == 0) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  const uint32_t scratch_bytes_per_thread =
      gate_up_batch_bytes + hidden_batch_bytes + 2U * weight_tile_bytes;
  const uint32_t required_vtcm_bytes = scratch_bytes_per_thread * cfg->numThreads + kAlign;
  if ((uint64_t)required_vtcm_bytes > (uint64_t)vtcm_size) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  uint8_t* vtcm_ptr = align_up_ptr(qshimGetBaseVtcmAddr(), kAlign) +
                      local_thread_id * scratch_bytes_per_thread;
  float16* gate_up_batch = (float16*)vtcm_ptr;
  vtcm_ptr += gate_up_batch_bytes;
  float16* hidden_batch = (float16*)vtcm_ptr;
  vtcm_ptr += hidden_batch_bytes;
  float16* weight_tile = weight_tile_bytes > 0 ? (float16*)vtcm_ptr : nullptr;

  const int32_t workers = (int32_t)(cfg->numCores * cfg->numThreads);
  const int32_t worker_id = (int32_t)(cfg->coreID * cfg->numThreads + local_thread_id);
  const int32_t batch_size = (int32_t)batch_size_u32;

  for (int32_t route_idx = worker_id; route_idx < (int32_t)total_routes; route_idx += workers) {
    const int32_t global_expert = (int32_t)topk_ids[route_idx];
    const int32_t local_expert = map_global_to_local_expert(
        global_expert, expert_map, num_experts, global_num_experts, has_expert_map);
    if (local_expert < 0) {
      zero_route_out(route_out + (int64_t)route_idx * hidden_size, hidden_size);
    }
  }

  if (workers >= num_experts) {
    const int32_t base_workers_per_expert = workers / num_experts;
    const int32_t extra_workers = workers - base_workers_per_expert * num_experts;

    for (int32_t expert = 0; expert < num_experts; ++expert) {
      const int32_t expert_workers = base_workers_per_expert + (expert < extra_workers ? 1 : 0);
      const int32_t expert_worker_start =
          expert * base_workers_per_expert + (expert < extra_workers ? expert : extra_workers);
      if (worker_id < expert_worker_start ||
          worker_id >= expert_worker_start + expert_workers) {
        continue;
      }

      const int32_t route_shard = worker_id - expert_worker_start;
      const int32_t route_begin = (int32_t)expert_offsets[expert];
      const int32_t route_count = (int32_t)(expert_offsets[expert + 1] - expert_offsets[expert]);
      const float* expert_indices = expert_route_indices + route_begin;
      const float16* expert_w13 = w13_weight + ((int64_t)expert * w13_dim * hidden_size);
      const float16* expert_w2 =
          w2_weight + ((int64_t)expert * hidden_size * intermediate_size);
      const float16* expert_w13_bias = has_bias ? bias + ((int64_t)expert * w13_dim) : nullptr;
      const float16* expert_w2_bias = has_bias ? bias + ((int64_t)num_experts * w13_dim) +
                                                     ((int64_t)expert * hidden_size)
                                               : nullptr;

      for (int32_t route_offset = route_shard * batch_size;
           route_offset < route_count;
           route_offset += expert_workers * batch_size) {
        int32_t actual_batch = batch_size;
        if (actual_batch > route_count - route_offset) {
          actual_batch = route_count - route_offset;
        }

        const float16* token_ptrs[kMaxBatchRoutes];
        float route_weights[kMaxBatchRoutes];
        int32_t route_indices[kMaxBatchRoutes];
        for (int32_t batch = 0; batch < actual_batch; ++batch) {
          const int32_t route_idx = (int32_t)expert_indices[route_offset + batch];
          route_indices[batch] = route_idx;
          route_weights[batch] = (float)topk_weights[route_idx];
          const int32_t token = route_idx / topk;
          token_ptrs[batch] = x + (int64_t)token * hidden_size;
        }

        status = compute_gate_up_batch_tiled(token_ptrs,
                                             route_weights,
                                             actual_batch,
                                             expert_w13,
                                             expert_w13_bias,
                                             gate_up_batch,
                                             weight_tile,
                                             weight_tile_bytes,
                                             thread_id,
                                             hidden_size,
                                             w13_dim,
                                             intermediate_size,
                                             activation_id,
                                             has_bias,
                                             apply_router_weight_on_input);
        if (status != JIT_DEV_STATUS_SUCCESS) {
          return status;
        }
        for (int32_t batch = 0; batch < actual_batch; ++batch) {
          apply_activation_vec(gate_up_batch + (int64_t)batch * w13_dim,
                               hidden_batch + (int64_t)batch * intermediate_size,
                               intermediate_size,
                               activation_id);
        }
        status = accumulate_w2_batch_tiled(hidden_batch,
                                           route_indices,
                                           route_weights,
                                           actual_batch,
                                           expert_w2,
                                           expert_w2_bias,
                                           route_out,
                                           weight_tile,
                                           weight_tile_bytes,
                                           thread_id,
                                           hidden_size,
                                           intermediate_size,
                                           has_bias,
                                           apply_router_weight_on_input);
        if (status != JIT_DEV_STATUS_SUCCESS) {
          return status;
        }
      }
    }
  } else {
    for (int32_t expert = worker_id; expert < num_experts; expert += workers) {
      const int32_t route_begin = (int32_t)expert_offsets[expert];
      const int32_t route_count = (int32_t)(expert_offsets[expert + 1] - expert_offsets[expert]);
      const float* expert_indices = expert_route_indices + route_begin;
      const float16* expert_w13 = w13_weight + ((int64_t)expert * w13_dim * hidden_size);
      const float16* expert_w2 =
          w2_weight + ((int64_t)expert * hidden_size * intermediate_size);
      const float16* expert_w13_bias = has_bias ? bias + ((int64_t)expert * w13_dim) : nullptr;
      const float16* expert_w2_bias = has_bias ? bias + ((int64_t)num_experts * w13_dim) +
                                                     ((int64_t)expert * hidden_size)
                                               : nullptr;

      for (int32_t route_offset = 0; route_offset < route_count; route_offset += batch_size) {
        int32_t actual_batch = batch_size;
        if (actual_batch > route_count - route_offset) {
          actual_batch = route_count - route_offset;
        }

        const float16* token_ptrs[kMaxBatchRoutes];
        float route_weights[kMaxBatchRoutes];
        int32_t route_indices[kMaxBatchRoutes];
        for (int32_t batch = 0; batch < actual_batch; ++batch) {
          const int32_t route_idx = (int32_t)expert_indices[route_offset + batch];
          route_indices[batch] = route_idx;
          route_weights[batch] = (float)topk_weights[route_idx];
          const int32_t token = route_idx / topk;
          token_ptrs[batch] = x + (int64_t)token * hidden_size;
        }

        status = compute_gate_up_batch_tiled(token_ptrs,
                                             route_weights,
                                             actual_batch,
                                             expert_w13,
                                             expert_w13_bias,
                                             gate_up_batch,
                                             weight_tile,
                                             weight_tile_bytes,
                                             thread_id,
                                             hidden_size,
                                             w13_dim,
                                             intermediate_size,
                                             activation_id,
                                             has_bias,
                                             apply_router_weight_on_input);
        if (status != JIT_DEV_STATUS_SUCCESS) {
          return status;
        }
        for (int32_t batch = 0; batch < actual_batch; ++batch) {
          apply_activation_vec(gate_up_batch + (int64_t)batch * w13_dim,
                               hidden_batch + (int64_t)batch * intermediate_size,
                               intermediate_size,
                               activation_id);
        }
        status = accumulate_w2_batch_tiled(hidden_batch,
                                           route_indices,
                                           route_weights,
                                           actual_batch,
                                           expert_w2,
                                           expert_w2_bias,
                                           route_out,
                                           weight_tile,
                                           weight_tile_bytes,
                                           thread_id,
                                           hidden_size,
                                           intermediate_size,
                                           has_bias,
                                           apply_router_weight_on_input);
        if (status != JIT_DEV_STATUS_SUCCESS) {
          return status;
        }
      }
    }
  }

  return JIT_DEV_STATUS_SUCCESS;
}

inline uint32_t reduce_kernel_main(const AicJitEntryPointConfig* cfg,
                                   const AicJitPointerArray* ptrs) {
  const float16* route_out = (const float16*)ptrs->pointers[0];
  float16* out = (float16*)ptrs->pointers[1];
  const float* params = (const float*)ptrs->pointers[2];

  const int32_t num_tokens = (int32_t)params[0];
  const int32_t hidden_size = (int32_t)params[1];
  const int32_t topk = (int32_t)params[5];
  if (num_tokens < 0 || hidden_size <= 0 || topk <= 0) {
    return JIT_DEV_ERROR_INVALID_PARAMETER;
  }

  constexpr int32_t kElemsPerHalfVector = sizeof(HVX_Vector) / sizeof(float16);
  const int32_t workers = (int32_t)(cfg->numCores * cfg->numThreads);
  const int32_t worker_id =
      (int32_t)(cfg->coreID * cfg->numThreads + (cfg->threadID % cfg->numThreads));
  const int32_t blocks_per_token =
      (hidden_size + kElemsPerHalfVector - 1) / kElemsPerHalfVector;
  const int32_t total_blocks = num_tokens * blocks_per_token;

  for (int32_t block_idx = worker_id; block_idx < total_blocks; block_idx += workers) {
    const int32_t token = block_idx / blocks_per_token;
    const int32_t block = block_idx - token * blocks_per_token;
    const int32_t h = block * kElemsPerHalfVector;
    float16* out_row = out + (int64_t)token * hidden_size;
    const int32_t elems = hidden_size - h;

    if (elems >= kElemsPerHalfVector) {
      HVX_Vector acc_lo = Q6_V_vzero();
      HVX_Vector acc_hi = Q6_V_vzero();
      for (int32_t route = 0; route < topk; ++route) {
        const float16* route_row = route_out + ((int64_t)token * topk + route) * hidden_size;
        HVX_Vector route_vhf = LoadUnaligned<HVX_Vector>((const int8_t*)(route_row + h));
        HVX_VectorPair route_pair = Q6_Wsf_vcvt_Vhf(route_vhf);
        acc_lo = Q6_Vsf_vadd_VsfVsf(acc_lo, Q6_V_lo_W(route_pair));
        acc_hi = Q6_Vsf_vadd_VsfVsf(acc_hi, Q6_V_hi_W(route_pair));
      }
      HVX_Vector out_vhf = Q6_Vhf_vcvt_VsfVsf(acc_lo, acc_hi);
      StoreUnalignedHVX((int8_t*)(out_row + h), out_vhf);
    } else {
      for (int32_t offset = 0; offset < elems; ++offset) {
        float acc = 0.0F;
        for (int32_t route = 0; route < topk; ++route) {
          const float16* route_row = route_out + ((int64_t)token * topk + route) * hidden_size;
          acc += (float)route_row[h + offset];
        }
        out_row[h + offset] = (float16)acc;
      }
    }
  }

  return JIT_DEV_STATUS_SUCCESS;
}

}  // namespace unquantized_fused_moe_route_reduce

QAIC_KERNEL_API int32_t multinsp_multithreaded_unquantized_fused_moe_route_group_count(
    const AicJitEntryPointConfig* entryConfig,
    const AicJitPointerArray* pointerArray) {
  return unquantized_fused_moe_route_reduce::route_group_count_kernel_main(
      entryConfig, pointerArray);
}

QAIC_KERNEL_API int32_t multinsp_multithreaded_unquantized_fused_moe_route_group_prefix(
    const AicJitEntryPointConfig* entryConfig,
    const AicJitPointerArray* pointerArray) {
  return unquantized_fused_moe_route_reduce::route_group_prefix_kernel_main(
      entryConfig, pointerArray);
}

QAIC_KERNEL_API int32_t multinsp_multithreaded_unquantized_fused_moe_route_group_fill(
    const AicJitEntryPointConfig* entryConfig,
    const AicJitPointerArray* pointerArray) {
  return unquantized_fused_moe_route_reduce::route_group_fill_kernel_main(
      entryConfig, pointerArray);
}

QAIC_KERNEL_API int32_t multinsp_multithreaded_unquantized_fused_moe_route_compute(
    const AicJitEntryPointConfig* entryConfig,
    const AicJitPointerArray* pointerArray) {
  return unquantized_fused_moe_route_reduce::route_compute_kernel_main(
      entryConfig, pointerArray);
}

QAIC_KERNEL_API int32_t multinsp_multithreaded_unquantized_fused_moe_route_reduce(
    const AicJitEntryPointConfig* entryConfig,
    const AicJitPointerArray* pointerArray) {
  return unquantized_fused_moe_route_reduce::reduce_kernel_main(entryConfig, pointerArray);
}
