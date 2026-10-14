// ---------------------------------------------------------------------------------------
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
// ---------------------------------------------------------------------------------------

#include <hexagon_types.h>
#include <hexagon_protos.h>
#include "QAicHexagonMath.h"
#include "QAicHexagonReducer.h"
#include "QAicHexagonUtils.h"
#include "QAicHexagonPlatformIntf.h"
#include "jit_dev_exe_function.h"
#include "jit_dev_status_codes.h"
#include "jit_qshim_api.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

static inline float sigmoidf(float x) {
  return 1.0f / (1.0f + __builtin_expf(-x));
}
static inline float siluf(float x) { return x * sigmoidf(x); }
static inline float softplusf(float x) {
  if (x > 20.0f) return x;
  return __builtin_logf(1.0f + __builtin_expf(x));
}

QAIC_KERNEL_API uint32_t gdn_gating(const AicJitEntryPointConfig* cfg,
                                    const AicJitPointerArray* ptrs) {
  if (ptrs->numPointers < 8) return JIT_DEV_ERROR_INVALID_PARAMETER;
  const __fp16* a = (const __fp16*)*ptrs->pointers;
  const __fp16* b = (const __fp16*)ptrs->pointers[1];
  const float* A_log = (const float*)ptrs->pointers[2];
  const float* dt_bias = (const float*)ptrs->pointers[3];
  float* g_out = (float*)ptrs->pointers[4];
  float* beta_out = (float*)ptrs->pointers[5];
  const int T = *(const int32_t*)ptrs->pointers[6];
  const int HV = *(const int32_t*)ptrs->pointers[7];

  const uint32_t localTid = cfg->threadID % cfg->numThreads;
  const uint32_t globalTid = cfg->coreID * cfg->numThreads + localTid;
  const uint32_t globalThreads = cfg->numCores * cfg->numThreads;

  const int total = T * HV;
  for (int idx = (int)globalTid; idx < total; idx += (int)globalThreads) {
    const int h = idx % HV;
    const float av = (float)a[idx];
    const float bv = (float)b[idx];
    const float sp = softplusf(av + dt_bias[h]);
    g_out[idx] = -__builtin_expf(A_log[h]) * sp;
    beta_out[idx] = sigmoidf(bv);
  }
  return JIT_DEV_STATUS_SUCCESS;
}

QAIC_KERNEL_API uint32_t gdn_conv1d_update(const AicJitEntryPointConfig* cfg,
                                           const AicJitPointerArray* ptrs) {
  if (ptrs->numPointers < 11) return JIT_DEV_ERROR_INVALID_PARAMETER;
  const __fp16* x = (const __fp16*)*ptrs->pointers;
  __fp16* cstate = (__fp16*)ptrs->pointers[1];
  const __fp16* weight = (const __fp16*)ptrs->pointers[2];
  const __fp16* bias = (const __fp16*)ptrs->pointers[3];
  __fp16* out = (__fp16*)ptrs->pointers[4];
  const int32_t* slot_ids = (const int32_t*)ptrs->pointers[5];
  const int ND = *(const int32_t*)ptrs->pointers[6];
  const int conv_dim = *(const int32_t*)ptrs->pointers[7];
  const int K_CONV = *(const int32_t*)ptrs->pointers[8];
  const int bias_present = *(const int32_t*)ptrs->pointers[9];
  const int silu = *(const int32_t*)ptrs->pointers[10];
  const int state_len = K_CONV - 1;

  const uint32_t localTid = cfg->threadID % cfg->numThreads;
  const uint32_t globalTid = cfg->coreID * cfg->numThreads + localTid;
  const uint32_t globalThreads = cfg->numCores * cfg->numThreads;

  const int total = ND * conv_dim;
  for (int idx = (int)globalTid; idx < total; idx += (int)globalThreads) {
    const int s = idx / conv_dim;
    const int c = idx % conv_dim;
    const int slot = slot_ids[s];
    __fp16* st = cstate + ((size_t)slot * conv_dim + c) * state_len;
    const __fp16* w = weight + (size_t)c * K_CONV;
    const float xc = (float)x[(size_t)s * conv_dim + c];

    float acc = bias_present ? (float)bias[c] : 0.0f;
    for (int j = 0; j < state_len; ++j) acc += (float)st[j] * (float)w[j];
    acc += xc * (float)w[state_len];

    for (int j = 0; j < state_len - 1; ++j) st[j] = st[j + 1];
    if (state_len > 0) st[state_len - 1] = (__fp16)xc;

    if (silu) acc = siluf(acc);
    out[(size_t)s * conv_dim + c] = (__fp16)acc;
  }
  return JIT_DEV_STATUS_SUCCESS;
}

QAIC_KERNEL_API uint32_t gdn_conv1d_prefill(const AicJitEntryPointConfig* cfg,
                                            const AicJitPointerArray* ptrs) {
  if (ptrs->numPointers < 14) return JIT_DEV_ERROR_INVALID_PARAMETER;
  const __fp16* x = (const __fp16*)*ptrs->pointers;
  const __fp16* weight = (const __fp16*)ptrs->pointers[1];
  const __fp16* bias = (const __fp16*)ptrs->pointers[2];
  __fp16* cstates = (__fp16*)ptrs->pointers[3];
  const int32_t* qstart = (const int32_t*)ptrs->pointers[4];
  const int32_t* cache_idx = (const int32_t*)ptrs->pointers[5];
  const int32_t* has_init = (const int32_t*)ptrs->pointers[6];
  __fp16* out = (__fp16*)ptrs->pointers[7];
  const int conv_dim = *(const int32_t*)ptrs->pointers[8];
  const int Ttot = *(const int32_t*)ptrs->pointers[9];
  const int num_seqs = *(const int32_t*)ptrs->pointers[10];
  const int K_CONV = *(const int32_t*)ptrs->pointers[11];
  const int bias_present = *(const int32_t*)ptrs->pointers[12];
  const int silu = *(const int32_t*)ptrs->pointers[13];
  const int state_len = K_CONV - 1;

  const uint32_t localTid = cfg->threadID % cfg->numThreads;
  const uint32_t globalTid = cfg->coreID * cfg->numThreads + localTid;
  const uint32_t globalThreads = cfg->numCores * cfg->numThreads;

  for (int c = (int)globalTid; c < conv_dim; c += (int)globalThreads) {
    const __fp16* w = weight + (size_t)c * K_CONV;
    const float bc = bias_present ? (float)bias[c] : 0.0f;

    for (int s = 0; s < num_seqs; ++s) {
      const int bos = qstart[s];
      const int eos = qstart[s + 1];
      const int slot = cache_idx[s];
      __fp16* st = cstates + ((size_t)slot * conv_dim + c) * state_len;

      float win[8];
      for (int j = 0; j < state_len; ++j)
        win[j] = has_init[s] ? (float)st[j] : 0.0f;

      for (int t = bos; t < eos; ++t) {
        const float xt = (float)x[(size_t)c * Ttot + t];
        win[state_len] = xt;
        float acc = bc;
        for (int j = 0; j < K_CONV; ++j) acc += win[j] * (float)w[j];
        if (silu) acc = siluf(acc);
        out[(size_t)c * Ttot + t] = (__fp16)acc;
        for (int j = 0; j < state_len; ++j) win[j] = win[j + 1];
      }
      for (int j = 0; j < state_len; ++j) st[j] = (__fp16)win[j];
    }
  }
  return JIT_DEV_STATUS_SUCCESS;
}

static inline float hvx_hsum_sf(HVX_Vector v_lo, HVX_Vector v_hi) {
  HVX_Vector s = Q6_Vsf_vadd_VsfVsf(v_lo, v_hi);
  s = Q6_Vsf_vadd_VsfVsf(s, Q6_V_vror_VR(s, 4));
  s = Q6_Vsf_vadd_VsfVsf(s, Q6_V_vror_VR(s, 8));
  s = Q6_Vsf_vadd_VsfVsf(s, Q6_V_vror_VR(s, 16));
  s = Q6_Vsf_vadd_VsfVsf(s, Q6_V_vror_VR(s, 32));
  float r;
  __builtin_memcpy(&r, &s, sizeof(float));
  return r;
}

QAIC_KERNEL_API uint32_t gdn_recurrent_decode(const AicJitEntryPointConfig* cfg,
                                              const AicJitPointerArray* ptrs) {
  if (ptrs->numPointers < 15) return JIT_DEV_ERROR_INVALID_PARAMETER;
  const __fp16* q = (const __fp16*)*ptrs->pointers;
  const __fp16* k = (const __fp16*)ptrs->pointers[1];
  const __fp16* v = (const __fp16*)ptrs->pointers[2];
  const float* g = (const float*)ptrs->pointers[3];
  const float* beta = (const float*)ptrs->pointers[4];
  __fp16* o = (__fp16*)ptrs->pointers[5];
  __fp16* ssm = (__fp16*)ptrs->pointers[6];
  const int32_t* slot_ids = (const int32_t*)ptrs->pointers[7];
  const int ND = *(const int32_t*)ptrs->pointers[8];
  const int H = *(const int32_t*)ptrs->pointers[9];
  const int HV = *(const int32_t*)ptrs->pointers[10];
  const int Kd = *(const int32_t*)ptrs->pointers[11];
  const int Vd = *(const int32_t*)ptrs->pointers[12];
  const float scale = *(const float*)ptrs->pointers[13];
  const float l2eps = *(const float*)ptrs->pointers[14];
  const int heads_per_group = HV / H;

  const uint32_t localTid = cfg->threadID % cfg->numThreads;
  const uint32_t globalTid = cfg->coreID * cfg->numThreads + localTid;
  const uint32_t globalThreads = cfg->numCores * cfg->numThreads;

  float qn[256] __attribute__((aligned(128)));
  float kn[256] __attribute__((aligned(128)));
  float delta[256] __attribute__((aligned(128)));
  __fp16 qn16[256] __attribute__((aligned(128)));
  __fp16 kn16[256] __attribute__((aligned(128)));

  constexpr int EPV = (int)(sizeof(HVX_Vector) / sizeof(__fp16));
  const int vlen = Kd / EPV;

  const int total = ND * HV;
  for (int idx = (int)globalTid; idx < total; idx += (int)globalThreads) {
    const int s = idx / HV;
    const int h = idx % HV;
    const int hk = h / heads_per_group;
    const int slot = slot_ids[s];

    const __fp16* qh = q + ((size_t)s * H + hk) * Kd;
    const __fp16* kh = k + ((size_t)s * H + hk) * Kd;
    const __fp16* vh = v + ((size_t)s * HV + h) * Vd;
    __fp16* oh = o + ((size_t)s * HV + h) * Vd;
    __fp16* S = ssm + ((size_t)slot * HV + h) * Vd * Kd;
    const float gv = g[idx];
    const float bv = beta[idx];
    const float g_exp = __builtin_expf(gv);

    float qss = 0.0f, kss = 0.0f;
    for (int j = 0; j < Kd; ++j) {
      const float qq = (float)qh[j];
      const float kk = (float)kh[j];
      qn[j] = qq;
      kn[j] = kk;
      qss += qq * qq;
      kss += kk * kk;
    }
    const float qinv = 1.0f / __builtin_sqrtf(qss + l2eps);
    const float kinv = 1.0f / __builtin_sqrtf(kss + l2eps);
    for (int j = 0; j < Kd; ++j) {
      qn[j] = qn[j] * qinv * scale;
      kn[j] = kn[j] * kinv;
      qn16[j] = (__fp16)qn[j];
      kn16[j] = (__fp16)kn[j];
    }

    {
      __fp16 g16 = (__fp16)g_exp;
      uint32_t gb;
      __builtin_memcpy(&gb, &g16, sizeof(__fp16));
      gb = (gb & 0xFFFFu) | (gb << 16);
      HVX_Vector gv_vec = Q6_V_vsplat_R(gb);

      for (int i = 0; i < Vd; ++i) {
        __fp16* Srow = S + (size_t)i * Kd;

        HVX_Vector acc_lo = Q6_V_vzero(), acc_hi = Q6_V_vzero();
        for (int jj = 0; jj < vlen; ++jj) {
          HVX_Vector sv = *(const HVX_Vector*)(Srow + jj * EPV);
          HVX_Vector kv = *(const HVX_Vector*)(kn16 + jj * EPV);
          HVX_VectorPair p = Q6_Wsf_vmpy_VhfVhf(sv, kv);
          acc_lo = Q6_Vsf_vadd_VsfVsf(acc_lo, Q6_V_lo_W(p));
          acc_hi = Q6_Vsf_vadd_VsfVsf(acc_hi, Q6_V_hi_W(p));
        }
        const float sk = hvx_hsum_sf(acc_lo, acc_hi);

        const float di_f = ((float)vh[i] - g_exp * sk) * bv;
        delta[i] = di_f;

        __fp16 d16 = (__fp16)di_f;
        uint32_t db;
        __builtin_memcpy(&db, &d16, sizeof(__fp16));
        db = (db & 0xFFFFu) | (db << 16);
        HVX_Vector dv_vec = Q6_V_vsplat_R(db);

        HVX_Vector acc2_lo = Q6_V_vzero(), acc2_hi = Q6_V_vzero();
        for (int jj = 0; jj < vlen; ++jj) {
          HVX_Vector sv = *(const HVX_Vector*)(Srow + jj * EPV);
          HVX_Vector kv = *(const HVX_Vector*)(kn16 + jj * EPV);
          HVX_Vector qv = *(const HVX_Vector*)(qn16 + jj * EPV);
          HVX_Vector gs = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(sv, gv_vec));
          HVX_Vector dk = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(kv, dv_vec));
          HVX_Vector ns = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(gs, dk));
          *(HVX_Vector*)(Srow + jj * EPV) = ns;
          HVX_VectorPair p2 = Q6_Wsf_vmpy_VhfVhf(ns, qv);
          acc2_lo = Q6_Vsf_vadd_VsfVsf(acc2_lo, Q6_V_lo_W(p2));
          acc2_hi = Q6_Vsf_vadd_VsfVsf(acc2_hi, Q6_V_hi_W(p2));
        }
        oh[i] = (__fp16)hvx_hsum_sf(acc2_lo, acc2_hi);
      }
    }
  }
  return JIT_DEV_STATUS_SUCCESS;
}

QAIC_KERNEL_API uint32_t gdn_recurrent_prefill(
    const AicJitEntryPointConfig* cfg, const AicJitPointerArray* ptrs) {
  if (ptrs->numPointers < 17) return JIT_DEV_ERROR_INVALID_PARAMETER;
  const __fp16* q = (const __fp16*)*ptrs->pointers;
  const __fp16* k = (const __fp16*)ptrs->pointers[1];
  const __fp16* v = (const __fp16*)ptrs->pointers[2];
  const float* g = (const float*)ptrs->pointers[3];
  const float* beta = (const float*)ptrs->pointers[4];
  __fp16* o = (__fp16*)ptrs->pointers[5];
  __fp16* ssm = (__fp16*)ptrs->pointers[6];
  const int32_t* qstart = (const int32_t*)ptrs->pointers[7];
  const int32_t* state_idx = (const int32_t*)ptrs->pointers[8];
  const int32_t* has_init = (const int32_t*)ptrs->pointers[9];
  const int num_seqs = *(const int32_t*)ptrs->pointers[10];
  const int H = *(const int32_t*)ptrs->pointers[11];
  const int HV = *(const int32_t*)ptrs->pointers[12];
  const int Kd = *(const int32_t*)ptrs->pointers[13];
  const int Vd = *(const int32_t*)ptrs->pointers[14];
  const float scale = *(const float*)ptrs->pointers[15];
  const float l2eps = *(const float*)ptrs->pointers[16];
  const int heads_per_group = HV / H;

  const uint32_t localTid = cfg->threadID % cfg->numThreads;
  const uint32_t globalTid = cfg->coreID * cfg->numThreads + localTid;
  const uint32_t globalThreads = cfg->numCores * cfg->numThreads;

  float qn[256] __attribute__((aligned(128)));
  float kn[256] __attribute__((aligned(128)));
  float delta[256] __attribute__((aligned(128)));
  __fp16 qn16[256] __attribute__((aligned(128)));
  __fp16 kn16[256] __attribute__((aligned(128)));

  constexpr int EPV = (int)(sizeof(HVX_Vector) / sizeof(__fp16));
  const int vlen = Kd / EPV;

  const int total = num_seqs * HV;
  for (int idx = (int)globalTid; idx < total; idx += (int)globalThreads) {
    const int s = idx / HV;
    const int h = idx % HV;
    const int hk = h / heads_per_group;
    const int bos = qstart[s];
    const int eos = qstart[s + 1];
    const int slot = state_idx[s];
    __fp16* S = ssm + ((size_t)slot * HV + h) * Vd * Kd;

    if (!has_init[s]) {
      for (int e = 0; e < Vd * Kd; ++e) S[e] = (__fp16)0.0f;
    }

    for (int t = bos; t < eos; ++t) {
      const __fp16* qh = q + ((size_t)t * H + hk) * Kd;
      const __fp16* kh = k + ((size_t)t * H + hk) * Kd;
      const __fp16* vh = v + ((size_t)t * HV + h) * Vd;
      __fp16* oh = o + ((size_t)t * HV + h) * Vd;
      const float gv = g[(size_t)t * HV + h];
      const float bv = beta[(size_t)t * HV + h];
      const float g_exp = __builtin_expf(gv);

      float qss = 0.0f, kss = 0.0f;
      for (int j = 0; j < Kd; ++j) {
        const float qq = (float)qh[j];
        const float kk = (float)kh[j];
        qn[j] = qq;
        kn[j] = kk;
        qss += qq * qq;
        kss += kk * kk;
      }
      const float qinv = 1.0f / __builtin_sqrtf(qss + l2eps);
      const float kinv = 1.0f / __builtin_sqrtf(kss + l2eps);
      for (int j = 0; j < Kd; ++j) {
        qn[j] = qn[j] * qinv * scale;
        kn[j] = kn[j] * kinv;
        qn16[j] = (__fp16)qn[j];
        kn16[j] = (__fp16)kn[j];
      }

      __fp16 g16t = (__fp16)g_exp;
      uint32_t gb;
      __builtin_memcpy(&gb, &g16t, sizeof(__fp16));
      gb = (gb & 0xFFFFu) | (gb << 16);
      HVX_Vector gv_vec = Q6_V_vsplat_R(gb);

      for (int i = 0; i < Vd; ++i) {
        __fp16* Srow = S + (size_t)i * Kd;

        HVX_Vector acc_lo = Q6_V_vzero(), acc_hi = Q6_V_vzero();
        for (int jj = 0; jj < vlen; ++jj) {
          HVX_Vector sv = *(const HVX_Vector*)(Srow + jj * EPV);
          HVX_Vector kv = *(const HVX_Vector*)(kn16 + jj * EPV);
          HVX_VectorPair p = Q6_Wsf_vmpy_VhfVhf(sv, kv);
          acc_lo = Q6_Vsf_vadd_VsfVsf(acc_lo, Q6_V_lo_W(p));
          acc_hi = Q6_Vsf_vadd_VsfVsf(acc_hi, Q6_V_hi_W(p));
        }
        const float sk = hvx_hsum_sf(acc_lo, acc_hi);

        const float di_f = ((float)vh[i] - g_exp * sk) * bv;

        __fp16 d16 = (__fp16)di_f;
        uint32_t db;
        __builtin_memcpy(&db, &d16, sizeof(__fp16));
        db = (db & 0xFFFFu) | (db << 16);
        HVX_Vector dv_vec = Q6_V_vsplat_R(db);

        HVX_Vector acc2_lo = Q6_V_vzero(), acc2_hi = Q6_V_vzero();
        for (int jj = 0; jj < vlen; ++jj) {
          HVX_Vector sv = *(const HVX_Vector*)(Srow + jj * EPV);
          HVX_Vector kv = *(const HVX_Vector*)(kn16 + jj * EPV);
          HVX_Vector qv = *(const HVX_Vector*)(qn16 + jj * EPV);
          HVX_Vector gs = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(sv, gv_vec));
          HVX_Vector dk = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(kv, dv_vec));
          HVX_Vector ns = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(gs, dk));
          *(HVX_Vector*)(Srow + jj * EPV) = ns;
          HVX_VectorPair p2 = Q6_Wsf_vmpy_VhfVhf(ns, qv);
          acc2_lo = Q6_Vsf_vadd_VsfVsf(acc2_lo, Q6_V_lo_W(p2));
          acc2_hi = Q6_Vsf_vadd_VsfVsf(acc2_hi, Q6_V_hi_W(p2));
        }
        oh[i] = (__fp16)hvx_hsum_sf(acc2_lo, acc2_hi);
      }
    }
  }
  return JIT_DEV_STATUS_SUCCESS;
}
