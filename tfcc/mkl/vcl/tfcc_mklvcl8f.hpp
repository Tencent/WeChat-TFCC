// Copyright 2021 Wechat Group, Tencent
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "tfcc_mklvclmath.hpp"

extern "C" {

extern __m256 __svml_expf8(__m256);
extern __m256 __svml_logf8(__m256);
extern __m256 __svml_tanhf8(__m256);
}

namespace tfcc {

class MKLVcl8f {
  __m256 _ymm;

 public:
  typedef float BaseType;

  MKLVcl8f() {}
  MKLVcl8f(const MKLVcl8f& v) = default;

  explicit MKLVcl8f(float v) { _ymm = _mm256_set1_ps(v); }

  explicit MKLVcl8f(
      float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8) {
    _ymm = _mm256_setr_ps(v1, v2, v3, v4, v5, v6, v7, v8);
  }

  explicit MKLVcl8f(const float* p) { _ymm = _mm256_loadu_ps(p); }

  explicit MKLVcl8f(__m256 x) { _ymm = x; }

  MKLVcl8f& operator=(const MKLVcl8f& v) = default;

  MKLVcl8f& operator+=(const MKLVcl8f& v) {
    _ymm = _mm256_add_ps(_ymm, v._ymm);
    return *this;
  }

  MKLVcl8f& operator-=(const MKLVcl8f& v) {
    _ymm = _mm256_sub_ps(_ymm, v._ymm);
    return *this;
  }

  MKLVcl8f& operator*=(const MKLVcl8f& v) {
    _ymm = _mm256_mul_ps(_ymm, v._ymm);
    return *this;
  }

  MKLVcl8f& operator/=(const MKLVcl8f& v) {
    _ymm = _mm256_div_ps(_ymm, v._ymm);
    return *this;
  }

  MKLVcl8f& operator&=(const MKLVcl8f& v) {
    _ymm = _mm256_and_ps(_ymm, v._ymm);
    return *this;
  }

  MKLVcl8f& operator|=(const MKLVcl8f& v) {
    _ymm = _mm256_or_ps(_ymm, v._ymm);
    return *this;
  }

  __m256 native() const { return _ymm; }

  void set(float v) { _ymm = _mm256_set1_ps(v); }

  void set(__m256 x) { _ymm = x; }

  void setInfinite() { _ymm = _mm256_set1_ps(std::numeric_limits<float>::infinity()); }

  void setNegativeInfinite() { _ymm = _mm256_set1_ps(-std::numeric_limits<float>::infinity()); }

  void setNaN(int x) { _ymm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fc00000 + x)); }

  void load(const float* p) { _ymm = _mm256_loadu_ps(p); }

  void store(float* p) const { _mm256_storeu_ps(p, _ymm); }

 public:
  static constexpr size_t size() { return 8; }
};

static inline MKLVcl8f operator+(const MKLVcl8f& a, const MKLVcl8f& b) {
  auto x = _mm256_add_ps(a.native(), b.native());
  return MKLVcl8f(x);
}

static inline MKLVcl8f operator-(const MKLVcl8f& a, const MKLVcl8f& b) {
  auto x = _mm256_sub_ps(a.native(), b.native());
  return MKLVcl8f(x);
}

static inline MKLVcl8f operator*(const MKLVcl8f& a, const MKLVcl8f& b) {
  auto x = _mm256_mul_ps(a.native(), b.native());
  return MKLVcl8f(x);
}

static inline MKLVcl8f operator/(const MKLVcl8f& a, const MKLVcl8f& b) {
  auto x = _mm256_div_ps(a.native(), b.native());
  return MKLVcl8f(x);
}

static inline MKLVcl8f operator&(const MKLVcl8f& a, const MKLVcl8f& b) {
  auto x = _mm256_and_ps(a.native(), b.native());
  return MKLVcl8f(x);
}

static inline MKLVcl8f operator|(const MKLVcl8f& a, const MKLVcl8f& b) {
  auto x = _mm256_or_ps(a.native(), b.native());
  return MKLVcl8f(x);
}

static inline MKLVcl8f operator==(const MKLVcl8f& a, const MKLVcl8f& b) {
  auto x = _mm256_cmp_ps(a.native(), b.native(), 0);
  return MKLVcl8f(x);
}

static inline MKLVcl8f operator!=(const MKLVcl8f& a, const MKLVcl8f& b) {
  auto x = _mm256_cmp_ps(a.native(), b.native(), 4);
  return MKLVcl8f(x);
}

static inline MKLVcl8f operator<(const MKLVcl8f& a, const MKLVcl8f& b) {
  auto x = _mm256_cmp_ps(a.native(), b.native(), 1);
  return MKLVcl8f(x);
}

static inline MKLVcl8f operator<=(const MKLVcl8f& a, const MKLVcl8f& b) {
  auto x = _mm256_cmp_ps(a.native(), b.native(), 2);
  return MKLVcl8f(x);
}

static inline MKLVcl8f operator>(const MKLVcl8f& a, const MKLVcl8f& b) {
  auto x = _mm256_cmp_ps(b.native(), a.native(), 1);
  return MKLVcl8f(x);
}

static inline MKLVcl8f operator>=(const MKLVcl8f& a, const MKLVcl8f& b) {
  auto x = _mm256_cmp_ps(b.native(), a.native(), 2);
  return MKLVcl8f(x);
}

static inline MKLVcl8f operator~(const MKLVcl8f& a) {
  auto t1 = _mm256_castps_si256(a.native());
  auto t2 = _mm256_xor_si256(t1, _mm256_set1_epi32(-1));
  auto x = _mm256_castsi256_ps(t2);
  return MKLVcl8f(x);
}

static inline MKLVcl8f operator!(const MKLVcl8f& a) {
  auto t1 = _mm256_castps_si256(a.native());
  auto t2 = _mm256_xor_si256(t1, _mm256_set1_epi32(-1));
  auto x = _mm256_castsi256_ps(t2);
  return MKLVcl8f(x);
}

static inline MKLVcl8f max(const MKLVcl8f& a, const MKLVcl8f& b) {
  auto x = _mm256_max_ps(a.native(), b.native());
  return MKLVcl8f(x);
}

static inline MKLVcl8f mul_add(const MKLVcl8f& a, const MKLVcl8f& b, const MKLVcl8f& c) {
  auto x = _mm256_fmadd_ps(a.native(), b.native(), c.native());
  return MKLVcl8f(x);
}

static inline MKLVcl8f nmul_add(const MKLVcl8f& a, const MKLVcl8f& b, const MKLVcl8f& c) {
  auto x = _mm256_fnmadd_ps(a.native(), b.native(), c.native());
  return MKLVcl8f(x);
}

static inline MKLVcl8f round(const MKLVcl8f& a) {
  auto x = _mm256_round_ps(a.native(), 0 + 8);
  return MKLVcl8f(x);
}

static inline MKLVcl8f abs(const MKLVcl8f& a) {
  auto mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  auto x = _mm256_and_ps(a.native(), mask);
  return MKLVcl8f(x);
}

static inline MKLVcl8f is_finite(const MKLVcl8f& a) {
  auto mask = _mm256_set1_epi32(0xff000000);
  auto t1 = _mm256_castps_si256(a.native());
  auto t2 = _mm256_slli_epi32(t1, 1);
  auto t3 = _mm256_and_si256(t2, mask);
  auto t4 = _mm256_cmpeq_epi32(t3, mask);
  auto x = _mm256_castsi256_ps(_mm256_xor_si256(t4, _mm256_set1_epi32(-1)));
  return MKLVcl8f(x);
}

static inline MKLVcl8f is_subnormal(const MKLVcl8f& a) {
  auto mask = _mm256_set1_epi32(0xff000000);
  auto zero = _mm256_set1_epi32(0);
  auto t1 = _mm256_castps_si256(a.native());
  auto t2 = _mm256_slli_epi32(t1, 1);
  auto t3 = _mm256_and_si256(t2, mask);
  auto t4 = _mm256_andnot_si256(mask, t2);

  auto t5 = _mm256_cmpeq_epi32(t3, zero);
  auto t6 = _mm256_cmpeq_epi32(t4, zero);

  auto x = _mm256_castsi256_ps(_mm256_andnot_si256(t6, t5));
  return MKLVcl8f(x);
}

static inline MKLVcl8f is_nan(const MKLVcl8f& a) {
  auto mask = _mm256_set1_epi32(0xff000000);
  auto t1 = _mm256_castps_si256(a.native());
  auto t2 = _mm256_slli_epi32(t1, 1);
  auto t3 = _mm256_andnot_si256(mask, t2);
  auto t4 = _mm256_cmpeq_epi32(t3, _mm256_set1_epi64x(0));
  auto t5 = _mm256_xor_si256(t4, _mm256_set1_epi32(-1));

  auto t6 = _mm256_and_si256(t2, mask);
  auto t7 = _mm256_cmpeq_epi32(t6, mask);

  auto x = _mm256_castsi256_ps(_mm256_and_si256(t5, t7));
  return MKLVcl8f(x);
}

static inline MKLVcl8f is_inf(const MKLVcl8f& a) {
  auto mask = _mm256_set1_epi32(0xff000000);
  auto t1 = _mm256_castps_si256(a.native());
  auto t2 = _mm256_slli_epi32(t1, 1);
  auto x = _mm256_castsi256_ps(_mm256_cmpeq_epi32(t2, mask));
  return MKLVcl8f(x);
}

static inline MKLVcl8f sign_bit(const MKLVcl8f& a) {
  auto t1 = _mm256_castps_si256(a.native());
  auto t2 = _mm256_sra_epi32(t1, _mm_cvtsi32_si128(31));
  auto x = _mm256_castsi256_ps(t2);
  return MKLVcl8f(x);
}

static inline MKLVcl8f sign_combine(const MKLVcl8f& a, const MKLVcl8f& b) {
  auto mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
  auto t1 = _mm256_and_ps(b.native(), mask);
  auto x = _mm256_xor_ps(a.native(), t1);
  return MKLVcl8f(x);
}

static inline bool horizontal_and(const MKLVcl8f& a) {
  auto mask = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
  int x = _mm256_testc_ps(a.native(), mask);
  return x != 0;
}

static inline bool horizontal_or(const MKLVcl8f& a) {
  int x = _mm256_testz_ps(a.native(), a.native());
  return x == 0;
}

static inline MKLVcl8f select(const MKLVcl8f& s, const MKLVcl8f& a, const MKLVcl8f& b) {
  auto x = _mm256_blendv_ps(b.native(), a.native(), s.native());
  return MKLVcl8f(x);
}

static inline MKLVcl8f pow2in(const MKLVcl8f& n) {
  constexpr float pow2_23 = 8388608.0;
  constexpr float bias = 127.0;
  auto a = n + MKLVcl8f(bias + pow2_23);
  auto b = _mm256_castps_si256(a.native());
  auto c = _mm256_slli_epi32(b, 23);
  return MKLVcl8f(_mm256_castsi256_ps(c));
}

static inline MKLVcl8f fraction_2(const MKLVcl8f& a) {
  auto mask1 = _mm256_set1_epi32(0x007fffff);
  auto mask2 = _mm256_set1_epi32(0x3f000000);
  auto t1 = _mm256_castps_si256(a.native());
  auto t2 = _mm256_and_si256(t1, mask1);
  auto t3 = _mm256_or_si256(t2, mask2);
  auto x = _mm256_castsi256_ps(t3);
  return MKLVcl8f(x);
}

static inline MKLVcl8f exponent_f(const MKLVcl8f& a) {
#ifdef __AVX512VL__
  auto x = _mm256_getexp_ps(a.native());
  return MKLVcl8f(x);
#else
  union FloatUIntUnion {
    float f;
    uint32_t i;
  };

  constexpr float pow2_23 = 8388608.0f;
  constexpr float bias = 127.f;
  constexpr FloatUIntUnion upow2_23 = {pow2_23};

  auto t1 = _mm256_castps_si256(a.native());
  auto t2 = _mm256_srli_epi32(t1, 23);
  auto t3 = _mm256_or_si256(t2, _mm256_set1_epi32(upow2_23.i));
  auto x = _mm256_castsi256_ps(t3);
  return MKLVcl8f(x) - MKLVcl8f(pow2_23 + bias);
#endif
}

static inline MKLVcl8f exp(const MKLVcl8f& a) {
#ifdef TFCC_USE_SVML
  auto x = __svml_expf8(a.native());
  return MKLVcl8f(x);
#else
  return taylor_exp_f(a);
#endif
}

static inline MKLVcl8f log(const MKLVcl8f& a) {
#ifdef TFCC_USE_SVML
  auto x = __svml_logf8(a.native());
  return MKLVcl8f(x);
#else
  return taylor_log_f(a);
#endif
}

static inline MKLVcl8f tanh(const MKLVcl8f& a) {
#ifdef TFCC_USE_SVML
  auto x = __svml_tanhf8(a.native());
  return MKLVcl8f(x);
#else
  return taylor_tanh_f(a);
#endif
}

}  // namespace tfcc
