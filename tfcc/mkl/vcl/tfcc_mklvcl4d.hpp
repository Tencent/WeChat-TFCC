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

extern __m256d __svml_exp4(__m256d);
extern __m256d __svml_log4(__m256d);
extern __m256d __svml_tanh4(__m256d);
}

namespace tfcc {

class MKLVcl4d {
  __m256d _ymm;

 public:
  typedef double BaseType;

  MKLVcl4d() {}
  MKLVcl4d(const MKLVcl4d& v) = default;

  explicit MKLVcl4d(double v) { _ymm = _mm256_set1_pd(v); }

  explicit MKLVcl4d(double v1, double v2, double v3, double v4) {
    _ymm = _mm256_setr_pd(v1, v2, v3, v4);
  }

  explicit MKLVcl4d(const double* p) { _ymm = _mm256_loadu_pd(p); }

  explicit MKLVcl4d(__m256d x) { _ymm = x; }

  MKLVcl4d& operator=(const MKLVcl4d& v) = default;

  MKLVcl4d& operator+=(const MKLVcl4d& v) {
    _ymm = _mm256_add_pd(_ymm, v._ymm);
    return *this;
  }

  MKLVcl4d& operator-=(const MKLVcl4d& v) {
    _ymm = _mm256_sub_pd(_ymm, v._ymm);
    return *this;
  }

  MKLVcl4d& operator*=(const MKLVcl4d& v) {
    _ymm = _mm256_mul_pd(_ymm, v._ymm);
    return *this;
  }

  MKLVcl4d& operator/=(const MKLVcl4d& v) {
    _ymm = _mm256_div_pd(_ymm, v._ymm);
    return *this;
  }

  MKLVcl4d& operator&=(const MKLVcl4d& v) {
    _ymm = _mm256_and_pd(_ymm, v._ymm);
    return *this;
  }

  MKLVcl4d& operator|=(const MKLVcl4d& v) {
    _ymm = _mm256_or_pd(_ymm, v._ymm);
    return *this;
  }

  __m256d native() const { return _ymm; }

  void set(double v) { _ymm = _mm256_set1_pd(v); }

  void set(__m256d x) { _ymm = x; }

  void setInfinite() { _ymm = _mm256_set1_pd(std::numeric_limits<double>::infinity()); }

  void setNegativeInfinite() { _ymm = _mm256_set1_pd(-std::numeric_limits<double>::infinity()); }

  void setNaN(long x) { _ymm = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FF8000000000000 + x)); }

  void load(const double* p) { _ymm = _mm256_loadu_pd(p); }

  void store(double* p) const { _mm256_storeu_pd(p, _ymm); }

 public:
  static constexpr size_t size() { return 4; }
};

static inline MKLVcl4d operator+(const MKLVcl4d& a, const MKLVcl4d& b) {
  auto x = _mm256_add_pd(a.native(), b.native());
  return MKLVcl4d(x);
}

static inline MKLVcl4d operator-(const MKLVcl4d& a, const MKLVcl4d& b) {
  auto x = _mm256_sub_pd(a.native(), b.native());
  return MKLVcl4d(x);
}

static inline MKLVcl4d operator*(const MKLVcl4d& a, const MKLVcl4d& b) {
  auto x = _mm256_mul_pd(a.native(), b.native());
  return MKLVcl4d(x);
}

static inline MKLVcl4d operator/(const MKLVcl4d& a, const MKLVcl4d& b) {
  auto x = _mm256_div_pd(a.native(), b.native());
  return MKLVcl4d(x);
}

static inline MKLVcl4d operator&(const MKLVcl4d& a, const MKLVcl4d& b) {
  auto x = _mm256_and_pd(a.native(), b.native());
  return MKLVcl4d(x);
}

static inline MKLVcl4d operator|(const MKLVcl4d& a, const MKLVcl4d& b) {
  auto x = _mm256_or_pd(a.native(), b.native());
  return MKLVcl4d(x);
}

static inline MKLVcl4d operator==(const MKLVcl4d& a, const MKLVcl4d& b) {
  auto x = _mm256_cmp_pd(a.native(), b.native(), 0);
  return MKLVcl4d(x);
}

static inline MKLVcl4d operator!=(const MKLVcl4d& a, const MKLVcl4d& b) {
  auto x = _mm256_cmp_pd(a.native(), b.native(), 4);
  return MKLVcl4d(x);
}

static inline MKLVcl4d operator<(const MKLVcl4d& a, const MKLVcl4d& b) {
  auto x = _mm256_cmp_pd(a.native(), b.native(), 1);
  return MKLVcl4d(x);
}

static inline MKLVcl4d operator<=(const MKLVcl4d& a, const MKLVcl4d& b) {
  auto x = _mm256_cmp_pd(a.native(), b.native(), 2);
  return MKLVcl4d(x);
}

static inline MKLVcl4d operator>(const MKLVcl4d& a, const MKLVcl4d& b) {
  auto x = _mm256_cmp_pd(b.native(), a.native(), 1);
  return MKLVcl4d(x);
}

static inline MKLVcl4d operator>=(const MKLVcl4d& a, const MKLVcl4d& b) {
  auto x = _mm256_cmp_pd(b.native(), a.native(), 2);
  return MKLVcl4d(x);
}

static inline MKLVcl4d operator~(const MKLVcl4d& a) {
  auto t1 = _mm256_castpd_si256(a.native());
  auto t2 = _mm256_xor_si256(t1, _mm256_set1_epi32(-1));
  auto x = _mm256_castsi256_pd(t2);
  return MKLVcl4d(x);
}

static inline MKLVcl4d operator!(const MKLVcl4d& a) {
  auto t1 = _mm256_castpd_si256(a.native());
  auto t2 = _mm256_xor_si256(t1, _mm256_set1_epi32(-1));
  auto x = _mm256_castsi256_pd(t2);
  return MKLVcl4d(x);
}

static inline MKLVcl4d max(const MKLVcl4d& a, const MKLVcl4d& b) {
  auto x = _mm256_max_pd(a.native(), b.native());
  return MKLVcl4d(x);
}

static inline MKLVcl4d mul_add(const MKLVcl4d& a, const MKLVcl4d& b, const MKLVcl4d& c) {
  auto x = _mm256_fmadd_pd(a.native(), b.native(), c.native());
  return MKLVcl4d(x);
}

static inline MKLVcl4d nmul_add(const MKLVcl4d& a, const MKLVcl4d& b, const MKLVcl4d& c) {
  auto x = _mm256_fnmadd_pd(a.native(), b.native(), c.native());
  return MKLVcl4d(x);
}

static inline MKLVcl4d round(const MKLVcl4d& a) {
  auto x = _mm256_round_pd(a.native(), 0 + 8);
  return MKLVcl4d(x);
}

static inline MKLVcl4d abs(const MKLVcl4d& a) {
  auto mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fffffffffffffffllu));
  auto x = _mm256_and_pd(a.native(), mask);
  return MKLVcl4d(x);
}

static inline MKLVcl4d is_finite(const MKLVcl4d& a) {
  auto mask = _mm256_set1_epi64x(0xffe0000000000000llu);
  auto t1 = _mm256_castpd_si256(a.native());
  auto t2 = _mm256_slli_epi64(t1, 1);
  auto t3 = _mm256_and_si256(t2, mask);
  auto t4 = _mm256_cmpeq_epi64(t3, mask);
  auto x = _mm256_castsi256_pd(_mm256_xor_si256(t4, _mm256_set1_epi32(-1)));
  return MKLVcl4d(x);
}

static inline MKLVcl4d is_subnormal(const MKLVcl4d& a) {
  auto mask = _mm256_set1_epi64x(0xffe0000000000000);
  auto zero = _mm256_set1_epi64x(0);
  auto t1 = _mm256_castpd_si256(a.native());
  auto t2 = _mm256_slli_epi64(t1, 1);
  auto t3 = _mm256_and_si256(t2, mask);
  auto t4 = _mm256_andnot_si256(mask, t2);

  auto t5 = _mm256_cmpeq_epi64(t3, zero);
  auto t6 = _mm256_cmpeq_epi64(t4, zero);

  auto x = _mm256_castsi256_pd(_mm256_andnot_si256(t6, t5));
  return MKLVcl4d(x);
}

static inline MKLVcl4d is_nan(const MKLVcl4d& a) {
  auto mask = _mm256_set1_epi64x(0xffe0000000000000llu);
  auto t1 = _mm256_castpd_si256(a.native());
  auto t2 = _mm256_slli_epi64(t1, 1);
  auto t3 = _mm256_andnot_si256(mask, t2);
  auto t4 = _mm256_cmpeq_epi64(t3, _mm256_set1_epi64x(0));
  auto t5 = _mm256_xor_si256(t4, _mm256_set1_epi32(-1));

  auto t6 = _mm256_and_si256(t2, mask);
  auto t7 = _mm256_cmpeq_epi64(t6, mask);

  auto x = _mm256_castsi256_pd(_mm256_and_si256(t5, t7));
  return MKLVcl4d(x);
}

static inline MKLVcl4d is_inf(const MKLVcl4d& a) {
  auto mask = _mm256_set1_epi64x(0xffe0000000000000);
  auto t1 = _mm256_castpd_si256(a.native());
  auto t2 = _mm256_slli_epi64(t1, 1);
  auto x = _mm256_castsi256_pd(_mm256_cmpeq_epi64(t2, mask));
  return MKLVcl4d(x);
}

static inline MKLVcl4d sign_bit(const MKLVcl4d& a) {
  auto t1 = _mm256_castpd_si256(a.native());
#ifdef __AVX512VL__
  auto t2 = _mm256_sra_epi64(t1, _mm_cvtsi32_si128(63));
#else
  auto mask = _mm256_set1_epi64x(0x8000000000000000);
  auto t2 = _mm256_and_si256(t1, mask);
  t2 = _mm256_cmpeq_epi64(t2, mask);
#endif
  auto x = _mm256_castsi256_pd(t2);
  return MKLVcl4d(x);
}

static inline MKLVcl4d sign_combine(const MKLVcl4d& a, const MKLVcl4d& b) {
  auto mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000));
  auto t1 = _mm256_and_pd(b.native(), mask);
  auto x = _mm256_xor_pd(a.native(), t1);
  return MKLVcl4d(x);
}

static inline bool horizontal_and(const MKLVcl4d& a) {
  auto mask = _mm256_castsi256_pd(_mm256_set1_epi32(-1));
  int x = _mm256_testc_pd(a.native(), mask);
  return x != 0;
}

static inline bool horizontal_or(const MKLVcl4d& a) {
  int x = _mm256_testz_pd(a.native(), a.native());
  return x == 0;
}

static inline MKLVcl4d select(const MKLVcl4d& s, const MKLVcl4d& a, const MKLVcl4d& b) {
  auto x = _mm256_blendv_pd(b.native(), a.native(), s.native());
  return MKLVcl4d(x);
}

static inline MKLVcl4d pow2in(const MKLVcl4d& n) {
  constexpr double pow2_52 = 4503599627370496.0;
  constexpr double bias = 1023.0;
  auto a = n + MKLVcl4d(bias + pow2_52);
  auto b = _mm256_castpd_si256(a.native());
  auto c = _mm256_slli_epi64(b, 52);
  return MKLVcl4d(_mm256_castsi256_pd(c));
}

static inline MKLVcl4d fraction_2(const MKLVcl4d& a) {
  auto mask1 = _mm256_set1_epi64x(0x000fffffffffffffllu);
  auto mask2 = _mm256_set1_epi64x(0x3fe0000000000000llu);
  auto t1 = _mm256_castpd_si256(a.native());
  auto t2 = _mm256_and_si256(t1, mask1);
  auto t3 = _mm256_or_si256(t2, mask2);
  auto x = _mm256_castsi256_pd(t3);
  return MKLVcl4d(x);
}

static inline MKLVcl4d exponent_f(const MKLVcl4d& a) {
#ifdef __AVX512VL__
  auto x = _mm256_getexp_pd(a.native());
  return MKLVcl4d(x);
#else
  union DoubleULongUnion {
    double f;
    uint64_t i;
  };

  constexpr double pow2_52 = 4503599627370496.0;
  constexpr double bias = 1023.0;
  constexpr DoubleULongUnion upow2_52 = {pow2_52};

  auto t1 = _mm256_castpd_si256(a.native());
  auto t2 = _mm256_srli_epi64(t1, 52);
  auto t3 = _mm256_or_si256(t2, _mm256_set1_epi64x(upow2_52.i));
  auto x = _mm256_castsi256_pd(t3);
  return MKLVcl4d(x) - MKLVcl4d(pow2_52 + bias);
#endif
}

static inline MKLVcl4d exp(const MKLVcl4d& a) {
#ifdef TFCC_USE_SVML
  auto x = __svml_exp4(a.native());
  return MKLVcl4d(x);
#else
  return taylor_exp_d(a);
#endif
}

static inline MKLVcl4d log(const MKLVcl4d& a) {
#ifdef TFCC_USE_SVML
  auto x = __svml_log4(a.native());
  return MKLVcl4d(x);
#else
  return taylor_log_d(a);
#endif
}

static inline MKLVcl4d tanh(const MKLVcl4d& a) {
#ifdef TFCC_USE_SVML
  auto x = __svml_tanh4(a.native());
  return MKLVcl4d(x);
#else
  return taylor_tanh_d(a);
#endif
}

}  // namespace tfcc
