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

extern "C" {

extern __m512d __svml_exp8(__m512d);
extern __m512d __svml_log8(__m512d);
extern __m512d __svml_tanh8(__m512d);
}

namespace tfcc {

class MKLVcl8d {
  __m512d _zmm;

 public:
  typedef double BaseType;

  MKLVcl8d() {}
  MKLVcl8d(const MKLVcl8d& v) = default;

  explicit MKLVcl8d(double v) { _zmm = _mm512_set1_pd(v); }

  explicit MKLVcl8d(
      double v1, double v2, double v3, double v4, double v5, double v6, double v7, double v8) {
    _zmm = _mm512_setr_pd(v1, v2, v3, v4, v5, v6, v7, v8);
  }

  explicit MKLVcl8d(const double* p) { _zmm = _mm512_loadu_pd(p); }

  explicit MKLVcl8d(__m512d x) { _zmm = x; }

  MKLVcl8d& operator=(const MKLVcl8d& v) = default;

  MKLVcl8d& operator+=(const MKLVcl8d& v) {
    _zmm = _mm512_add_pd(_zmm, v._zmm);
    return *this;
  }

  MKLVcl8d& operator-=(const MKLVcl8d& v) {
    _zmm = _mm512_sub_pd(_zmm, v._zmm);
    return *this;
  }

  MKLVcl8d& operator*=(const MKLVcl8d& v) {
    _zmm = _mm512_mul_pd(_zmm, v._zmm);
    return *this;
  }

  MKLVcl8d& operator/=(const MKLVcl8d& v) {
    _zmm = _mm512_div_pd(_zmm, v._zmm);
    return *this;
  }

  MKLVcl8d& operator&=(const MKLVcl8d& v) {
    _zmm = _mm512_and_pd(_zmm, v._zmm);
    return *this;
  }

  MKLVcl8d& operator|=(const MKLVcl8d& v) {
    _zmm = _mm512_or_pd(_zmm, v._zmm);
    return *this;
  }

  __m512d native() const { return _zmm; }

  void set(double v) { _zmm = _mm512_set1_pd(v); }

  void set(__m512d x) { _zmm = x; }

  void load(const double* p) { _zmm = _mm512_loadu_pd(p); }

  void store(double* p) const { _mm512_storeu_pd(p, _zmm); }

 public:
  static constexpr size_t size() { return 8; }
};

static inline MKLVcl8d operator+(const MKLVcl8d& a, const MKLVcl8d& b) {
  auto x = _mm512_add_pd(a.native(), b.native());
  return MKLVcl8d(x);
}

static inline MKLVcl8d operator-(const MKLVcl8d& a, const MKLVcl8d& b) {
  auto x = _mm512_sub_pd(a.native(), b.native());
  return MKLVcl8d(x);
}

static inline MKLVcl8d operator*(const MKLVcl8d& a, const MKLVcl8d& b) {
  auto x = _mm512_mul_pd(a.native(), b.native());
  return MKLVcl8d(x);
}

static inline MKLVcl8d operator/(const MKLVcl8d& a, const MKLVcl8d& b) {
  auto x = _mm512_div_pd(a.native(), b.native());
  return MKLVcl8d(x);
}

static inline MKLVcl8d operator&(const MKLVcl8d& a, const MKLVcl8d& b) {
  auto x = _mm512_and_pd(a.native(), b.native());
  return MKLVcl8d(x);
}

static inline MKLVcl8d operator|(const MKLVcl8d& a, const MKLVcl8d& b) {
  auto x = _mm512_or_pd(a.native(), b.native());
  return MKLVcl8d(x);
}

static inline MKLVcl8d operator==(const MKLVcl8d& a, const MKLVcl8d& b) {
  auto x = _mm512_cmp_pd_mask(a.native(), b.native(), 0);
  return MKLVcl8d(x);
}

static inline MKLVcl8d operator!=(const MKLVcl8d& a, const MKLVcl8d& b) {
  auto x = _mm512_cmp_pd_mask(a.native(), b.native(), 4);
  return MKLVcl8d(x);
}

static inline MKLVcl8d operator<(const MKLVcl8d& a, const MKLVcl8d& b) {
  auto x = _mm512_cmp_pd_mask(a.native(), b.native(), 1);
  return MKLVcl8d(x);
}

static inline MKLVcl8d operator<=(const MKLVcl8d& a, const MKLVcl8d& b) {
  auto x = _mm512_cmp_pd_mask(a.native(), b.native(), 2);
  return MKLVcl8d(x);
}

static inline MKLVcl8d operator>(const MKLVcl8d& a, const MKLVcl8d& b) {
  auto x = _mm512_cmp_pd_mask(b.native(), a.native(), 1);
  return MKLVcl8d(x);
}

static inline MKLVcl8d operator>=(const MKLVcl8d& a, const MKLVcl8d& b) {
  auto x = _mm512_cmp_pd_mask(b.native(), a.native(), 2);
  return MKLVcl8d(x);
}

static inline MKLVcl8d max(const MKLVcl8d& a, const MKLVcl8d& b) {
  auto x = _mm512_max_pd(a.native(), b.native());
  return MKLVcl8d(x);
}

static inline MKLVcl8d mul_add(const MKLVcl8d& a, const MKLVcl8d& b, const MKLVcl8d& c) {
  auto x = _mm512_fmadd_pd(a.native(), b.native(), c.native());
  return MKLVcl8d(x);
}

static inline MKLVcl8d nmul_add(const MKLVcl8d& a, const MKLVcl8d& b, const MKLVcl8d& c) {
  auto x = _mm512_fnmadd_pd(a.native(), b.native(), c.native());
  return MKLVcl8d(x);
}

static inline MKLVcl8d round(const MKLVcl8d& a) {
  auto x = _mm512_roundscale_pd(a.native(), 0 + 8);
  return MKLVcl8d(x);
}

static inline MKLVcl8d abs(const MKLVcl8d& a) {
  auto mask = _mm512_set1_epi64(0x7fffffffffffffffllu);
  auto x = _mm512_castsi512_pd(_mm512_and_epi32(_mm512_castpd_si512(a.native()), mask));
  return MKLVcl8d(x);
}

static inline MKLVcl8d exp(const MKLVcl8d& a) {
#ifdef TFCC_USE_SVML
  auto x = __svml_exp8(a.native());
  return MKLVcl8d(x);
#else
  double tmp[8];
  a.store(tmp);
  for (size_t i = 0; i < 8; ++i) tmp[i] = ::exp(i);
  MKLVcl8d result;
  result.load(tmp);
  return result;
#endif
}

static inline MKLVcl8d log(const MKLVcl8d& a) {
#ifdef TFCC_USE_SVML
  auto x = __svml_log8(a.native());
  return MKLVcl8d(x);
#else
  double tmp[8];
  a.store(tmp);
  for (size_t i = 0; i < 8; ++i) tmp[i] = ::log(i);
  MKLVcl8d result;
  result.load(tmp);
  return result;
#endif
}

static inline MKLVcl8d tanh(const MKLVcl8d& a) {
#ifdef TFCC_USE_SVML
  auto x = __svml_tanh8(a.native());
  return MKLVcl8d(x);
#else
  double tmp[8];
  a.store(tmp);
  for (size_t i = 0; i < 8; ++i) tmp[i] = ::tanh(i);
  MKLVcl8d result;
  result.load(tmp);
  return result;
#endif
}

}  // namespace tfcc
