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
#include <cmath>
#include <cstddef>

extern "C" {

extern __m512 __svml_expf16(__m512);
extern __m512 __svml_logf16(__m512);
extern __m512 __svml_tanhf16(__m512);
}

namespace tfcc {

class MKLVcl16f {
  __m512 _zmm;

 public:
  typedef float BaseType;

  MKLVcl16f() {}
  MKLVcl16f(const MKLVcl16f& v) = default;

  explicit MKLVcl16f(float v) { _zmm = _mm512_set1_ps(v); }

  explicit MKLVcl16f(
      float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9,
      float v10, float v11, float v12, float v13, float v14, float v15, float v16) {
    _zmm = _mm512_setr_ps(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16);
  }

  explicit MKLVcl16f(const float* p) { _zmm = _mm512_loadu_ps(p); }

  explicit MKLVcl16f(__m512 x) { _zmm = x; }

  MKLVcl16f& operator=(const MKLVcl16f& v) = default;

  MKLVcl16f& operator+=(const MKLVcl16f& v) {
    _zmm = _mm512_add_ps(_zmm, v._zmm);
    return *this;
  }

  MKLVcl16f& operator-=(const MKLVcl16f& v) {
    _zmm = _mm512_sub_ps(_zmm, v._zmm);
    return *this;
  }

  MKLVcl16f& operator*=(const MKLVcl16f& v) {
    _zmm = _mm512_mul_ps(_zmm, v._zmm);
    return *this;
  }

  MKLVcl16f& operator/=(const MKLVcl16f& v) {
    _zmm = _mm512_div_ps(_zmm, v._zmm);
    return *this;
  }

  MKLVcl16f& operator&=(const MKLVcl16f& v) {
    _zmm = _mm512_and_ps(_zmm, v._zmm);
    return *this;
  }

  MKLVcl16f& operator|=(const MKLVcl16f& v) {
    _zmm = _mm512_or_ps(_zmm, v._zmm);
    return *this;
  }

  __m512 native() const { return _zmm; }

  void set(float v) { _zmm = _mm512_set1_ps(v); }

  void set(__m512 x) { _zmm = x; }

  void load(const float* p) { _zmm = _mm512_loadu_ps(p); }

  void store(float* p) const { _mm512_storeu_ps(p, _zmm); }

 public:
  static constexpr size_t size() { return 16; }
};

static inline MKLVcl16f operator+(const MKLVcl16f& a, const MKLVcl16f& b) {
  auto x = _mm512_add_ps(a.native(), b.native());
  return MKLVcl16f(x);
}

static inline MKLVcl16f operator-(const MKLVcl16f& a, const MKLVcl16f& b) {
  auto x = _mm512_sub_ps(a.native(), b.native());
  return MKLVcl16f(x);
}

static inline MKLVcl16f operator*(const MKLVcl16f& a, const MKLVcl16f& b) {
  auto x = _mm512_mul_ps(a.native(), b.native());
  return MKLVcl16f(x);
}

static inline MKLVcl16f operator/(const MKLVcl16f& a, const MKLVcl16f& b) {
  auto x = _mm512_div_ps(a.native(), b.native());
  return MKLVcl16f(x);
}

static inline MKLVcl16f operator&(const MKLVcl16f& a, const MKLVcl16f& b) {
  auto x = _mm512_and_ps(a.native(), b.native());
  return MKLVcl16f(x);
}

static inline MKLVcl16f operator|(const MKLVcl16f& a, const MKLVcl16f& b) {
  auto x = _mm512_or_ps(a.native(), b.native());
  return MKLVcl16f(x);
}

static inline MKLVcl16f operator==(const MKLVcl16f& a, const MKLVcl16f& b) {
  auto x = _mm512_cmp_ps_mask(a.native(), b.native(), 0);
  return MKLVcl16f(x);
}

static inline MKLVcl16f operator!=(const MKLVcl16f& a, const MKLVcl16f& b) {
  auto x = _mm512_cmp_ps_mask(a.native(), b.native(), 4);
  return MKLVcl16f(x);
}

static inline MKLVcl16f operator<(const MKLVcl16f& a, const MKLVcl16f& b) {
  auto x = _mm512_cmp_ps_mask(a.native(), b.native(), 1);
  return MKLVcl16f(x);
}

static inline MKLVcl16f operator<=(const MKLVcl16f& a, const MKLVcl16f& b) {
  auto x = _mm512_cmp_ps_mask(a.native(), b.native(), 2);
  return MKLVcl16f(x);
}

static inline MKLVcl16f operator>(const MKLVcl16f& a, const MKLVcl16f& b) {
  auto x = _mm512_cmp_ps_mask(b.native(), a.native(), 1);
  return MKLVcl16f(x);
}

static inline MKLVcl16f operator>=(const MKLVcl16f& a, const MKLVcl16f& b) {
  auto x = _mm512_cmp_ps_mask(b.native(), a.native(), 2);
  return MKLVcl16f(x);
}

static inline MKLVcl16f max(const MKLVcl16f& a, const MKLVcl16f& b) {
  auto x = _mm512_max_ps(a.native(), b.native());
  return MKLVcl16f(x);
}

static inline MKLVcl16f mul_add(const MKLVcl16f& a, const MKLVcl16f& b, const MKLVcl16f& c) {
  auto x = _mm512_fmadd_ps(a.native(), b.native(), c.native());
  return MKLVcl16f(x);
}

static inline MKLVcl16f nmul_add(const MKLVcl16f& a, const MKLVcl16f& b, const MKLVcl16f& c) {
  auto x = _mm512_fnmadd_ps(a.native(), b.native(), c.native());
  return MKLVcl16f(x);
}

static inline MKLVcl16f round(const MKLVcl16f& a) {
  auto x = _mm512_roundscale_ps(a.native(), 0 + 8);
  return MKLVcl16f(x);
}

static inline MKLVcl16f abs(const MKLVcl16f& a) {
  auto mask = _mm512_set1_epi32(0x7fffffff);
  auto x = _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(a.native()), mask));
  return MKLVcl16f(x);
}

static inline MKLVcl16f fraction_2(const MKLVcl16f& a) {
  auto x = _mm512_getmant_ps(a.native(), _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_zero);
  return MKLVcl16f(x);
}

static inline MKLVcl16f exponent_f(const MKLVcl16f& a) {
  auto x = _mm512_getexp_ps(a.native());
  return MKLVcl16f(x);
}

static inline MKLVcl16f exp(const MKLVcl16f& a) {
#ifdef TFCC_USE_SVML
  auto x = __svml_expf16(a.native());
  return MKLVcl16f(x);
#else
  float tmp[16];
  a.store(tmp);
  for (size_t i = 0; i < 16; ++i) tmp[i] = ::exp(i);
  MKLVcl16f result;
  result.load(tmp);
  return result;
#endif
}

static inline MKLVcl16f log(const MKLVcl16f& a) {
#ifdef TFCC_USE_SVML
  auto x = __svml_logf16(a.native());
  return MKLVcl16f(x);
#else
  float tmp[16];
  a.store(tmp);
  for (size_t i = 0; i < 16; ++i) tmp[i] = ::log(i);
  MKLVcl16f result;
  result.load(tmp);
  return result;
#endif
}

static inline MKLVcl16f tanh(const MKLVcl16f& a) {
#ifdef TFCC_USE_SVML
  auto x = __svml_tanhf16(a.native());
  return MKLVcl16f(x);
#else
  float tmp[16];
  a.store(tmp);
  for (size_t i = 0; i < 16; ++i) tmp[i] = ::tanh(i);
  MKLVcl16f result;
  result.load(tmp);
  return result;
#endif
}

}  // namespace tfcc
