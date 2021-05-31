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

namespace tfcc {

class MKLVcl4q {
  __m256i _ymm;

 public:
  typedef int64_t BaseType;

  MKLVcl4q() {}
  MKLVcl4q(const MKLVcl4q& v) = default;

  explicit MKLVcl4q(int64_t v) { _ymm = _mm256_set1_epi64x(v); }

  explicit MKLVcl4q(int64_t v1, int64_t v2, int64_t v3, int64_t v4) {
    _ymm = _mm256_setr_epi64x(v1, v2, v3, v4);
  }

  explicit MKLVcl4q(const int64_t* p) {
    _ymm = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  }

  explicit MKLVcl4q(__m256i x) { _ymm = x; }

  MKLVcl4q& operator=(const MKLVcl4q& v) = default;

  MKLVcl4q& operator+=(const MKLVcl4q& v) {
    _ymm = _mm256_add_epi64(_ymm, v._ymm);
    return *this;
  }

  MKLVcl4q& operator-=(const MKLVcl4q& v) {
    _ymm = _mm256_sub_epi64(_ymm, v._ymm);
    return *this;
  }

  MKLVcl4q& operator&=(const MKLVcl4q& v) {
    _ymm = _mm256_and_si256(_ymm, v._ymm);
    return *this;
  }

  MKLVcl4q& operator|=(const MKLVcl4q& v) {
    _ymm = _mm256_or_si256(_ymm, v._ymm);
    return *this;
  }

  __m256i native() const { return _ymm; }

  void set(int64_t v) { _ymm = _mm256_set1_epi64x(v); }

  void set(__m256i x) { _ymm = x; }

  void load(const int64_t* p) { _ymm = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); }

  void store(int64_t* p) const { _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), _ymm); }

 public:
  static constexpr size_t size() { return 4; }
};

static inline MKLVcl4q operator+(const MKLVcl4q& a, const MKLVcl4q& b) {
  auto x = _mm256_add_epi64(a.native(), b.native());
  return MKLVcl4q(x);
}

static inline MKLVcl4q operator-(const MKLVcl4q& a, const MKLVcl4q& b) {
  auto x = _mm256_sub_epi64(a.native(), b.native());
  return MKLVcl4q(x);
}

static inline MKLVcl4q operator&(const MKLVcl4q& a, const MKLVcl4q& b) {
  auto x = _mm256_and_si256(a.native(), b.native());
  return MKLVcl4q(x);
}

static inline MKLVcl4q operator|(const MKLVcl4q& a, const MKLVcl4q& b) {
  auto x = _mm256_or_si256(a.native(), b.native());
  return MKLVcl4q(x);
}

static inline MKLVcl4q operator==(const MKLVcl4q& a, const MKLVcl4q& b) {
  auto x = _mm256_cmpeq_epi64(a.native(), b.native());
  return MKLVcl4q(x);
}

static inline MKLVcl4q operator<(const MKLVcl4q& a, const MKLVcl4q& b) {
  auto x = _mm256_cmpgt_epi64(b.native(), a.native());
  return MKLVcl4q(x);
}

static inline MKLVcl4q operator>(const MKLVcl4q& a, const MKLVcl4q& b) {
  auto x = _mm256_cmpgt_epi64(a.native(), b.native());
  return MKLVcl4q(x);
}

static inline MKLVcl4q max(const MKLVcl4q& a, const MKLVcl4q& b) {
  auto mask = _mm256_cmpgt_epi64(a.native(), b.native());
  auto x = _mm256_blendv_epi8(b.native(), a.native(), mask);
  return MKLVcl4q(x);
}

static inline MKLVcl4q min(const MKLVcl4q& a, const MKLVcl4q& b) {
  auto mask = _mm256_cmpgt_epi64(b.native(), a.native());
  auto x = _mm256_blendv_epi8(b.native(), a.native(), mask);
  return MKLVcl4q(x);
}

}  // namespace tfcc
