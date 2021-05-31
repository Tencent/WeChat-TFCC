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

#include <cstdint>

#include "tfcc_mklvcl4d.hpp"
#include "tfcc_mklvcl4q.hpp"
#include "tfcc_mklvcl8f.hpp"

namespace tfcc {

static inline MKLVcl4d extend_low(const MKLVcl8f& v) {
  auto x = _mm256_cvtps_pd(_mm256_castps256_ps128(v.native()));
  return MKLVcl4d(x);
}

static inline MKLVcl4d extend_high(const MKLVcl8f& v) {
  auto x = _mm256_cvtps_pd(_mm256_extractf128_ps(v.native(), 1));
  return MKLVcl4d(x);
}

static inline MKLVcl8f compress(const MKLVcl4d& low, const MKLVcl4d& high) {
  __m128 tlo = _mm256_cvtpd_ps(low.native());
  __m128 thi = _mm256_cvtpd_ps(high.native());
  auto x = _mm256_insertf128_ps(_mm256_castps128_ps256(tlo), thi, 1);
  return MKLVcl8f(x);
}

static inline MKLVcl4q truncate_to_int64(const MKLVcl4d& v) {
  double tmp[4];
  v.store(tmp);
  return MKLVcl4q(
      static_cast<int64_t>(tmp[0]), static_cast<int64_t>(tmp[1]), static_cast<int64_t>(tmp[2]),
      static_cast<int64_t>(tmp[3]));
}

static inline MKLVcl4q round_to_int64(const MKLVcl4d& v) { return truncate_to_int64(round(v)); }

}  // namespace tfcc
