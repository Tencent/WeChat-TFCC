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

namespace tfcc {

template <class T>
class _MKLQuantizationKernelAVX256 {
 public:
#ifdef TFCC_USE_AVX2
  static void quantize(const float* a, unsigned total, double scale, int64_t offset, T* result);
  static void dequantize(
      const T* a, unsigned total, double scale, double minRounded, float* result);
#endif
};

template <class TI, class TO>
class _MKLRequantizationKernelAVX256 {
 public:
#ifdef TFCC_USE_AVX2
  static void requantize(
      const TI* a, unsigned total, double inputScale, double inputMinRounded, double outputScale,
      int64_t outputOffset, TO* result);
#endif
};

}  // namespace tfcc
