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
class _MKLQuantizationKernel {
 public:
  static void quantize(const float* a, unsigned total, double scale, int64_t offset, T* result) {
#pragma omp parallel for
    for (unsigned i = 0; i < total; ++i) {
      int64_t quantized = static_cast<int64_t>(round(a[i] * scale)) - offset;
      quantized += static_cast<int64_t>(std::numeric_limits<T>::lowest());
      quantized = std::max(quantized, static_cast<int64_t>(std::numeric_limits<T>::lowest()));
      quantized = std::min(quantized, static_cast<int64_t>(std::numeric_limits<T>::max()));
      result[i] = static_cast<T>(quantized);
    }
  }

  static void dequantize(
      const T* a, unsigned total, double scale, double minRounded, float* result) {
#pragma omp parallel for
    for (unsigned i = 0; i < total; ++i) {
      double offset =
          static_cast<double>(a[i]) - static_cast<double>(std::numeric_limits<T>::lowest());
      result[i] = static_cast<float>(minRounded + (offset * scale));
    }
  }
};

template <class TI, class TO>
class _MKLRequantizationKernel {
 public:
  static void requantize(
      const TI* a, unsigned total, double inputScale, double inputMinRounded, double outputScale,
      int64_t outputOffset, TO* result) {
#pragma omp parallel for
    for (unsigned i = 0; i < total; ++i) {
      double offset =
          static_cast<double>(a[i]) - static_cast<double>(std::numeric_limits<TI>::lowest());
      double real = inputMinRounded + (offset * inputScale);
      int64_t quantized = static_cast<int64_t>(round(real * outputScale)) - outputOffset;
      quantized += static_cast<int64_t>(std::numeric_limits<TO>::lowest());
      quantized = std::max(quantized, static_cast<int64_t>(std::numeric_limits<TO>::lowest()));
      quantized = std::min(quantized, static_cast<int64_t>(std::numeric_limits<TO>::max()));
      result[i] = static_cast<TO>(quantized);
    }
  }
};

}  // namespace tfcc
