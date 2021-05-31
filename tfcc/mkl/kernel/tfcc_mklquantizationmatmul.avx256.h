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

class _MKLQuantizationMatmulAVX256 {
 public:
  static void quantizedMatmulN1(
      const uint8_t* a, const int32_t* reduceA, int32_t offsetA, const int8_t* b, int32_t reduceB,
      int32_t offsetB, unsigned m, unsigned k, int32_t* c);

  static void quantizedMatmulColMajor(
      unsigned batch, const int8_t* a, unsigned strideA, const uint8_t* b, unsigned strideB,
      unsigned m, unsigned n, unsigned k, int32_t* c, unsigned strideC);
};

}  // namespace tfcc
