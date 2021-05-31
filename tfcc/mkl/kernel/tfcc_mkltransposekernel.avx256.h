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

namespace tfcc {

template <class T>
class _MKLTransposeKernelAVX256 {
 public:
};

#ifdef TFCC_USE_AVX2

template <>
class _MKLTransposeKernelAVX256<float> {
 public:
  static void transposeBA(const float* a, unsigned row, unsigned col, float* b);
  static void transposeACB(const float* a, unsigned depth, unsigned row, unsigned col, float* b);
  static void transposeBAC(const float* a, unsigned depth, unsigned row, unsigned col, float* b);
};

#endif

}  // namespace tfcc
