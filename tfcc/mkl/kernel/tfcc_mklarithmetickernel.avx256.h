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
class _MKLArithmeticKernelAVX256 {};

#ifdef TFCC_USE_AVX2

template <>
class _MKLArithmeticKernelAVX256<float> {
 public:
  static void batchAdd(const float* a, const float* b, float* c, unsigned total);
  static void batchSub(const float* a, const float* b, float* c, unsigned total);
  static void batchMul(const float* a, const float* b, float* c, unsigned total);
  static void batchDiv(const float* a, const float* b, float* c, unsigned total);
};

template <>
class _MKLArithmeticKernelAVX256<double> {
 public:
  static void batchAdd(const double* a, const double* b, double* c, unsigned total);
  static void batchSub(const double* a, const double* b, double* c, unsigned total);
  static void batchMul(const double* a, const double* b, double* c, unsigned total);
  static void batchDiv(const double* a, const double* b, double* c, unsigned total);
};

#endif

}  // namespace tfcc
