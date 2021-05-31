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
class _MKLTransformationKernelAVX256 {
 public:
};

#ifdef TFCC_USE_AVX2

template <>
class _MKLTransformationKernelAVX256<float> {
 public:
  static void transform(const float* a, unsigned total, float alpha, float beta, float* b);
  static void transform2(const float* a, unsigned total, float alpha, float beta, float* b);
  static void transform3(const float* a, unsigned total, float alpha, float beta, float* b);
  static void transform4(const float* a, unsigned total, float alpha, float beta, float* b);
  static void transform5(const float* a, unsigned total, float alpha, float beta, float* b);
  static void transform6(const float* a, unsigned total, float alpha, float beta, float* b);
};

template <>
class _MKLTransformationKernelAVX256<double> {
 public:
  static void transform(const double* a, unsigned total, double alpha, double beta, double* b);
  static void transform2(const double* a, unsigned total, double alpha, double beta, double* b);
  static void transform3(const double* a, unsigned total, double alpha, double beta, double* b);
  static void transform4(const double* a, unsigned total, double alpha, double beta, double* b);
  static void transform5(const double* a, unsigned total, double alpha, double beta, double* b);
  static void transform6(const double* a, unsigned total, double alpha, double beta, double* b);
};

#endif

}  // namespace tfcc
