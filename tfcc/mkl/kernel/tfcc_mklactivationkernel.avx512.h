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
class _MKLActivationKernelAVX512 {
 public:
};

#ifdef TFCC_USE_AVX512

template <>
class _MKLActivationKernelAVX512<float> {
 public:
  static void sigmoid(const float* a, unsigned total, float* b);
  static void tanh(const float* a, unsigned total, float* b);
  static void relu(const float* a, unsigned total, float* b);
  static void leakyRelu(const float* a, unsigned total, float alpha, float* b);
  static void log(const float* a, unsigned total, float* b);
  static void rsqrt(const float* a, unsigned total, float* b);
  static void softmax(const float* a, unsigned s1, unsigned s2, unsigned s3, float* b);
  static void softmaxV2(const float* a, unsigned s1, unsigned s2, float* b);
  static void sin(const float* a, unsigned total, float* b);
  static void cos(const float* a, unsigned total, float* b);
  static void pow(const float* a, unsigned total, float exponent, float* b);
  static void gelu(const float* a, unsigned total, float* b);
  static void asin(const float* a, unsigned total, float* b);
  static void asinh(const float* a, unsigned total, float* b);
  static void acos(const float* a, unsigned total, float* b);
  static void acosh(const float* a, unsigned total, float* b);
  static void atan(const float* a, unsigned total, float* b);
  static void atanh(const float* a, unsigned total, float* b);
#  ifdef TFCC_USE_SVML
  static void erf(const float* a, unsigned total, float* b);
#  endif
};

template <>
class _MKLActivationKernelAVX512<double> {
 public:
  static void sigmoid(const double* a, unsigned total, double* b);
  static void tanh(const double* a, unsigned total, double* b);
  static void relu(const double* a, unsigned total, double* b);
  static void leakyRelu(const double* a, unsigned total, double alpha, double* b);
  static void log(const double* a, unsigned total, double* b);
  static void rsqrt(const double* a, unsigned total, double* b);
  static void softmax(const double* a, unsigned s1, unsigned s2, unsigned s3, double* b);
  static void softmaxV2(const double* a, unsigned s1, unsigned s2, double* b);
  static void sin(const double* a, unsigned total, double* b);
  static void cos(const double* a, unsigned total, double* b);
  static void pow(const double* a, unsigned total, double exponent, double* b);
  static void gelu(const double* a, unsigned total, double* b);
  static void asin(const double* a, unsigned total, double* b);
  static void asinh(const double* a, unsigned total, double* b);
  static void acos(const double* a, unsigned total, double* b);
  static void acosh(const double* a, unsigned total, double* b);
  static void atan(const double* a, unsigned total, double* b);
  static void atanh(const double* a, unsigned total, double* b);
#  ifdef TFCC_USE_SVML
  static void erf(const double* a, unsigned total, double* b);
#  endif
};

#endif

}  // namespace tfcc
