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

#include "framework/tfcc_types.h"

namespace tfcc {

template <class T>
class _MKLArithmeticKernel {
 public:
  static void batchAdd(const T* a, const T* b, T* c, unsigned total) {
#pragma omp parallel for
    for (unsigned i = 0; i < total; ++i) c[i] = a[i] + b[i];
  }

  static void batchSub(const T* a, const T* b, T* c, unsigned total) {
#pragma omp parallel for
    for (unsigned i = 0; i < total; ++i) c[i] = a[i] - b[i];
  }

  static void batchMul(const T* a, const T* b, T* c, unsigned total) {
#pragma omp parallel for
    for (unsigned i = 0; i < total; ++i) c[i] = a[i] * b[i];
  }

  static void batchDiv(const T* a, const T* b, T* c, unsigned total) {
#pragma omp parallel for
    for (unsigned i = 0; i < total; ++i) c[i] = a[i] / b[i];
  }
};

template <class T>
class _MKLArithmeticKernel<Complex<T>> {
 public:
  static void batchMul(const Complex<T>* a, const Complex<T>* b, Complex<T>* c, unsigned total) {
#pragma omp parallel for
    for (unsigned i = 0; i < total; ++i) {
      c[i].real = a[i].real * b[i].real - a[i].imag * b[i].imag;
      c[i].imag = a[i].real * b[i].imag + a[i].imag * b[i].real;
    }
  }
};

}  // namespace tfcc
