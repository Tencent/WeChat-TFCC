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
class _MKLTransformationKernel {
 public:
  static void transform(const T* a, unsigned total, T alpha, T beta, T* b) {
#pragma omp parallel for
    for (unsigned i = 0; i < total; ++i) {
      auto v = alpha * a[i] + beta;
      b[i] = v;
    }
  }

  static void transform2(const T* a, unsigned total, T alpha, T beta, T* b) {
#pragma omp parallel for
    for (unsigned i = 0; i < total; ++i) {
      auto v = a[i] / alpha + beta;
      b[i] = v;
    }
  }

  static void transform3(const T* a, unsigned total, T alpha, T beta, T* b) {
#pragma omp parallel for
    for (unsigned i = 0; i < total; ++i) {
      auto v = alpha / a[i] + beta;
      b[i] = v;
    }
  }

  static void transform4(const T* a, unsigned total, T alpha, T beta, T* b) {
#pragma omp parallel for
    for (unsigned i = 0; i < total; ++i) {
      auto v = beta - a[i] * alpha;
      b[i] = v;
    }
  }

  static void transform5(const T* a, unsigned total, T alpha, T beta, T* b) {
#pragma omp parallel for
    for (unsigned i = 0; i < total; ++i) {
      auto v = beta - a[i] / alpha;
      b[i] = v;
    }
  }

  // transform6
  static void transform6(const T* a, unsigned total, T alpha, T beta, T* b) {
#pragma omp parallel for
    for (unsigned i = 0; i < total; ++i) {
      auto v = beta - alpha / a[i];
      b[i] = v;
    }
  }
};

}  // namespace tfcc
