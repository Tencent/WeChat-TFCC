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

#include "tfcc_mklminmaxinterface.h"

#include <algorithm>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_mklsession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "framework/tfcc_view.h"
#include "interfaces/tfcc_mklbroadcasthelper.h"
#include "interfaces/tfcc_mklinterfacehelper.h"
#include "kernel/tfcc_mklarithmetickernel.avx256.h"
#include "kernel/tfcc_mklarithmetickernel.avx512.h"
#include "kernel/tfcc_mklarithmetickernel.hpp"

namespace tfcc {

template <class T>
static inline void _mkl_min(const T* a, const T* b, T* c, unsigned total) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    c[i] = std::min(a[i], b[i]);
  }
}

template <class T>
static inline void _mkl_min(const T* a, T b, T* c, unsigned total) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    c[i] = std::min(a[i], b);
  }
}

template <class T>
static inline void _mkl_max(const T* a, const T* b, T* c, unsigned total) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    c[i] = std::max(a[i], b[i]);
  }
}

template <class T>
static inline void _mkl_max(const T* a, T b, T* c, unsigned total) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    c[i] = std::max(a[i], b);
  }
}

template <class T>
Variable<T> MKLMinMaxInterface<T>::min(const Tensor<T>& a, const Tensor<T>& b) {
  return _mkl_process_broadcast_op(
      a, b, [](T a, T b) { return std::min(a, b); },
      [](const T* a, const T* b, T* c, unsigned total) { return _mkl_min(a, b, c, total); }, "min");
}

template <class T>
Variable<T> MKLMinMaxInterface<T>::min(const Tensor<T>& a, T b) {
  Variable<T> result(a.shape());

  mkl_async_wrapper(
      "min", [](const T* a, T b, T* c, unsigned total) { _mkl_min(a, b, c, total); }, a.data(), b,
      result.data(), result.size());

  return result;
}

template <class T>
Variable<T> MKLMinMaxInterface<T>::max(const Tensor<T>& a, const Tensor<T>& b) {
  return _mkl_process_broadcast_op(
      a, b, [](T a, T b) { return std::max(a, b); },
      [](const T* a, const T* b, T* c, unsigned total) { return _mkl_max(a, b, c, total); }, "max");
}

template <class T>
Variable<T> MKLMinMaxInterface<T>::max(const Tensor<T>& a, T b) {
  Variable<T> result(a.shape());

  mkl_async_wrapper(
      "max", [](const T* a, T b, T* c, unsigned total) { _mkl_max(a, b, c, total); }, a.data(), b,
      result.data(), result.size());

  return result;
}

#define DEFINE_FUNC(type) template class MKLMinMaxInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
