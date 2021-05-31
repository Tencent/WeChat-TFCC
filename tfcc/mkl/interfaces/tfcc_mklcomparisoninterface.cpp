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

#include "tfcc_mklcomparisoninterface.h"
#include "tfcc_mklinterfacehelper.h"

#include <omp.h>

#include "framework/tfcc_mklsession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "interfaces/tfcc_mklinterfacehelper.h"

namespace tfcc {

// mkl function
template <class T>
static void _mkl_equal(const T* a, unsigned total, T b, uint8_t* result) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    result[i] = a[i] == b ? 1 : 0;
  }
}

template <class T>
static void _mkl_unequal(const T* a, unsigned total, T b, uint8_t* result) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    result[i] = a[i] != b ? 1 : 0;
  }
}

template <class T>
static void _mkl_greater(const T* a, unsigned total, T b, uint8_t* result) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    result[i] = a[i] > b ? 1 : 0;
  }
}

template <class T>
static void _mkl_greater_equal(const T* a, unsigned total, T b, uint8_t* result) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    result[i] = a[i] >= b ? 1 : 0;
  }
}

template <class T>
static void _mkl_less(const T* a, unsigned total, T b, uint8_t* result) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    result[i] = a[i] < b ? 1 : 0;
  }
}

template <class T>
static void _mkl_less_equal(const T* a, unsigned total, T b, uint8_t* result) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    result[i] = a[i] <= b ? 1 : 0;
  }
}

// class function
template <class T>
Variable<uint8_t> MKLComparisonInterface<T>::equal(const Tensor<T>& a, T b) {
  Variable<uint8_t> result(a.shape());
  mkl_async_wrapper(
      "equal",
      [](const T* a, unsigned total, T b, uint8_t* result) { _mkl_equal(a, total, b, result); },
      a.data(), a.size(), b, result.data());
  return result;
}

template <class T>
Variable<uint8_t> MKLComparisonInterface<T>::unequal(const Tensor<T>& a, T b) {
  Variable<uint8_t> result(a.shape());
  mkl_async_wrapper(
      "unequal",
      [](const T* a, unsigned total, T b, uint8_t* result) { _mkl_unequal(a, total, b, result); },
      a.data(), a.size(), b, result.data());
  return result;
}

template <class T>
Variable<uint8_t> MKLComparisonInterface<T>::greater(const Tensor<T>& a, T b) {
  Variable<uint8_t> result(a.shape());
  mkl_async_wrapper(
      "greater",
      [](const T* a, unsigned total, T b, uint8_t* result) { _mkl_greater(a, total, b, result); },
      a.data(), a.size(), b, result.data());
  return result;
}

template <class T>
Variable<uint8_t> MKLComparisonInterface<T>::greaterEqual(const Tensor<T>& a, T b) {
  Variable<uint8_t> result(a.shape());
  mkl_async_wrapper(
      "greater_equal",
      [](const T* a, unsigned total, T b, uint8_t* result) {
        _mkl_greater_equal(a, total, b, result);
      },
      a.data(), a.size(), b, result.data());
  return result;
}

template <class T>
Variable<uint8_t> MKLComparisonInterface<T>::less(const Tensor<T>& a, T b) {
  Variable<uint8_t> result(a.shape());
  mkl_async_wrapper(
      "less",
      [](const T* a, unsigned total, T b, uint8_t* result) { _mkl_less(a, total, b, result); },
      a.data(), a.size(), b, result.data());
  return result;
}

template <class T>
Variable<uint8_t> MKLComparisonInterface<T>::lessEqual(const Tensor<T>& a, T b) {
  Variable<uint8_t> result(a.shape());
  mkl_async_wrapper(
      "less_equal",
      [](const T* a, unsigned total, T b, uint8_t* result) {
        _mkl_less_equal(a, total, b, result);
      },
      a.data(), a.size(), b, result.data());
  return result;
}

#define DEFINE_FUNC(type) template class MKLComparisonInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
