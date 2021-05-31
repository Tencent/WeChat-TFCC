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

#include "tfcc_mklsegmentinterface.h"

#include <atomic>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_mklsession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "interfaces/tfcc_mklinterfacehelper.h"

namespace tfcc {

template <class T>
static void _atomic_add(std::atomic<T>* a, T b) {
  (*a) += b;
}

static void _atomic_add(std::atomic<float>* a, float b) {
  float old, newValue;
  do {
    old = a->load();
    newValue = old + b;
  } while (!a->compare_exchange_weak(old, newValue));
}

static void _atomic_add(std::atomic<double>* a, double b) {
  double old, newValue;
  do {
    old = a->load();
    newValue = old + b;
  } while (!a->compare_exchange_weak(old, newValue));
}

template <class T, class Func>
static void _mkl_unsorted_segment_op(
    const T* a, unsigned batch, unsigned k, const int* ids, unsigned num, T* b, Func op) {
  unsigned total = k * num;
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    b[i] = static_cast<T>(0);
  }

#pragma omp parallel for
  for (unsigned i = 0; i < batch; ++i) {
    if (ids[i] < 0) {
      continue;
    }
    T* rb = b + k * ids[i];
    const T* ra = a + k * i;
    for (unsigned j = 0; j < k; ++j) {
      op(&rb[j], ra[j]);
    }
  }
}

template <class T>
Variable<T> MKLSegmentInterface<T>::unsortedSegmentSum(
    const Tensor<T>& a, const Tensor<int>& ids, unsigned num) {
  std::vector<unsigned> s = a.shape().toVector();
  s[0] = num;
  Variable<T> result(std::move(s));

  mkl_async_wrapper(
      "unsorted_segment_op",
      [](const T* a, unsigned batch, unsigned k, const int* ids, unsigned num, T* b) {
        auto addOp = [](T* a, T b) {
          static_assert(sizeof(std::atomic<T>) == sizeof(T), "invalid T");
          std::atomic<T>* ra = reinterpret_cast<std::atomic<T>*>(a);
          _atomic_add(ra, b);
        };
        _mkl_unsorted_segment_op(a, batch, k, ids, num, b, addOp);
      },
      a.data(), a.shape(0), a.size() / a.shape(0), ids.data(), num, result.data());
  return result;
}

#define DEFINE_FUNC(type) template class MKLSegmentInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
