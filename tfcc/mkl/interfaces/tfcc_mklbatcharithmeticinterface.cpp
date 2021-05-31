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

#include "tfcc_mklbatcharithmeticinterface.h"

#include <omp.h>
#include <cassert>
#include <cstring>
#include <memory>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_mklsession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "framework/tfcc_view.h"
#include "interfaces/tfcc_mklinterfacehelper.h"

namespace tfcc {

template <class T>
static void _mkl_batch_add(const T** values, unsigned batch, unsigned length, T* result) {
#pragma omp parallel
  {
    unsigned threadCount = omp_get_num_threads();
    unsigned threadId = omp_get_thread_num();
    unsigned chunkSize = (length + threadCount - 1) / threadCount;
    unsigned start = chunkSize * threadId;
    if (start < length) {
      chunkSize = std::min(chunkSize, length - start);
      memcpy(result + start, values[0] + start, sizeof(T) * chunkSize);
      for (unsigned i = 1; i < batch; ++i) {
        const T* v = values[i] + start;
        T* r = result + start;
        for (unsigned j = 0; j < chunkSize; ++j) {
          r[j] += v[j];
        }
      }
    }
  }
}

template <class T>
static void _mkl_batch_mul(const T** values, unsigned batch, unsigned length, T* result) {
#pragma omp parallel
  {
    unsigned threadCount = omp_get_num_threads();
    unsigned threadId = omp_get_thread_num();
    unsigned chunkSize = (length + threadCount - 1) / threadCount;
    unsigned start = chunkSize * threadId;
    if (start < length) {
      chunkSize = std::min(chunkSize, length - start);
      memcpy(result + start, values[0] + start, sizeof(T) * chunkSize);
      for (unsigned i = 1; i < batch; ++i) {
        const T* v = values[i] + start;
        T* r = result + start;
        for (unsigned j = 0; j < chunkSize; ++j) {
          r[j] *= v[j];
        }
      }
    }
  }
}

template <class T>
Variable<T> MKLBatchArithmeticInterface<T>::add(const std::vector<const Tensor<T>*>& values) {
  Variable<T> result(values[0]->shape());
  std::shared_ptr<std::vector<const T*>> vs(new std::vector<const T*>(values.size()));
  for (size_t i = 0; i < values.size(); ++i) {
    vs->data()[i] = values[i]->data();
    assert(vs->data()[i] != 0);
  }
  mkl_async_wrapper(
      "batch_add",
      [](std::shared_ptr<std::vector<const T*>> values, unsigned batch, unsigned length,
         T* result) { _mkl_batch_add(values->data(), batch, length, result); },
      vs, values.size(), result.size(), result.data());
  return result;
}

template <class T>
Variable<T> MKLBatchArithmeticInterface<T>::mul(const std::vector<const Tensor<T>*>& values) {
  Variable<T> result(values[0]->shape());
  std::shared_ptr<std::vector<const T*>> vs(new std::vector<const T*>(values.size()));
  for (size_t i = 0; i < values.size(); ++i) {
    vs->data()[i] = values[i]->data();
    assert(vs->data()[i] != 0);
  }
  mkl_async_wrapper(
      "batch_mul",
      [](std::shared_ptr<std::vector<const T*>> values, unsigned batch, unsigned length,
         T* result) { _mkl_batch_mul(values->data(), batch, length, result); },
      vs, values.size(), result.size(), result.data());
  return result;
}

#define DEFINE_FUNC(type) template class MKLBatchArithmeticInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
