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

#include "tfcc_mklgatherinterface.h"

#include <omp.h>
#include <algorithm>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_mklsession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "interfaces/tfcc_mklinterfacehelper.h"

namespace tfcc {

template <class T, class IDX>
static void _mkl_gather(
    const T* params, unsigned batch, unsigned chunkSize, const IDX* indices, unsigned length,
    T* result) {
#pragma omp parallel for
  for (unsigned i = 0; i < length; ++i) {
    T* r = result + chunkSize * i;
    if (indices[i] < 0 || indices[i] >= static_cast<IDX>(batch)) {
      for (unsigned j = 0; j < chunkSize; ++j) {
        r[j] = static_cast<T>(0);
      }
    } else {
      const T* p = params + chunkSize * indices[i];
      for (unsigned j = 0; j < chunkSize; ++j) {
        r[j] = p[j];
      }
    }
  }
}

template <class T, class IDX>
static inline Variable<T> _gather_helper(const Tensor<T>& params, const Tensor<IDX>& indices) {
  std::vector<unsigned> shape;
  for (size_t i = 0; i < indices.shape().size(); ++i) {
    shape.push_back(indices.shape(i));
  }
  for (size_t i = 1; i < params.shape().size(); ++i) {
    shape.push_back(params.shape(i));
  }

  Variable<T> result(shape);
  mkl_async_wrapper(
      "gather",
      [](const T* params, unsigned batch, unsigned chunkSize, const IDX* indices, unsigned length,
         T* result) { _mkl_gather(params, batch, chunkSize, indices, length, result); },
      params.data(), params.shape(0), params.size() / params.shape(0), indices.data(),
      indices.size(), result.data());
  return result;
}

template <class T>
Variable<T> MKLGatherInterface<T>::gather(
    const Tensor<T>& params, const Tensor<uint32_t>& indices) {
  return _gather_helper(params, indices);
}

template <class T>
Variable<T> MKLGatherInterface<T>::gather(const Tensor<T>& params, const Tensor<int32_t>& indices) {
  return _gather_helper(params, indices);
}

template <class T>
Variable<T> MKLGatherInterface<T>::gather(
    const Tensor<T>& params, const Tensor<uint64_t>& indices) {
  return _gather_helper(params, indices);
}

template <class T>
Variable<T> MKLGatherInterface<T>::gather(const Tensor<T>& params, const Tensor<int64_t>& indices) {
  return _gather_helper(params, indices);
}

#define DEFINE_FUNC(type) template class MKLGatherInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
