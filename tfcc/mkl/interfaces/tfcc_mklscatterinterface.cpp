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

#include "tfcc_mklscatterinterface.h"

#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_mklsession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "framework/tfcc_view.h"
#include "interfaces/tfcc_mklinterfacehelper.h"
#include "kernel/tfcc_mklarithmetickernel.avx256.h"
#include "kernel/tfcc_mklarithmetickernel.avx512.h"
#include "kernel/tfcc_mklarithmetickernel.hpp"

namespace tfcc {

template <class T, class IDX>
static inline void _mkl_scatter_nd(
    const IDX* indices, const T* updates, unsigned count, std::vector<unsigned> offsets,
    unsigned batch, unsigned chunk, T* result) {
#pragma omp parallel for
  for (unsigned i = 0; i < batch * chunk; ++i) {
    result[i] = static_cast<T>(0);
  }

#pragma omp parallel for
  for (unsigned i = 0; i < count; ++i) {
    unsigned pos = 0;
    for (unsigned j = 0; j < offsets.size(); ++j) {
      pos += indices[i * offsets.size() + j] * offsets[j];
    }
    for (unsigned j = 0; j < chunk; ++j) {
      result[pos + j] = updates[i * chunk + j];
    }
  }
}

template <class T, class IDX>
static inline void _mkl_scatter_nd(
    const T* data, const IDX* indices, const T* updates, unsigned count,
    std::vector<unsigned> offsets, unsigned batch, unsigned chunk, T* result) {
#pragma omp parallel for
  for (unsigned i = 0; i < batch * chunk; ++i) {
    result[i] = data[i];
  }

#pragma omp parallel for
  for (unsigned i = 0; i < count; ++i) {
    unsigned pos = 0;
    for (unsigned j = 0; j < offsets.size(); ++j) {
      pos += indices[i * offsets.size() + j] * offsets[j];
    }
    for (unsigned j = 0; j < chunk; ++j) {
      result[pos + j] = updates[i * chunk + j];
    }
  }
}

template <class T, class IDX>
static inline Variable<T> _scatter_nd_helper(
    const Tensor<IDX>& indices, const Tensor<T>& updates, const Shape& shape) {
  unsigned dims = indices.shape(indices.shape().size() - 1);
  unsigned offset = 1;
  for (unsigned i = dims; i < shape.size(); ++i) {
    offset *= shape[shape.size() + dims - 1 - i];
  }
  unsigned chunk = offset;
  std::vector<unsigned> offsets(dims);
  for (unsigned i = 0; i < dims; ++i) {
    offsets[dims - 1 - i] = offset;
    offset *= shape[dims - 1 - i];
  }

  Variable<T> result(shape);

  mkl_async_wrapper(
      "scatter_nd",
      [](const IDX* indices, const T* updates, unsigned count, std::vector<unsigned> offsets,
         unsigned batch, unsigned chunk, T* result) {
        _mkl_scatter_nd(indices, updates, count, std::move(offsets), batch, chunk, result);
      },
      indices.data(), updates.data(), indices.size() / indices.shape(indices.shape().size() - 1),
      offsets, shape.area() / chunk, chunk, result.data());
  return result;
}

template <class T, class IDX>
static inline Variable<T> _scatter_nd_helper(
    const Tensor<T>& data, const Tensor<IDX>& indices, const Tensor<T>& updates) {
  unsigned dims = indices.shape(indices.shape().size() - 1);
  unsigned offset = 1;
  for (unsigned i = dims; i < data.shape().size(); ++i) {
    offset *= data.shape(data.shape().size() + dims - 1 - i);
  }
  unsigned chunk = offset;
  std::vector<unsigned> offsets(dims);
  for (unsigned i = 0; i < dims; ++i) {
    offsets[dims - 1 - i] = offset;
    offset *= data.shape(dims - 1 - i);
  }

  Variable<T> result(data.shape());

  mkl_async_wrapper(
      "scatter_nd",
      [](const T* data, const IDX* indices, const T* updates, unsigned count,
         std::vector<unsigned> offsets, unsigned batch, unsigned chunk, T* result) {
        _mkl_scatter_nd(data, indices, updates, count, std::move(offsets), batch, chunk, result);
      },
      data.data(), indices.data(), updates.data(),
      indices.size() / indices.shape(indices.shape().size() - 1), offsets, data.size() / chunk,
      chunk, result.data());
  return result;
}

template <class T>
Variable<T> MKLScatterInterface<T>::scatterND(
    const Tensor<uint32_t>& indices, const Tensor<T>& updates, const Shape& shape) {
  return _scatter_nd_helper(indices, updates, shape);
}

template <class T>
Variable<T> MKLScatterInterface<T>::scatterND(
    const Tensor<int32_t>& indices, const Tensor<T>& updates, const Shape& shape) {
  return _scatter_nd_helper(indices, updates, shape);
}

template <class T>
Variable<T> MKLScatterInterface<T>::scatterND(
    const Tensor<uint64_t>& indices, const Tensor<T>& updates, const Shape& shape) {
  return _scatter_nd_helper(indices, updates, shape);
}

template <class T>
Variable<T> MKLScatterInterface<T>::scatterND(
    const Tensor<int64_t>& indices, const Tensor<T>& updates, const Shape& shape) {
  return _scatter_nd_helper(indices, updates, shape);
}

template <class T>
Variable<T> MKLScatterInterface<T>::scatterND(
    const Tensor<T>& data, const Tensor<uint32_t>& indices, const Tensor<T>& updates) {
  return _scatter_nd_helper(data, indices, updates);
}

template <class T>
Variable<T> MKLScatterInterface<T>::scatterND(
    const Tensor<T>& data, const Tensor<int32_t>& indices, const Tensor<T>& updates) {
  return _scatter_nd_helper(data, indices, updates);
}

template <class T>
Variable<T> MKLScatterInterface<T>::scatterND(
    const Tensor<T>& data, const Tensor<uint64_t>& indices, const Tensor<T>& updates) {
  return _scatter_nd_helper(data, indices, updates);
}

template <class T>
Variable<T> MKLScatterInterface<T>::scatterND(
    const Tensor<T>& data, const Tensor<int64_t>& indices, const Tensor<T>& updates) {
  return _scatter_nd_helper(data, indices, updates);
}

#define DEFINE_FUNC(type) template class MKLScatterInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
