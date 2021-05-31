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

#include "tfcc_scatterinterface.h"

#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_types.h"

namespace tfcc {

template <class T>
Variable<T> ScatterInterface<T>::scatterND(
    const Tensor<uint32_t>& indices, const Tensor<T>& updates, const Shape& shape) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ScatterInterface<T>::scatterND(
    const Tensor<int32_t>& indices, const Tensor<T>& updates, const Shape& shape) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ScatterInterface<T>::scatterND(
    const Tensor<uint64_t>& indices, const Tensor<T>& updates, const Shape& shape) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ScatterInterface<T>::scatterND(
    const Tensor<int64_t>& indices, const Tensor<T>& updates, const Shape& shape) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ScatterInterface<T>::scatterND(
    const Tensor<T>& data, const Tensor<uint32_t>& indices, const Tensor<T>& updates) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ScatterInterface<T>::scatterND(
    const Tensor<T>& data, const Tensor<int32_t>& indices, const Tensor<T>& updates) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ScatterInterface<T>::scatterND(
    const Tensor<T>& data, const Tensor<uint64_t>& indices, const Tensor<T>& updates) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ScatterInterface<T>::scatterND(
    const Tensor<T>& data, const Tensor<int64_t>& indices, const Tensor<T>& updates) {
  throw NotImplementedError();
}

#define DEFINE_FUNC(type) template class ScatterInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);
TFCC_FOR_COMPLEX_TYPES(DEFINE_FUNC);

}  // namespace tfcc
