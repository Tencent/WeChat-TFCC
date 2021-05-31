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

#include "tfcc_datainterface.h"

#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_types.h"

namespace tfcc {

template <class T>
void DataInterface<T>::set(T* dst, const T* data, size_t len) {
  throw NotImplementedError();
}

template <class T>
void DataInterface<T>::set(Variable<T>& a, std::vector<T>&& data) {
  throw NotImplementedError();
}

template <class T>
void DataInterface<T>::set(Variable<T>& a, const T* data) {
  throw NotImplementedError();
}

template <class T>
void DataInterface<T>::get(const Tensor<T>& a, T* data) {
  throw NotImplementedError();
}

template <class T>
void DataInterface<T>::zeros(Variable<T>& a) {
  throw NotImplementedError();
}

template <class T>
void DataInterface<T>::ones(Variable<T>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> DataInterface<T>::copy(const Tensor<T>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> DataInterface<T>::cast(const Tensor<float>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> DataInterface<T>::cast(const Tensor<double>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> DataInterface<T>::cast(const Tensor<int8_t>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> DataInterface<T>::cast(const Tensor<uint8_t>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> DataInterface<T>::cast(const Tensor<int16_t>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> DataInterface<T>::cast(const Tensor<uint16_t>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> DataInterface<T>::cast(const Tensor<int32_t>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> DataInterface<T>::cast(const Tensor<uint32_t>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> DataInterface<T>::cast(const Tensor<int64_t>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> DataInterface<T>::cast(const Tensor<uint64_t>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<uint8_t> DataInterface<T>::cast_to_boolean(const Tensor<T>& a) {
  throw NotImplementedError();
}

#define DEFINE_FUNC(type) template class DataInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);
TFCC_FOR_COMPLEX_TYPES(DEFINE_FUNC);

}  // namespace tfcc
