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

#include "tfcc_data.h"

#include "exceptions/tfcc_invalidargumenterror.h"
#include "framework/tfcc_types.h"
#include "interfaces/tfcc_datainterface.h"
#include "operations/tfcc_operation.h"

namespace tfcc {
namespace data {

template <class T>
void set(Variable<T>& a, const T* data) {
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  interface.getDataInterface().set(a, data);
}

template <class T>
void set(Variable<T>& a, const std::vector<T>& data) {
  if (a.size() != data.size()) {
    throw InvalidArgumentError("tensor's size not equal to data's size");
  }
  set(a, data.data());
}

template <class T>
void set(Variable<T>& a, std::vector<T>&& data) {
  if (a.size() != data.size()) {
    throw InvalidArgumentError("tensor's size not equal to data's size");
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  interface.getDataInterface().set(a, std::move(data));
}

template <class T>
tfcc::Variable<T> set(const std::vector<T>& data, Shape shape) {
  if (data.size() != shape.area()) {
    throw InvalidArgumentError("shape's area not equal to data's size");
  }
  tfcc::Variable<T> result(shape);
  set(result, data.data());
  return result;
}

template <class T>
void get(const Tensor<T>& a, T* data) {
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  interface.getDataInterface().get(a, data);
}

template <class T>
std::vector<T> get(const Tensor<T>& a) {
  std::vector<T> data;
  data.resize(a.size());
  get(a, data.data());
  return data;
}

template <class T>
void zeros(Variable<T>& a) {
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  interface.getDataInterface().zeros(a);
}

template <class T>
void ones(Variable<T>& a) {
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  interface.getDataInterface().ones(a);
}

template <class T>
Variable<T> copy(const Tensor<T>& a) {
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getDataInterface().copy(a);
}

template <class SRC, class DST>
static inline Variable<DST> _cast_helper(const Tensor<SRC>& a, DST) {
  Interface<DST>& interface = Operation<DST>::getCurrentInterface();
  return interface.getDataInterface().cast(a);
}

template <class T>
Variable<T> cast(const Tensor<float>& a) {
  return _cast_helper(a, T());
}

template <class T>
Variable<T> cast(const Tensor<double>& a) {
  return _cast_helper(a, T());
}

template <class T>
Variable<T> cast(const Tensor<int8_t>& a) {
  return _cast_helper(a, T());
}

template <class T>
Variable<T> cast(const Tensor<uint8_t>& a) {
  return _cast_helper(a, T());
}

template <class T>
Variable<T> cast(const Tensor<int16_t>& a) {
  return _cast_helper(a, T());
}

template <class T>
Variable<T> cast(const Tensor<uint16_t>& a) {
  return _cast_helper(a, T());
}

template <class T>
Variable<T> cast(const Tensor<int32_t>& a) {
  return _cast_helper(a, T());
}

template <class T>
Variable<T> cast(const Tensor<uint32_t>& a) {
  return _cast_helper(a, T());
}

template <class T>
Variable<T> cast(const Tensor<int64_t>& a) {
  return _cast_helper(a, T());
}

template <class T>
Variable<T> cast(const Tensor<uint64_t>& a) {
  return _cast_helper(a, T());
}

template <class T>
Variable<uint8_t> cast_to_boolean(const Tensor<T>& a) {
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getDataInterface().cast_to_boolean(a);
}

#define DEFINE_FUNC(type)                                                        \
  template void set(Variable<type>& a, const type* data);                        \
  template void set(Variable<type>& a, const std::vector<type>& data);           \
  template void set(Variable<type>& a, std::vector<type>&& data);                \
  template tfcc::Variable<type> set(const std::vector<type>& data, Shape shape); \
  template void get(const Tensor<type>& a, type* data);                          \
  template std::vector<type> get(const Tensor<type>& a);                         \
  template void zeros(Variable<type>& a);                                        \
  template void ones(Variable<type>& a);                                         \
  template Variable<type> copy(const Tensor<type>& a);                           \
  template Variable<type> cast(const Tensor<float>& a);                          \
  template Variable<type> cast(const Tensor<double>& a);                         \
  template Variable<type> cast(const Tensor<int8_t>& a);                         \
  template Variable<type> cast(const Tensor<uint8_t>& a);                        \
  template Variable<type> cast(const Tensor<int16_t>& a);                        \
  template Variable<type> cast(const Tensor<uint16_t>& a);                       \
  template Variable<type> cast(const Tensor<int32_t>& a);                        \
  template Variable<type> cast(const Tensor<uint32_t>& a);                       \
  template Variable<type> cast(const Tensor<int64_t>& a);                        \
  template Variable<type> cast(const Tensor<uint64_t>& a);                       \
  template Variable<uint8_t> cast_to_boolean(const Tensor<type>& a)

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

#define COMPLEX_DEFINE_FUNC(type)                                                \
  template void set(Variable<type>& a, const type* data);                        \
  template void set(Variable<type>& a, const std::vector<type>& data);           \
  template void set(Variable<type>& a, std::vector<type>&& data);                \
  template tfcc::Variable<type> set(const std::vector<type>& data, Shape shape); \
  template void get(const Tensor<type>& a, type* data);                          \
  template std::vector<type> get(const Tensor<type>& a);                         \
  template void zeros(Variable<type>& a);                                        \
  template void ones(Variable<type>& a);

TFCC_FOR_COMPLEX_TYPES(COMPLEX_DEFINE_FUNC);

}  // namespace data
}  // namespace tfcc
