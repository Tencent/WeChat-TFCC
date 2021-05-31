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

#include "tfcc_basicinterface.h"

#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_types.h"

namespace tfcc {

template <class T>
Variable<T> BasicInterface<T>::slice(
    const Tensor<T>& a, size_t axis, unsigned start, unsigned end) {
  throw NotImplementedError();
}

template <class T>
void BasicInterface<T>::assignTo(const Tensor<T>& a, size_t axis, unsigned start, Variable<T>& b) {
  throw NotImplementedError();
}

template <class T>
Variable<T> BasicInterface<T>::transpose(const Tensor<T>& a, const std::vector<size_t>& perm) {
  throw NotImplementedError();
}

template <class T>
Variable<T> BasicInterface<T>::clip(const Tensor<T>& a, T minValue, T maxValue) {
  throw NotImplementedError();
}

template <class T>
Variable<T> BasicInterface<T>::concat(const std::vector<const Tensor<T>*>& values, size_t axis) {
  throw NotImplementedError();
}

template <class T>
Variable<T> BasicInterface<T>::where(
    const Tensor<uint8_t>& condition, const Tensor<T>& x, const Tensor<T>& y) {
  throw NotImplementedError();
}

template <class T>
Variable<T> BasicInterface<T>::where(const Tensor<uint8_t>& condition, T x, const Tensor<T>& y) {
  throw NotImplementedError();
}

template <class T>
Variable<T> BasicInterface<T>::where(const Tensor<uint8_t>& condition, const Tensor<T>& x, T y) {
  throw NotImplementedError();
}

template <class T>
Variable<T> BasicInterface<T>::abs(const Tensor<T>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> BasicInterface<T>::tril(const Tensor<T>& a, int64_t k) {
  throw NotImplementedError();
}

template <class T>
Variable<T> BasicInterface<T>::triu(const Tensor<T>& a, int64_t k) {
  throw NotImplementedError();
}

template <class T>
Variable<int64_t> BasicInterface<T>::argmax(const Tensor<T>& a, size_t axis) {
  throw NotImplementedError();
}

template <class T>
std::tuple<Variable<T>, Variable<uint32_t>> BasicInterface<T>::topK(
    const Tensor<T>& a, unsigned k) {
  throw NotImplementedError();
}

#define DEFINE_FUNC(type) template class BasicInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);
TFCC_FOR_COMPLEX_TYPES(DEFINE_FUNC);

}  // namespace tfcc
