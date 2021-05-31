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

#include "tfcc_activationinterface.h"

#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_types.h"

namespace tfcc {

template <class T>
Variable<T> ActivationInterface<T>::sigmoid(const Tensor<T>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ActivationInterface<T>::tanh(const Tensor<T>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ActivationInterface<T>::relu(const Tensor<T>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ActivationInterface<T>::leakyRelu(const Tensor<T>& a, T alpha) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ActivationInterface<T>::softplus(const Tensor<T>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ActivationInterface<T>::log(const Tensor<T>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ActivationInterface<T>::rsqrt(const Tensor<T>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ActivationInterface<T>::softmax(const Tensor<T>& a, size_t axis) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ActivationInterface<T>::sin(const Tensor<T>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ActivationInterface<T>::cos(const Tensor<T>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ActivationInterface<T>::pow(const Tensor<T>& a, T exponent) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ActivationInterface<T>::pow(const Tensor<T>& a, const Tensor<T>& exponent) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ActivationInterface<T>::pow(T a, const Tensor<T>& exponent) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ActivationInterface<T>::gelu(const Tensor<T>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ActivationInterface<T>::geluAccurate(const Tensor<T>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ActivationInterface<T>::erf(const Tensor<T>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ActivationInterface<T>::asin(const Tensor<T>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ActivationInterface<T>::asinh(const Tensor<T>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ActivationInterface<T>::acos(const Tensor<T>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ActivationInterface<T>::acosh(const Tensor<T>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ActivationInterface<T>::atan(const Tensor<T>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ActivationInterface<T>::atanh(const Tensor<T>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ActivationInterface<T>::sign(const Tensor<T>& a) {
  throw NotImplementedError();
}

#define DEFINE_FUNC(type) template class ActivationInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);
TFCC_FOR_COMPLEX_TYPES(DEFINE_FUNC);

}  // namespace tfcc
