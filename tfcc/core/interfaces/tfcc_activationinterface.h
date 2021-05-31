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

#pragma once

#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"

namespace tfcc {

template <class T>
class ActivationInterface {
 public:
  ActivationInterface() {}
  ActivationInterface(const ActivationInterface&) = delete;
  virtual ~ActivationInterface() {}

  ActivationInterface& operator=(const ActivationInterface&) = delete;

  virtual Variable<T> sigmoid(const Tensor<T>& a);
  virtual Variable<T> tanh(const Tensor<T>& a);
  virtual Variable<T> relu(const Tensor<T>& a);
  virtual Variable<T> leakyRelu(const Tensor<T>& a, T alpha);
  virtual Variable<T> softplus(const Tensor<T>& a);
  virtual Variable<T> log(const Tensor<T>& a);
  virtual Variable<T> rsqrt(const Tensor<T>& a);
  virtual Variable<T> softmax(const Tensor<T>& a, size_t axis);
  virtual Variable<T> sin(const Tensor<T>& a);
  virtual Variable<T> cos(const Tensor<T>& a);
  virtual Variable<T> pow(const Tensor<T>& a, T exponent);
  virtual Variable<T> pow(const Tensor<T>& a, const Tensor<T>& exponent);
  virtual Variable<T> pow(T a, const Tensor<T>& exponent);
  virtual Variable<T> gelu(const Tensor<T>& a);
  virtual Variable<T> geluAccurate(const Tensor<T>& a);
  virtual Variable<T> erf(const Tensor<T>& a);
  virtual Variable<T> asin(const Tensor<T>& a);
  virtual Variable<T> asinh(const Tensor<T>& a);
  virtual Variable<T> acos(const Tensor<T>& a);
  virtual Variable<T> acosh(const Tensor<T>& a);
  virtual Variable<T> atan(const Tensor<T>& a);
  virtual Variable<T> atanh(const Tensor<T>& a);
  virtual Variable<T> sign(const Tensor<T>& a);
};

}  // namespace tfcc
