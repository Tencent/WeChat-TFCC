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

#include "framework/tfcc_cudadeviceproperty.h"
#include "interfaces/tfcc_activationinterface.h"

namespace tfcc {

template <class T>
class CUDAActivationInterface : public ActivationInterface<T> {
  CUDADeviceProperty _property;

 public:
  explicit CUDAActivationInterface(const CUDADeviceProperty& property);
  ~CUDAActivationInterface();

  Variable<T> sigmoid(const Tensor<T>& a) override;
  Variable<T> tanh(const Tensor<T>& a) override;
  Variable<T> relu(const Tensor<T>& a) override;
  Variable<T> leakyRelu(const Tensor<T>& a, T alpha) override;
  Variable<T> softplus(const Tensor<T>& a) override;
  Variable<T> log(const Tensor<T>& a) override;
  Variable<T> rsqrt(const Tensor<T>& a) override;
  Variable<T> softmax(const Tensor<T>& a, size_t axis) override;
  Variable<T> sin(const Tensor<T>& a) override;
  Variable<T> cos(const Tensor<T>& a) override;
  Variable<T> pow(const Tensor<T>& a, T exponent) override;
  Variable<T> pow(const Tensor<T>& a, const Tensor<T>& exponent) override;
  Variable<T> pow(T a, const Tensor<T>& exponent) override;
  Variable<T> gelu(const Tensor<T>& a) override;
  Variable<T> geluAccurate(const Tensor<T>& a) override;
  Variable<T> erf(const Tensor<T>& a) override;
  Variable<T> asin(const Tensor<T>& a) override;
  Variable<T> asinh(const Tensor<T>& a) override;
  Variable<T> acos(const Tensor<T>& a) override;
  Variable<T> acosh(const Tensor<T>& a) override;
  Variable<T> atan(const Tensor<T>& a) override;
  Variable<T> atanh(const Tensor<T>& a) override;
  Variable<T> sign(const Tensor<T>& a) override;
};

}  // namespace tfcc
