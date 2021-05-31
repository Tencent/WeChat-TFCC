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
#include "interfaces/tfcc_transformationinterface.h"

namespace tfcc {

template <class T>
class CUDATransformationInterface : public TransformationInterface<T> {
  CUDADeviceProperty _property;

 public:
  explicit CUDATransformationInterface(const CUDADeviceProperty& property);
  ~CUDATransformationInterface();

  Variable<T> transform(const Tensor<T>& a, T alpha, T beta) override;
  Variable<T> transform2(const Tensor<T>& a, T alpha, T beta) override;
  Variable<T> transform3(const Tensor<T>& a, T alpha, T beta) override;
  Variable<T> transform4(const Tensor<T>& a, T alpha, T beta) override;
  Variable<T> transform5(const Tensor<T>& a, T alpha, T beta) override;
  Variable<T> transform6(const Tensor<T>& a, T alpha, T beta) override;
};

}  // namespace tfcc
