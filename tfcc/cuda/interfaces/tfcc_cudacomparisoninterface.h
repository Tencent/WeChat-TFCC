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
#include "interfaces/tfcc_comparisoninterface.h"

namespace tfcc {

template <class T>
class CUDAComparisonInterface : public ComparisonInterface<T> {
  CUDADeviceProperty _property;

 public:
  explicit CUDAComparisonInterface(const CUDADeviceProperty& property);
  ~CUDAComparisonInterface();

  Variable<uint8_t> equal(const Tensor<T>& a, T b) override;
  Variable<uint8_t> unequal(const Tensor<T>& a, T b) override;
  Variable<uint8_t> greater(const Tensor<T>& a, T b) override;
  Variable<uint8_t> greaterEqual(const Tensor<T>& a, T b) override;
  Variable<uint8_t> less(const Tensor<T>& a, T b) override;
  Variable<uint8_t> lessEqual(const Tensor<T>& a, T b) override;
};

}  // namespace tfcc
