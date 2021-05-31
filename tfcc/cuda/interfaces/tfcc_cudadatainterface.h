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
#include "interfaces/tfcc_datainterface.h"

namespace tfcc {

template <class T>
class CUDADataInterface : public DataInterface<T> {
  CUDADeviceProperty _property;

 public:
  explicit CUDADataInterface(const CUDADeviceProperty& property);
  ~CUDADataInterface();

  void set(T* dst, const T* data, size_t len) override;

  void set(Variable<T>& a, const T* data) override;
  void set(Variable<T>& a, std::vector<T>&& data) override;
  void get(const Tensor<T>& a, T* data) override;
  void zeros(Variable<T>& a) override;
  void ones(Variable<T>& a) override;

  Variable<T> copy(const Tensor<T>& a) override;

  Variable<T> cast(const Tensor<float>& a) override;
  Variable<T> cast(const Tensor<double>& a) override;
  Variable<T> cast(const Tensor<int8_t>& a) override;
  Variable<T> cast(const Tensor<uint8_t>& a) override;
  Variable<T> cast(const Tensor<int16_t>& a) override;
  Variable<T> cast(const Tensor<uint16_t>& a) override;
  Variable<T> cast(const Tensor<int32_t>& a) override;
  Variable<T> cast(const Tensor<uint32_t>& a) override;
  Variable<T> cast(const Tensor<int64_t>& a) override;
  Variable<T> cast(const Tensor<uint64_t>& a) override;

  Variable<uint8_t> cast_to_boolean(const Tensor<T>& a) override;
};

}  // namespace tfcc
