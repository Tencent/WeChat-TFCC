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

#include "allocators/tfcc_allocator.h"
#include "framework/tfcc_cudadeviceproperty.h"
#include "framework/tfcc_device.h"

namespace tfcc {

class CUDADevice : public Device {
  CUDADeviceProperty _property;

 public:
  explicit CUDADevice(unsigned deviceID);
  CUDADevice(unsigned deviceID, std::unique_ptr<Allocator> allocator);
  CUDADevice(CUDADevice&&) = default;
  ~CUDADevice();

  CUDADevice& operator=(const CUDADevice&) = delete;
  CUDADevice& operator=(CUDADevice&&) = delete;

  void attach() override;

  const CUDADeviceProperty& getProperty() const;

  void enableTensorCore();
  void disableTensorCore();
  bool isTensorCoreEnabled() const;

 public:
  static size_t getDeviceCount();
  bool _tensorCore;
};

}  // namespace tfcc
