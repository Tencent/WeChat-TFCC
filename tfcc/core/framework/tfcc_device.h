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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>

#include "allocators/tfcc_allocator.h"
#include "framework/tfcc_types.h"
#include "interfaces/tfcc_interface.h"
#include "utils/tfcc_spinlock.h"

namespace tfcc {

class Allocator;

template <class T>
class ConstantManager;

enum class DeviceType : uint8_t {
  CUDA = 0,
  MKL = 1,
};

class Device {
  static thread_local Device* _threadDevice;
  static std::set<Device*> _allDevices;

 protected:
  unsigned _deviceID;
  std::unique_ptr<Allocator> _allocator;

  std::shared_ptr<ConstantManager<float>> _floatConstantManager;
  std::shared_ptr<ConstantManager<double>> _doubleConstantManager;
  std::shared_ptr<ConstantManager<int8_t>> _int8ConstantManager;
  std::shared_ptr<ConstantManager<uint8_t>> _uint8ConstantManager;
  std::shared_ptr<ConstantManager<int16_t>> _int16ConstantManager;
  std::shared_ptr<ConstantManager<uint16_t>> _uint16ConstantManager;
  std::shared_ptr<ConstantManager<int32_t>> _int32ConstantManager;
  std::shared_ptr<ConstantManager<uint32_t>> _uint32ConstantManager;
  std::shared_ptr<ConstantManager<int64_t>> _int64ConstantManager;
  std::shared_ptr<ConstantManager<uint64_t>> _uint64ConstantManager;

  Interface<float> _floatInterface;
  Interface<double> _doubleInterface;
  Interface<int8_t> _int8Interface;
  Interface<uint8_t> _uint8Interface;
  Interface<int16_t> _int16Interface;
  Interface<uint16_t> _uint16Interface;
  Interface<int32_t> _int32Interface;
  Interface<uint32_t> _uint32Interface;
  Interface<int64_t> _int64Interface;
  Interface<uint64_t> _uint64Interface;
  Interface<Complex<float>> _complex64Interface;
  Interface<Complex<double>> _complex128Interface;

 public:
  Device(unsigned deviceID, std::unique_ptr<Allocator> allocator);
  Device(const Device&) = delete;
  Device(Device&&) = default;
  virtual ~Device();

  Device& operator=(const Device&) = delete;
  Device& operator=(Device&&) = delete;

  virtual void attach() = 0;

  void* malloc(size_t len) { return _allocator->malloc(len); }

  void free(void* p) { _allocator->free(p); }

  void releaseCache() { _allocator->releaseCache(); }

  Allocator& getAllocator() { return *_allocator; }

  const Allocator& getAllocator() const { return *_allocator; }

  ConstantManager<float>& getConstantManager(float) { return *_floatConstantManager; }
  ConstantManager<double>& getConstantManager(double) { return *_doubleConstantManager; }
  ConstantManager<int8_t>& getConstantManager(int8_t) { return *_int8ConstantManager; }
  ConstantManager<uint8_t>& getConstantManager(uint8_t) { return *_uint8ConstantManager; }
  ConstantManager<int16_t>& getConstantManager(int16_t) { return *_int16ConstantManager; }
  ConstantManager<uint16_t>& getConstantManager(uint16_t) { return *_uint16ConstantManager; }
  ConstantManager<int32_t>& getConstantManager(int32_t) { return *_int32ConstantManager; }
  ConstantManager<uint32_t>& getConstantManager(uint32_t) { return *_uint32ConstantManager; }
  ConstantManager<int64_t>& getConstantManager(int64_t) { return *_int64ConstantManager; }
  ConstantManager<uint64_t>& getConstantManager(uint64_t) { return *_uint64ConstantManager; }

  Interface<float>& getInterface(float) { return _floatInterface; }
  Interface<double>& getInterface(double) { return _doubleInterface; }
  Interface<int8_t>& getInterface(int8_t) { return _int8Interface; }
  Interface<uint8_t>& getInterface(uint8_t) { return _uint8Interface; }
  Interface<int16_t>& getInterface(int16_t) { return _int16Interface; }
  Interface<uint16_t>& getInterface(uint16_t) { return _uint16Interface; }
  Interface<int32_t>& getInterface(int32_t) { return _int32Interface; }
  Interface<uint32_t>& getInterface(uint32_t) { return _uint32Interface; }
  Interface<int64_t>& getInterface(int64_t) { return _int64Interface; }
  Interface<uint64_t>& getInterface(uint64_t) { return _uint64Interface; }
  Interface<Complex<float>>& getInterface(Complex<float>) { return _complex64Interface; }
  Interface<Complex<double>>& getInterface(Complex<double>) { return _complex128Interface; }

  size_t getConstantMemoryUsed() const;

  /**
   * 8bit device type + 24bit id
   */
  unsigned getDeviceID() const;
  DeviceType getDeviceType() const;

 public:
  static Device* getThreadDefault() { return _threadDevice; }

  static void setThreadDefault(Device* device) {
    _threadDevice = device;
    if (_threadDevice) {
      _threadDevice->attach();
    }
  }

  static std::set<Device*> getAllDevices();
};

}  // namespace tfcc
