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

#include "tfcc_device.h"

#include <mutex>
#include <utility>

#include "framework/tfcc_constantmanager.h"

namespace tfcc {

thread_local Device* Device::_threadDevice = nullptr;
std::set<Device*> Device::_allDevices;

static SpinLock mtx;

Device::Device(unsigned deviceID, std::unique_ptr<Allocator> allocator)
    : _deviceID(deviceID), _allocator(std::move(allocator)) {
  std::lock_guard<SpinLock> lck(mtx);
  _allDevices.insert(this);
}

Device::~Device() { _allDevices.erase(this); }

size_t Device::getConstantMemoryUsed() const {
  size_t used = 0;
  if (_floatConstantManager) {
    used += _floatConstantManager->getAllocator().used();
  }
  if (_doubleConstantManager) {
    used += _doubleConstantManager->getAllocator().used();
  }
  if (_int8ConstantManager) {
    used += _int8ConstantManager->getAllocator().used();
  }
  if (_uint8ConstantManager) {
    used += _uint8ConstantManager->getAllocator().used();
  }
  if (_int16ConstantManager) {
    used += _int16ConstantManager->getAllocator().used();
  }
  if (_uint16ConstantManager) {
    used += _uint16ConstantManager->getAllocator().used();
  }
  if (_int32ConstantManager) {
    used += _int32ConstantManager->getAllocator().used();
  }
  if (_uint32ConstantManager) {
    used += _uint32ConstantManager->getAllocator().used();
  }
  if (_int64ConstantManager) {
    used += _int64ConstantManager->getAllocator().used();
  }
  if (_uint64ConstantManager) {
    used += _uint64ConstantManager->getAllocator().used();
  }
  return used;
}

std::set<Device*> Device::getAllDevices() {
  std::lock_guard<SpinLock> lck(mtx);
  return _allDevices;
}

unsigned Device::getDeviceID() const { return _deviceID; }

DeviceType Device::getDeviceType() const { return static_cast<DeviceType>(_deviceID >> 24); }

}  // namespace tfcc
