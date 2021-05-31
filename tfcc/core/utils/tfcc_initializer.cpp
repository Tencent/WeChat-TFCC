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

#include "tfcc_initializer.h"

#include <limits>
#include <mutex>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "framework/tfcc_device.h"
#include "framework/tfcc_session.h"

namespace tfcc {

InitializerGuard::InitializerGuard(std::shared_ptr<Device> device, std::shared_ptr<Session> session)
    : _device(device), _session(session) {}

InitializerGuard::InitializerGuard() {}

InitializerGuard::~InitializerGuard() {
  _session.reset();
  _device.reset();
}

Initializer::Initializer(
    size_t count, std::function<std::shared_ptr<Device>(size_t)> deviceCreator,
    std::function<std::shared_ptr<Session>(Device&)> sessionCreator)
    : _sessionCreator(sessionCreator) {
  if (count == 0) {
    throw InvalidArgumentError("device count must greater than zero");
  }
  for (size_t i = 0; i < count; ++i) {
    _devices.emplace_back(deviceCreator(i));
  }
}

Initializer::~Initializer() {}

InitializerGuard Initializer::allocate() {
  std::shared_ptr<Device> device;
  long count = std::numeric_limits<long>::max();
  {
    size_t pos = 0;
    std::lock_guard<SpinLock> lck(_mtx);
    for (size_t i = 0; i < _devices.size(); ++i) {
      if (_devices[i].use_count() < count) {
        count = _devices[i].use_count();
        pos = i;
      }
    }
    device = _devices[pos];
  }
  device->attach();
  return InitializerGuard(device, _sessionCreator(*device));
}

InitializerGuard Initializer::allocate(unsigned deviceID) {
  std::shared_ptr<Device> device;
  {
    std::lock_guard<SpinLock> lck(_mtx);
    for (size_t i = 0; i < _devices.size(); ++i) {
      if (_devices[i]->getDeviceID() == deviceID) {
        device = _devices[i];
        break;
      }
    }
  }
  if (!device) {
    throw InvalidArgumentError(
        "device id [" + std::to_string(deviceID) + "] could not found in this allocator");
  }
  device->attach();
  return InitializerGuard(device, _sessionCreator(*device));
}

}  // namespace tfcc
