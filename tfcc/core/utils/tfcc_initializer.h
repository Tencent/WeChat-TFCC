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

#include <functional>
#include <memory>
#include <vector>

#include "utils/tfcc_spinlock.h"

namespace tfcc {

class Device;
class Session;

class Initializer;

class InitializerGuard {
  std::shared_ptr<Device> _device;
  std::shared_ptr<Session> _session;

 private:
  InitializerGuard(std::shared_ptr<Device> device, std::shared_ptr<Session> session);

  friend class Initializer;

 public:
  InitializerGuard();
  InitializerGuard(const InitializerGuard&) = default;
  InitializerGuard(InitializerGuard&&) = default;
  ~InitializerGuard();

  InitializerGuard& operator=(const InitializerGuard&) = default;
  InitializerGuard& operator=(InitializerGuard&&) = default;

  Device& getDevice() { return *_device; }
  Session& getSession() { return *_session; }
};

class Initializer {
  std::vector<std::shared_ptr<Device>> _devices;
  std::function<std::shared_ptr<Session>(Device&)> _sessionCreator;
  SpinLock _mtx;

 public:
  Initializer(
      size_t count, std::function<std::shared_ptr<Device>(size_t)> deviceCreator,
      std::function<std::shared_ptr<Session>(Device&)> sessionCreator);
  Initializer(const Initializer&) = delete;
  Initializer(Initializer&&) = delete;
  ~Initializer();

  Initializer& operator=(const Initializer&) = delete;
  Initializer& operator=(Initializer&&) = delete;

  InitializerGuard allocate();
  InitializerGuard allocate(unsigned deviceID);
};

}  // namespace tfcc
