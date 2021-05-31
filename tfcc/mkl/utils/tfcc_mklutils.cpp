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

#include "tfcc_mklutils.h"

#include "framework/tfcc_mkldevice.h"
#include "framework/tfcc_mklsession.h"
#include "utils/tfcc_initializer.h"

namespace tfcc {

static Initializer& _get_initializer(size_t deviceCount, size_t threadCount) {
  static Initializer initializer(
      deviceCount,
      [threadCount](size_t i) {
        MKLDevice* device = new MKLDevice();
        if (threadCount > 0) {
          device->setNumberThread(threadCount);
        }
        return std::shared_ptr<Device>(device);
      },
      [](Device& device) {
        MKLSession* session = new MKLSession(*dynamic_cast<MKLDevice*>(&device));
        return std::shared_ptr<Session>(session);
      });
  return initializer;
}

static InitializerGuard& _get_initializer_guard() {
  thread_local InitializerGuard guard;
  return guard;
}

void initialize_mkl(size_t deviceCount, size_t threadCount) {
  if (Device::getThreadDefault() == nullptr || Session::getThreadDefault() == nullptr) {
    InitializerGuard& guard = _get_initializer_guard();
    guard = _get_initializer(deviceCount, threadCount).allocate();
    Device::setThreadDefault(&guard.getDevice());
    Session::setThreadDefault(&guard.getSession());
  }
  Device::getThreadDefault()->attach();
}

}  // namespace tfcc
