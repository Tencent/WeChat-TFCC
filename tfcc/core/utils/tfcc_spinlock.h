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

#include <atomic>

namespace tfcc {

class SpinLock {
  std::atomic<bool> spin;

 public:
  SpinLock() : spin(false) {}
  SpinLock(const SpinLock&) = delete;
  SpinLock(SpinLock&&) = delete;

  SpinLock& operator=(const SpinLock&) = delete;
  SpinLock& operator=(SpinLock&&) = delete;

  void lock() {
    bool expire;
    do {
      expire = false;
    } while (!spin.compare_exchange_weak(expire, true));
  }

  bool try_lock() {
    bool expire = false;
    return spin.compare_exchange_weak(expire, true);
  }

  void unlock() { spin.store(false); }
};

}  // namespace tfcc
