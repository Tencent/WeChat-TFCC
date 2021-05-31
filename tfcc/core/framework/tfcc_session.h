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
#include <functional>
#include <memory>
#include <vector>

#include "allocators/tfcc_allocator.h"

namespace tfcc {

class Device;

class Session {
  static thread_local Session* _threadSession;

 protected:
  std::unique_ptr<Allocator> _allocator;
  std::vector<std::function<void(const Session*)>> _releaseCallbacks;

 public:
  explicit Session(std::unique_ptr<Allocator> allocator);
  Session(const Session&) = delete;
  Session(Session&&) = default;
  virtual ~Session();

  Session& operator=(const Session&) = delete;
  Session& operator=(Session&&) = delete;

  virtual void sync() const = 0;
  virtual Device& getDevice() = 0;
  virtual const Device& getDevice() const = 0;

  void registerReleaseCallback(std::function<void(const Session*)> callBack);

  void* malloc(size_t len) { return _allocator->malloc(len); }

  void free(void* p) { _allocator->free(p); }

  void releaseCache() { _allocator->releaseCache(); }

  Allocator& getAllocator() { return *_allocator; }

  const Allocator& getAllocator() const { return *_allocator; }

  static Session* getThreadDefault() { return _threadSession; }

  static void setThreadDefault(Session* session) { _threadSession = session; }

 protected:
  void preRelease();
};

}  // namespace tfcc
