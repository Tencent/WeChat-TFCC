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

#include "tfcc_session.h"

#include <utility>

namespace tfcc {

thread_local Session* Session::_threadSession = nullptr;

Session::Session(std::unique_ptr<Allocator> allocator) : _allocator(std::move(allocator)) {}

Session::~Session() {}

void Session::registerReleaseCallback(std::function<void(const Session*)> callBack) {
  _releaseCallbacks.push_back(std::move(callBack));
}

void Session::preRelease() {
  for (auto& callBack : _releaseCallbacks) {
    callBack(this);
  }
  _allocator.reset();
}

}  // namespace tfcc
