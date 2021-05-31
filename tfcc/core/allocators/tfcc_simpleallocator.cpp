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

#include "tfcc_simpleallocator.h"

#include <cassert>
#include <mutex>

namespace tfcc {

SimpleAllocator::SimpleAllocator() : _used(0), _limit(0) {}

SimpleAllocator::~SimpleAllocator() {}

void* SimpleAllocator::malloc(size_t len) {
  if (_limit > 0 && _used + len > _limit) {
    return nullptr;
  }
  len = (len + 63) / 64 * 64;
  void* result = _realMalloc(len);
  if (result != nullptr) {
    std::lock_guard<SpinLock> lck(_mtx);
    _used += len;
    _sizeMap.insert(std::make_pair(result, len));
  }
  return result;
}

void SimpleAllocator::free(void* p) {
  if (p == nullptr) {
    return;
  }
  {
    std::lock_guard<SpinLock> lck(_mtx);
    auto it = _sizeMap.find(p);
    assert(it != _sizeMap.end());
    _used -= it->second;
    _sizeMap.erase(it);
  }
  _realFree(p);
}

void SimpleAllocator::releaseCache() {}

void SimpleAllocator::setLimit(size_t limit) { _limit = limit; }

size_t SimpleAllocator::used() const { return _used; }

}  // namespace tfcc
