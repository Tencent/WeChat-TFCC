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

#include "tfcc_flexallocator.h"

#include <cassert>
#include <mutex>

#include "exceptions/tfcc_runtimeerror.h"

namespace tfcc {

FlexAllocator::FlexAllocator() : _used(0), _limit(0), _flexLimit(0) {}

FlexAllocator::~FlexAllocator() {
  checkDestruction();
  releaseCache();
}

void FlexAllocator::setLimit(size_t limit) { _limit = limit; }

size_t FlexAllocator::used() const { return _used.load(); }

void FlexAllocator::setFlexLimit(size_t flexLimit) { _flexLimit = flexLimit; }

void* FlexAllocator::malloc(size_t len) {
  assert(len > 0);
  len = getFlexSize(len);

  void* result = mallocFromCache(len);
  if (result == nullptr) {
    result = mallocFromCallbackFunc(len);
  }

  if (result == nullptr) {
    return nullptr;
  }

  {
    std::lock_guard<SpinLock> lck(_sizeMapMtx);
    assert(_sizeMap.find(result) == _sizeMap.end());
    _sizeMap[result] = len;
  }

  if (_flexLimit != 0 && _used > _flexLimit) {
    releaseSome(_flexLimit / 2);
  }

  return result;
}

void FlexAllocator::free(void* p) {
  if (p == nullptr) {
    return;
  }
  size_t len = 0;
  {
    std::lock_guard<SpinLock> lck(_sizeMapMtx);
    auto it = _sizeMap.find(p);
    assert(it != _sizeMap.end());
    len = it->second;
    _sizeMap.erase(it);
  }

  assert(len > 0);

  {
    std::lock_guard<SpinLock> lck(_freeMapMtx);
    _freeMap.emplace(len, p);
  }
}

void FlexAllocator::releaseCache() { releaseSome(0); }

size_t FlexAllocator::getFlexSize(size_t len) {
  constexpr size_t FLEX_SMALL = 4096;
  constexpr size_t FLEX_LARGE = 1024 * 1024;
  len += 64;
  // flex = 4k if len < 4m
  if (len < 1024 * 1024 * 4) {
    return (len + FLEX_SMALL - 1) / FLEX_SMALL * FLEX_SMALL;
  }
  // flex = 1m if len >= 4m
  return (len + FLEX_LARGE - 1) / FLEX_LARGE * FLEX_LARGE;
}

void* FlexAllocator::mallocFromCache(size_t len) {
  std::lock_guard<SpinLock> lck(_freeMapMtx);
  auto it = _freeMap.find(len);
  if (it == _freeMap.end()) {
    return nullptr;
  }
  void* result = it->second;
  _freeMap.erase(it);
  return result;
}

void* FlexAllocator::mallocFromCallbackFunc(size_t len) {
  void* result = nullptr;
  if (_limit == 0 || _used.load() + len <= _limit) {
    result = _realMalloc(len);
  }
  if (result == nullptr) {
    releaseCache();
    if (_used.load() + len <= _limit) {
      result = _realMalloc(len);
    }
  }
  if (result != nullptr) {
    _used += len;
  }
  return result;
}

void FlexAllocator::checkDestruction() {
  if (!_sizeMap.empty()) {
    throw RuntimeError("release allocator before release all memory");
  }
}

void FlexAllocator::releaseSome(size_t target) {
  while (_used > target) {
    void* p = nullptr;
    size_t len = 0;
    {
      std::lock_guard<SpinLock> lck(_freeMapMtx);
      if (_freeMap.empty()) {
        break;
      }
      auto it = _freeMap.begin();
      p = it->second;
      len = it->first;
      _freeMap.erase(it);
    }
    assert(p != nullptr);
    _realFree(p);
    _used -= len;
  }
}

}  // namespace tfcc
