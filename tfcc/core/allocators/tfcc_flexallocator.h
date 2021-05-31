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
#include <map>

#include "allocators/tfcc_allocator.h"
#include "utils/tfcc_spinlock.h"

namespace tfcc {

class FlexAllocator : public Allocator {
  std::multimap<size_t, void*> _freeMap;
  std::map<void*, size_t> _sizeMap;
  SpinLock _freeMapMtx, _sizeMapMtx;
  std::atomic<size_t> _used;
  size_t _limit;
  size_t _flexLimit;

 public:
  FlexAllocator();
  ~FlexAllocator();

  void* malloc(size_t len) override;
  void free(void* p) override;
  void releaseCache() override;
  void setLimit(size_t limit) override;
  size_t used() const override;

  void setFlexLimit(size_t flexLimit);

 private:
  size_t getFlexSize(size_t len);
  void* mallocFromCache(size_t len);
  void* mallocFromCallbackFunc(size_t len);
  void checkDestruction();
  void releaseSome(size_t target);
};

}  // namespace tfcc
