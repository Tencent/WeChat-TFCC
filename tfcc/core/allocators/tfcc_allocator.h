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

namespace tfcc {

class Allocator {
 protected:
  std::function<void*(size_t)> _realMalloc;
  std::function<void(void*)> _realFree;

 public:
  Allocator() = default;
  Allocator(const Allocator&) = delete;
  Allocator(Allocator&&) = default;
  virtual ~Allocator() {}

  Allocator& operator=(const Allocator&) = delete;
  Allocator& operator=(Allocator&&) = default;

  virtual void* malloc(size_t len) = 0;
  virtual void free(void* p) = 0;
  virtual void releaseCache() = 0;
  virtual void setLimit(size_t limit) = 0;
  virtual size_t used() const = 0;

  void setRealMalloc(std::function<void*(size_t)> func) { _realMalloc = func; }

  void setRealFree(std::function<void(void*)> func) { _realFree = func; }
};

}  // namespace tfcc
