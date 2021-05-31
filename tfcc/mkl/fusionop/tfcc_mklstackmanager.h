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

#include <map>
#include <stack>

#include "xbyak/xbyak.h"

namespace tfcc {
namespace fusionop {

class StackManager {
  Xbyak::CodeGenerator& _jit;
  size_t _stackSize;
  size_t _stackUsed;
  std::map<size_t, std::stack<size_t>> _alloc_map;

  size_t allocStackPool(size_t size);
  void freeStackPool();

 public:
  explicit StackManager(Xbyak::CodeGenerator& jit);
  ~StackManager();

  /**
   * @return The stack offset.
   */
  size_t allocStack(size_t size);
  void freeStack(size_t size, size_t offset);
  size_t stackSize() const;
};

}  // namespace fusionop
}  // namespace tfcc
