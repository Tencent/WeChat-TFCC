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

#include "tfcc_mklstackmanager.h"

#include <iostream>
#include <stdexcept>

namespace tfcc {
namespace fusionop {

StackManager::StackManager(Xbyak::CodeGenerator& jit) : _jit(jit), _stackSize(1024), _stackUsed(8) {
  _jit.sub(_jit.rsp, _stackSize);
}

StackManager::~StackManager() { freeStackPool(); }

size_t StackManager::allocStack(size_t size) {
  size_t offset = 0;
  if (_alloc_map[size].empty()) {
    offset = allocStackPool(size);
  } else {
    offset = _alloc_map[size].top();
    _alloc_map[size].pop();
  }
  return offset;
}

void StackManager::freeStack(size_t size, size_t offset) { _alloc_map[size].push(offset); }

size_t StackManager::allocStackPool(size_t size) {
  if (_stackUsed + size > _stackSize) {
    throw std::runtime_error("no stack memory");
  }
  _stackUsed += size;
  return _stackUsed;
}

void StackManager::freeStackPool() {
  // std::cout << "stackSize:" << _stackSize << std::endl;
  if (_stackSize > 0) {
    _jit.add(_jit.rsp, _stackSize);
    _stackSize = 0;
    _stackUsed = 0;
  }
}

size_t StackManager::stackSize() const { return _stackSize; }

}  // namespace fusionop
}  // namespace tfcc