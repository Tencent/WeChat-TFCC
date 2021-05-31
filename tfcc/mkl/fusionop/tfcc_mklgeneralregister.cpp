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

#include "tfcc_mklgeneralregister.h"
#include "tfcc_mklregistermanager.h"

#include <iostream>

namespace tfcc {
namespace fusionop {

GeneralRegister::GeneralRegister(
    Xbyak::Reg reg, bool alloc, size_t priority, RegisterManager& manager)
    : _reg(reg), _alloc(alloc), _offset(0), _priority(priority), _manager(manager) {}

GeneralRegister::~GeneralRegister() {
  if (_offset > 0) {
    _manager.freeStack(8, _offset);
  } else {
    _manager.freeGeneralRegister(_reg);
  }
}

Xbyak::Reg GeneralRegister::reg() {
  Xbyak::CodeGenerator& jit = _manager.jit();
  if (!_alloc) {
    _reg = _manager.allocGeneralRegister();
    _alloc = true;
  } else if (_offset > 0) {
    _reg = _manager.allocGeneralRegister();
    size_t offset = _manager.stackSize() - _offset;
    jit.mov(_reg, jit.qword[jit.rsp + offset]);
    _manager.freeStack(8, _offset);
    _offset = 0;
  }
  return _reg;
}

void GeneralRegister::store() {
  std::cout << "stpre\n";
  assert(_alloc);
  assert(_offset == 0);
  Xbyak::CodeGenerator& jit = _manager.jit();
  _offset = _manager.allocStack(8);
  size_t offset = _manager.stackSize() - _offset;
  jit.mov(jit.qword[jit.rsp + offset], _reg);
  _manager.freeGeneralRegister(_reg);
}

size_t GeneralRegister::score() const { return _priority; }

bool GeneralRegister::isAllocRegister() const { return (_alloc && _offset == 0); }

}  // namespace fusionop
}  // namespace tfcc