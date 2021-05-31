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

#include "tfcc_mkltempregister.h"

#include <iostream>
#include <utility>

#include "tfcc_mklregistermanager.h"

namespace tfcc {
namespace fusionop {

TempRegister::TempRegister(std::set<size_t> positions, size_t priority, RegisterManager& manager)
    : Register(std::move(positions), priority, manager), _offset(0) {
  _reg = manager.allocAvx256Register();
  _stored = false;
  // std::cout << this << " temp create used:" << checkInUsed() << std::endl;
}

TempRegister::~TempRegister() {
  if (_offset > 0) {
    _manager.freeStack(32, _offset);
  }
  // std::cout << this << " temp destory used:" << checkInUsed() << std::endl;
}

void TempRegister::store() {
  // std::cout << this << " temp store used:" << checkInUsed() << std::endl;
  assert(_offset == 0);
  assert(!_stored);
  if (checkInUsed()) {
    Xbyak::CodeGenerator& jit = _manager.jit();
    _offset = _manager.allocStack(32);
    size_t offset = _manager.stackSize() - _offset;
    jit.vmovups(jit.ptr[jit.rsp + offset], _reg);
    // std::cout << this << " store: " << offset << std::endl;
  }

  _manager.freeAvx256Register(_reg);
  _stored = true;
}

void TempRegister::load() {
  // std::cout << this << " temp load used:" << checkInUsed() << std::endl;
  assert(_offset != 0);
  assert(_stored);
  Xbyak::CodeGenerator& jit = _manager.jit();

  _reg = _manager.allocAvx256Register();
  size_t offset = _manager.stackSize() - _offset;
  // std::cout << this << " load: " << offset << std::endl;
  jit.vmovups(_reg, jit.ptr[jit.rsp + offset]);
  _manager.freeStack(32, _offset);
  _offset = 0;
  _stored = false;
}

}  // namespace fusionop
}  // namespace tfcc