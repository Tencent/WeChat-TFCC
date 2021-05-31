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

#include "tfcc_mklinputregister.h"

#include <iostream>
#include <utility>

#include "tfcc_mklregistermanager.h"

namespace tfcc {
namespace fusionop {

InputRegister::InputRegister(
    Xbyak::Reg input, std::set<size_t> positions, size_t priority, bool broadcast,
    RegisterManager& manager)
    : Register(std::move(positions), priority, manager), _input(input), _broadcast(broadcast) {}

InputRegister::~InputRegister() {}

void InputRegister::store() {
  // std::cout << "input store used:" << checkInUsed() << std::endl;
  assert(!_stored);
  _manager.freeAvx256Register(_reg);
  _stored = true;
}

void InputRegister::load() {
  // std::cout << "input load used:" << checkInUsed() << std::endl;
  assert(_stored);
  Xbyak::CodeGenerator& jit = _manager.jit();
  _reg = _manager.allocAvx256Register();
  if (_broadcast) {
    jit.vbroadcastss(_reg, jit.ptr[_input]);
  } else {
    jit.vmovups(_reg, jit.ptr[_input + jit.rsi * sizeof(float)]);
  }
  _stored = false;
}

}  // namespace fusionop
}  // namespace tfcc