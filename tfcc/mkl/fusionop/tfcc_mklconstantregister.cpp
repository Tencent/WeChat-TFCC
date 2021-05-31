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

#include "tfcc_mklconstantregister.h"

#include <iostream>
#include <utility>

#include "tfcc_mklregistermanager.h"

namespace tfcc {
namespace fusionop {

static inline bool is_in_int32(uint64_t x) {
  return ~uint64_t(0x7fffffffu) <= x || x <= 0x7fffffffu;
}

ConstantRegister::ConstantRegister(
    const uint32_t* value, std::set<size_t> positions, size_t priority, RegisterManager& manager)
    : Register(std::move(positions), priority, manager), _value(value) {}

ConstantRegister::~ConstantRegister() {}

size_t ConstantRegister::score(size_t position) const {
  size_t score = std::numeric_limits<size_t>::max();
  if (!checkInUsed()) {
    return _manager.getMaxPosition() * 128 + _priority;
  }
  for (size_t pos : _positions) {
    if (pos < position) {
      continue;
    }
    score = std::min(score, (pos - position) * 128 + _priority);
    break;
  }
  return score;
}

void ConstantRegister::store() {
  // std::cout << "constant store used:" << checkInUsed() << std::endl;
  assert(!_stored);
  _manager.freeAvx256Register(_reg);
  _stored = true;
}

void ConstantRegister::load() {
  // std::cout << "constant load used:" << checkInUsed() << std::endl;
  assert(_stored);
  Xbyak::CodeGenerator& jit = _manager.jit();
  _reg = _manager.allocAvx256Register();

  if (is_in_int32(reinterpret_cast<uintptr_t>(_value))) {
    jit.vbroadcastss(_reg, jit.ptr[_value]);
  } else {
    // auto temp = _manager.getGeneralRegister(0);
    // jit.mov(temp->reg(), reinterpret_cast<uintptr_t>(_value));
    // jit.vbroadcastss(_reg, jit.ptr[temp->reg()]);
    jit.mov(jit.rdi, reinterpret_cast<uintptr_t>(_value));
    jit.vbroadcastss(_reg, jit.ptr[jit.rdi]);
  }
  _stored = false;
}

}  // namespace fusionop
}  // namespace tfcc
