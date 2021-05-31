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

#include "tfcc_mklregistermanager.h"

#include <string.h>
#include <cassert>
#include <iostream>

#include "tfcc_mklconstantregister.h"
#include "tfcc_mklinputregister.h"
#include "tfcc_mklregister.h"
#include "tfcc_mkltempregister.h"

namespace tfcc {
namespace fusionop {

static constexpr size_t CONSTANT_TABLE_SIZE = 4 * 1024;

RegisterManager::RegisterManager(
    size_t maxPosition, const std::vector<Xbyak::Reg>& generalRegs,
    const std::vector<Xbyak::Ymm>& avx256Regs, Xbyak::CodeGenerator& jit)
    : _maxPosition(maxPosition),
      _stack_manager(new StackManager(jit)),
      _jit(jit),
      _position(0),
      _tableUsed(0) {
  assert(!generalRegs.empty());
  assert(!avx256Regs.empty());
  for (int i = generalRegs.size() - 1; i >= 0; --i) {
    _general_regs.push(generalRegs[i]);
  }
  for (int i = avx256Regs.size() - 1; i >= 0; --i) {
    _avx256_regs.push(avx256Regs[i]);
  }
  _tables.emplace_back(new uint32_t[CONSTANT_TABLE_SIZE]);
  _callee_saved_regs = {std::make_tuple(_jit.r12, 0, false), std::make_tuple(_jit.r13, 0, false),
                        std::make_tuple(_jit.r14, 0, false), std::make_tuple(_jit.r15, 0, false),
                        std::make_tuple(_jit.rbx, 0, false), std::make_tuple(_jit.rbp, 0, false)};
  for (auto& it : _callee_saved_regs) {
    std::get<1>(it) = allocStack(8);
    _jit.mov(_jit.qword[_jit.rsp + stackSize() - std::get<1>(it)], std::get<0>(it));
    // std::cout << "reg:" << std::get<0>(it).toString() << " offset:" << stackSize() -
    // std::get<1>(it)<< "\n";
  }
}

RegisterManager::~RegisterManager() { recoverGeneralRegisters(); }

void RegisterManager::nextPosition() { ++_position; }

size_t RegisterManager::getPosition() const { return _position; }

size_t RegisterManager::getMaxPosition() const { return _maxPosition; }

std::shared_ptr<Register> RegisterManager::getFloatConstantRegister(float value, size_t priority) {
  uint32_t key = 0;
  memcpy(&key, &value, sizeof(uint32_t));
  return getUint32ConstantRegister(key, priority);
}

std::shared_ptr<Register> RegisterManager::getUint32ConstantRegister(
    uint32_t value, size_t priority) {
  if (_dataMap.find(value) == _dataMap.end()) {
    if (_tableUsed >= CONSTANT_TABLE_SIZE) {
      _tables.emplace_back(new uint32_t[CONSTANT_TABLE_SIZE]);
      _tableUsed = 0;
    }
    _tables[_tables.size() - 1][_tableUsed] = value;
    std::shared_ptr<Register> r = std::make_shared<ConstantRegister>(
        _tables[_tables.size() - 1].get() + _tableUsed, std::set<size_t>(), priority, *this);
    _dataMap[value] = r;
    ++_tableUsed;
    _avx256_reg_guards.push_back(r);
  }

  _dataMap[value]->update({_position}, priority);
  return _dataMap[value];
}

std::shared_ptr<Register> RegisterManager::getInputRegister(
    OperationType type, Xbyak::Reg input, std::set<size_t> positions, size_t priority,
    bool broadcast) {
  if (_input_regs.find(type) == _input_regs.end()) {
    std::shared_ptr<Register> r =
        std::make_shared<InputRegister>(input, std::move(positions), priority, broadcast, *this);
    _input_regs[type] = r;
    _avx256_reg_guards.push_back(r);
  }
  // update positions and priority?
  return _input_regs[type];
}

std::shared_ptr<Register> RegisterManager::getTempRegister(
    std::set<size_t> positions, size_t priority) {
  std::shared_ptr<Register> r =
      std::make_shared<TempRegister>(std::move(positions), priority, *this);
  _avx256_reg_guards.push_back(r);
  return r;
}

std::shared_ptr<GeneralRegister> RegisterManager::getGeneralRegister(size_t priority) {
  Xbyak::Reg reg;
  bool alloc = false;
  if (!_general_regs.empty()) {
    reg = _general_regs.top();
    _general_regs.pop();
    alloc = true;
    for (auto& it : _callee_saved_regs) {
      if (std::get<0>(it) == reg) {
        std::get<2>(it) = true;
        break;
      }
    }
  }

  std::shared_ptr<GeneralRegister> r =
      std::make_shared<GeneralRegister>(reg, alloc, priority, *this);
  _general_reg_guards.push_back(r);
  return r;
}

Xbyak::CodeGenerator& RegisterManager::jit() { return _jit; }

Xbyak::Ymm RegisterManager::allocAvx256Register() {
  if (_avx256_regs.empty()) {
    size_t score = 0;
    std::shared_ptr<Register> axv256Sp;
    for (auto& wp : _avx256_reg_guards) {
      auto sp = wp.lock();
      if (!sp || sp->stored()) {
        continue;
      }
      if (!sp->checkInUsed()) {
        axv256Sp = sp;
        break;
      }

      size_t s = sp->score(_position);
      if (s > score || !axv256Sp) {
        score = s;
        axv256Sp = sp;
      }
    }
    assert(axv256Sp);
    axv256Sp->store();
  }
  assert(!_avx256_regs.empty());
  Xbyak::Ymm reg = _avx256_regs.top();
  _avx256_regs.pop();
  return reg;
}

void RegisterManager::freeAvx256Register(Xbyak::Ymm reg) { _avx256_regs.push(reg); }

Xbyak::Reg RegisterManager::allocGeneralRegister() {
  if (_general_regs.empty()) {
    size_t score = 0;
    std::shared_ptr<GeneralRegister> generalSp;
    for (auto& wp : _general_reg_guards) {
      auto sp = wp.lock();
      if (sp && sp->isAllocRegister()) {
        if (sp->score() > score || !generalSp) {
          score = sp->score();
          generalSp = sp;
        }
      }
    }
    assert(generalSp);
    generalSp->store();
  }
  assert(!_general_regs.empty());
  Xbyak::Reg reg = _general_regs.top();
  _general_regs.pop();
  return reg;
}

void RegisterManager::freeGeneralRegister(Xbyak::Reg reg) { _general_regs.push(reg); }

size_t RegisterManager::allocStack(size_t size) { return _stack_manager->allocStack(size); }

void RegisterManager::freeStack(size_t size, size_t offset) {
  return _stack_manager->freeStack(size, offset);
}

size_t RegisterManager::stackSize() const { return _stack_manager->stackSize(); }

void RegisterManager::recoverGeneralRegisters() {
  for (auto& it : _callee_saved_regs) {
    if (std::get<2>(it)) {
      // std::cout << "reg2:" << std::get<0>(it).toString() << " offset:" << stackSize() -
      // std::get<1>(it) << "\n";
      _jit.mov(std::get<0>(it), _jit.qword[_jit.rsp + stackSize() - std::get<1>(it)]);
    }
    std::get<2>(it) = false;
  }
}

void RegisterManager::freeStack() {
  recoverGeneralRegisters();
  _stack_manager.reset();
}

}  // namespace fusionop
}  // namespace tfcc