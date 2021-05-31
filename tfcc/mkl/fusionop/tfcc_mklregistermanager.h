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
#include <memory>
#include <set>
#include <stack>
#include <vector>

#include "operations/tfcc_mkloperationtype.h"
#include "tfcc_mklgeneralregister.h"
#include "tfcc_mklstackmanager.h"
#include "xbyak/xbyak.h"

namespace tfcc {
namespace fusionop {

class Register;

class RegisterManager {
  size_t _maxPosition;
  std::unique_ptr<StackManager> _stack_manager;
  Xbyak::CodeGenerator& _jit;
  size_t _position;

  std::stack<Xbyak::Ymm> _avx256_regs;
  std::stack<Xbyak::Reg> _general_regs;
  std::vector<std::weak_ptr<Register>> _avx256_reg_guards;
  std::vector<std::weak_ptr<GeneralRegister>> _general_reg_guards;

  // For ConstantRegister
  size_t _tableUsed;
  std::vector<std::unique_ptr<uint32_t[]>> _tables;
  std::map<uint32_t, std::shared_ptr<Register>> _dataMap;
  // For InputRegister
  std::map<OperationType, std::shared_ptr<Register>> _input_regs;
  // For GeneralRegister
  std::vector<std::tuple<Xbyak::Reg, size_t, bool>> _callee_saved_regs;

 private:
  void recoverGeneralRegisters();

 public:
  RegisterManager(
      size_t maxPosition, const std::vector<Xbyak::Reg>& generalRegs,
      const std::vector<Xbyak::Ymm>& avx256Regs, Xbyak::CodeGenerator& jit);
  ~RegisterManager();
  void nextPosition();
  size_t getPosition() const;
  size_t getMaxPosition() const;

  // register getter, priority [0, 128]
  std::shared_ptr<Register> getFloatConstantRegister(float value, size_t priority);
  std::shared_ptr<Register> getUint32ConstantRegister(uint32_t value, size_t priority);
  std::shared_ptr<Register> getInputRegister(
      OperationType type, Xbyak::Reg input, std::set<size_t> positions, size_t priority,
      bool broadcast);
  std::shared_ptr<Register> getTempRegister(std::set<size_t> positions, size_t priority);
  std::shared_ptr<GeneralRegister> getGeneralRegister(size_t priority);

  Xbyak::CodeGenerator& jit();

  Xbyak::Ymm allocAvx256Register();
  void freeAvx256Register(Xbyak::Ymm reg);

  void freeGeneralRegister(Xbyak::Reg reg);
  Xbyak::Reg allocGeneralRegister();

  /**
   * @return The stack offset.
   */
  size_t allocStack(size_t size);
  void freeStack(size_t size, size_t offset);
  size_t stackSize() const;

  /**
   * must call it before jit ret.
   */
  void freeStack();
};

}  // namespace fusionop
}  // namespace tfcc
