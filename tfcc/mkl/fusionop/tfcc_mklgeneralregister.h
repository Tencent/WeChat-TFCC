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

#include <memory>

#include "xbyak/xbyak.h"

namespace tfcc {
namespace fusionop {

class RegisterManager;

class GeneralRegister {
  Xbyak::Reg _reg;
  bool _alloc;
  size_t _offset;
  size_t _priority;
  RegisterManager& _manager;

 public:
  GeneralRegister(Xbyak::Reg reg, bool alloc, size_t priority, RegisterManager& manager);
  ~GeneralRegister();

  Xbyak::Reg reg();
  void store();

  size_t score() const;
  bool isAllocRegister() const;
};

}  // namespace fusionop
}  // namespace tfcc
