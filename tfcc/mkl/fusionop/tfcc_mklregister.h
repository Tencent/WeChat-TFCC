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
#include <memory>
#include <set>

#include "xbyak/xbyak.h"

namespace tfcc {
namespace fusionop {

class RegisterManager;

class Register {
 protected:
  std::set<size_t> _positions;
  size_t _priority;
  RegisterManager& _manager;
  bool _stored;
  Xbyak::Ymm _reg;

 public:
  Register(std::set<size_t> positions, size_t priority, RegisterManager& manager);
  virtual ~Register();

  void update(std::set<size_t> positions, size_t priority);

  /**
   * Load data to register (If stored) and return.
   * @return Register object.
   */
  Xbyak::Ymm reg();

  /**
   * Check if the register is stored
   */
  bool stored() const;

  /**
   * Check if the register is still in used
   */
  bool checkInUsed() const;

  /**
   * Get distance score.
   * As the score decreases, the register is more likely to be used.
   */
  virtual size_t score(size_t position) const;

  /**
   * Store register data.
   */
  virtual void store() = 0;

  /**
   * Load register data.
   */
  virtual void load() = 0;
};

}  // namespace fusionop
}  // namespace tfcc
