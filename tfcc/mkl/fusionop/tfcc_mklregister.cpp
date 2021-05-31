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

#include "tfcc_mklregister.h"

#include <algorithm>
#include <limits>

#include "tfcc_mklregistermanager.h"

namespace tfcc {
namespace fusionop {

Register::Register(std::set<size_t> positions, size_t priority, RegisterManager& manager)
    : _positions(std::move(positions)), _priority(priority), _manager(manager), _stored(true) {}

Register::~Register() {
  if (!_stored) {
    _manager.freeAvx256Register(_reg);
  }
}

void Register::update(std::set<size_t> positions, size_t priority) {
  assert(!positions.empty());
  _positions = std::move(positions);
  _priority = priority;
}

Xbyak::Ymm Register::reg() {
  if (_stored) {
    assert(checkInUsed());
    load();
  }
  return _reg;
}

bool Register::stored() const { return _stored; }

bool Register::checkInUsed() const {
  if (_positions.empty() || (*_positions.rbegin() < _manager.getPosition())) {
    return false;
  }
  return true;
}

size_t Register::score(size_t position) const {
  size_t score = std::numeric_limits<size_t>::max();
  if (!checkInUsed()) {
    return score;
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

}  // namespace fusionop
}  // namespace tfcc