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

#include "tfcc_mklfusion_mul.h"

namespace tfcc {
namespace fusionop {

OperationType Mul::optype() const { return OperationType::MUL; }

size_t Mul::input_count() const { return 2; }

std::shared_ptr<Register> Mul::process(
    RegisterManager& manager, std::vector<std::shared_ptr<Register>> inputs,
    std::set<size_t> positions) {
  assert(inputs.size() == input_count());
  auto result = manager.getTempRegister(std::move(positions), 64);
  manager.jit().vmulps(result->reg(), inputs[0]->reg(), inputs[1]->reg());
  return result;
}

}  // namespace fusionop
}  // namespace tfcc
