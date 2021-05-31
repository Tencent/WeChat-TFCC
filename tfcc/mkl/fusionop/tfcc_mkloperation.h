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

#include "operations/tfcc_mkloperationtype.h"
#include "tfcc_mklregister.h"
#include "tfcc_mklregistermanager.h"

namespace tfcc {
namespace fusionop {

class Operation {
 public:
  Operation() {}
  virtual ~Operation() {}

  virtual OperationType optype() const = 0;
  virtual size_t input_count() const = 0;
  virtual std::shared_ptr<Register> process(
      RegisterManager& manager, std::vector<std::shared_ptr<Register>> registers,
      std::set<size_t> positions) = 0;
};

std::unique_ptr<Operation> get_operation(OperationType opType);

}  // namespace fusionop
}  // namespace tfcc
