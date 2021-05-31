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
#include <vector>

#include "tfcc_mkloperation.h"
#include "tfcc_mklregistermanager.h"

namespace tfcc {
namespace fusionop {

// paramPositionMap and opPositionsMap
std::tuple<std::map<OperationType, std::set<size_t>>, std::map<size_t, std::set<size_t>>>
get_position_map(const std::vector<OperationType>& opTypes);
std::vector<std::vector<int64_t>> get_inputs_skip_size(
    const std::vector<unsigned>& resultShape, const std::vector<std::vector<bool>>& broadcastMarks);
std::shared_ptr<Register> do_rpn(
    const std::vector<OperationType>& opTypes,
    const std::map<OperationType, std::set<size_t>>& paramPositionMap,
    const std::map<size_t, std::set<size_t>>& opPositionsMap,
    const std::map<OperationType, std::shared_ptr<GeneralRegister>>& regMap,
    const std::map<OperationType, bool>& broadcastMarksMap, RegisterManager& manager);

}  // namespace fusionop
}  // namespace tfcc
