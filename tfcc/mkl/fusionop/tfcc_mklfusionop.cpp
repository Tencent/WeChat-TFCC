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

#include "tfcc_mklfusionop.h"

#include <iostream>
#include <stack>

namespace tfcc {
namespace fusionop {

std::tuple<std::map<OperationType, std::set<size_t>>, std::map<size_t, std::set<size_t>>>
get_position_map(const std::vector<OperationType>& opTypes) {
  std::map<OperationType, std::set<size_t>> paramPositionMap;
  std::map<size_t, std::set<size_t>> opPositionsMap;
  std::stack<size_t> stacks;
  for (size_t i = 0; i < opTypes.size(); ++i) {
    if (opTypes[i] >= OperationType::PARAM_0) {
      paramPositionMap[opTypes[i]].insert(i);
    } else if (opTypes[i] >= OperationType::REPEATED) {
      opPositionsMap[opTypes[i] - OperationType::REPEATED].insert(i);
    } else {
      opPositionsMap[i].insert(i);
      std::unique_ptr<Operation> op = get_operation(opTypes[i]);
      for (size_t j = 0; j < op->input_count(); ++j) {
        size_t index = stacks.top();
        stacks.pop();
        if (opTypes[index] >= OperationType::PARAM_0) {
          paramPositionMap[opTypes[index]].insert(i);
        } else if (opTypes[index] >= OperationType::REPEATED) {
          opPositionsMap[opTypes[index] - OperationType::REPEATED].insert(i);
        } else {
          opPositionsMap[index].insert(i);
        }
      }
    }
    stacks.push(i);
  }
  stacks.pop();
  assert(stacks.empty());
  return std::make_tuple(std::move(paramPositionMap), std::move(opPositionsMap));
}

std::vector<std::vector<int64_t>> get_inputs_skip_size(
    const std::vector<unsigned>& resultShape,
    const std::vector<std::vector<bool>>& broadcastMarks) {
  std::vector<std::vector<int64_t>> inputsSkipSize(
      broadcastMarks.size(), std::vector<int64_t>(resultShape.size(), 1));
  if (resultShape.size() >= 2) {
    std::vector<int64_t> inputsCurrentSkip(broadcastMarks.size(), 1);
    for (size_t i = 0; i < resultShape.size(); ++i) {
      for (size_t j = 0; j < inputsSkipSize.size(); ++j) {
        size_t idx = resultShape.size() - i - 1;
        inputsSkipSize[j][idx] = inputsCurrentSkip[j];
        inputsCurrentSkip[j] *= broadcastMarks[j][idx] ? 1 : resultShape[idx];
      }
    }

    std::vector<std::stack<size_t>> st(broadcastMarks.size());
    for (size_t i = 0; i < broadcastMarks.size(); ++i) {
      for (int j = resultShape.size() - 2; j >= 0; --j) {
        if (st[i].empty()) {
          if (broadcastMarks[i][j]) {
            inputsSkipSize[i][j] = 0;
          }
          st[i].push(j);
        } else {
          if (broadcastMarks[i][st[i].top()] == broadcastMarks[i][j]) {
            inputsSkipSize[i][j] = 0;
          } else {
            if (broadcastMarks[i][j]) {
              inputsSkipSize[i][j] = 0 - inputsSkipSize[i][j];
            }
            st[i].push(j);
          }
        }
      }
    }
  }
  // std::cout << "\nresultShape:\n";
  // for (size_t i = 0; i < resultShape.size(); ++i) {
  //     std::cout << resultShape[i] << " ";
  // }
  // std::cout << "\n";
  // std::cout << "broadcastMarks:\n";
  // for (size_t i = 0; i < broadcastMarks.size(); ++i) {
  //     for (size_t j = 0; j < broadcastMarks[0].size(); ++j) {
  //         std::cout << broadcastMarks[i][j] << " ";
  //     }
  //     std::cout << "\n";
  // }
  // std::cout << "inputsSkipSize:\n";
  // for (size_t i = 0; i < inputsSkipSize.size(); ++i) {
  //     for (size_t j = 0; j < inputsSkipSize[0].size(); ++j) {
  //         std::cout << inputsSkipSize[i][j] << " ";
  //     }
  //     std::cout << "\n";
  // }
  return inputsSkipSize;
}

std::shared_ptr<Register> do_rpn(
    const std::vector<OperationType>& opTypes,
    const std::map<OperationType, std::set<size_t>>& paramPositionMap,
    const std::map<size_t, std::set<size_t>>& opPositionsMap,
    const std::map<OperationType, std::shared_ptr<GeneralRegister>>& regMap,
    const std::map<OperationType, bool>& broadcastMarksMap, RegisterManager& manager) {
  // for (size_t i = 0; i < opTypes.size(); ++i) {
  //     std::cout << (int)opTypes[i] << " ";
  // }
  // std::cout << "\n";
  // for (size_t i = 0; i < opTypes.size(); ++i) {
  //     std::cout << (int)opTypes[i] << ":";
  //     if (opTypes[i] >= OperationType::PARAM_0) {
  //         auto s = paramPositionMap.at(opTypes[i]);
  //         for (auto j : s) {
  //             std::cout << j << " ";
  //         }
  //     } else if (opTypes[i] < OperationType::REPEATED) {
  //         auto s = opPositionsMap.at(i);
  //         for (auto j : s) {
  //             std::cout << j << " ";
  //         }
  //     }
  //     std::cout << "\n";
  // }

  std::map<size_t, std::shared_ptr<Register>> repeatedMap;
  std::stack<std::shared_ptr<Register>> regs;
  for (size_t i = 0; i < opTypes.size(); ++i) {
    if (opTypes[i] >= OperationType::PARAM_0) {
      auto reg = manager.getInputRegister(
          opTypes[i], regMap.at(opTypes[i])->reg(), paramPositionMap.at(opTypes[i]), 64,
          broadcastMarksMap.at(opTypes[i]));
      regs.push(reg);
    } else if (opTypes[i] >= OperationType::REPEATED) {
      regs.push(repeatedMap[opTypes[i] - OperationType::REPEATED]);
    } else {
      std::vector<std::shared_ptr<Register>> inputs;
      std::unique_ptr<Operation> op = get_operation(opTypes[i]);
      for (size_t j = 0; j < op->input_count(); ++j) {
        inputs.push_back(regs.top());
        regs.pop();
      }
      std::reverse(inputs.begin(), inputs.end());
      // std::cout << "op:" << int(op->optype()) << std::endl;
      auto temp = op->process(manager, std::move(inputs), opPositionsMap.at(i));
      if (opPositionsMap.at(i).size() > 1) {
        repeatedMap[i] = temp;
      }
      regs.push(temp);
    }
    manager.nextPosition();
  }
  auto result = regs.top();
  regs.pop();
  assert(regs.empty());
  return result;
}

}  // namespace fusionop
}  // namespace tfcc