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
#include <vector>

#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"
#include "operations/tfcc_mkloperationtype.h"

namespace tfcc {
namespace fusionop {

class FusionOpFixedShape;
class FusionOpDynamicShape;
class FusionHandler {
 public:
  FusionHandler() : fixedHandler(nullptr), dynamicHandlers(nullptr) {}

  FusionHandler(
      FusionOpFixedShape* a,
      std::map<std::vector<std::vector<bool>>, std::unique_ptr<FusionOpDynamicShape>>* b)
      : fixedHandler(a), dynamicHandlers(b) {}

  FusionHandler(const FusionHandler& x) {
    fixedHandler = x.fixedHandler;
    dynamicHandlers = x.dynamicHandlers;
  }

 private:
  FusionOpFixedShape* fixedHandler;
  std::map<std::vector<std::vector<bool>>, std::unique_ptr<FusionOpDynamicShape>>* dynamicHandlers;
  friend tfcc::Variable<float> fixedShapeFusion(
      const std::vector<OperationType>& opTypes, const std::vector<unsigned>& resultShape,
      const std::vector<std::vector<bool>>& broadcastMarks,
      const std::vector<const tfcc::Tensor<float>*>& inputs, const FusionHandler& handler);
  friend tfcc::Variable<float> dynamicShapeFusion(
      const std::vector<OperationType>& opTypes,
      const std::vector<const tfcc::Tensor<float>*>& inputs, const FusionHandler& handler);
};

// FusionOpFixedShape before call
FusionHandler getFixedShapeFusionHandler(
    const std::vector<OperationType>& opTypes, const std::vector<unsigned>& resultShape,
    const std::vector<std::vector<bool>>& broadcastMarks);

// FusionOpFixedShape in call
tfcc::Variable<float> fixedShapeFusion(
    const std::vector<OperationType>& opTypes, const std::vector<unsigned>& resultShape,
    const std::vector<std::vector<bool>>& broadcastMarks,
    const std::vector<const tfcc::Tensor<float>*>& inputs, const FusionHandler& handler);

// FusionOpDynamicShape before call
FusionHandler getDynamicShapeFusionHandler(const std::vector<OperationType>& opTypes);

// FusionOpDynamicShape in call
tfcc::Variable<float> dynamicShapeFusion(
    const std::vector<OperationType>& opTypes,
    const std::vector<const tfcc::Tensor<float>*>& inputs, const FusionHandler& handler);

}  // namespace fusionop
}  // namespace tfcc
