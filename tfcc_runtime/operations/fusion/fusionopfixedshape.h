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

#include <cstdint>
#include <memory>
#include <vector>

#include "operations/tfcc_mklfusionoperation.h"
#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/framework/types.h"
#include "tfcc_runtime/operations/operation.h"
#include "tfcc_runtime/proto/common.pb.h"

namespace tfcc {
namespace runtime {
namespace fusion {

template <class T>
struct FusionOpFixedShapeResource : public OperationResource {
  std::vector<tfcc::fusionop::OperationType> opTypes;
  std::vector<unsigned> resultShape;
  std::vector<std::vector<bool>> broadcastMarks;
  std::vector<const tfcc::Tensor<T>*> symbols;
  tfcc::fusionop::FusionHandler handler;
};

template <tfcc::runtime::common::DataType DTYPE>
class FusionOpFixedShape : public Operation {
 public:
  std::string getOperationTypeUrl() const override;
  std::set<unsigned> getOperationVersions() const override;
  std::unique_ptr<OperationResource> process(
      const tfcc::runtime::model::Node& node, Graph& graph) const override;

 private:
  using T = typename DataTypeInfo<DTYPE>::CPPType;
};

std::vector<std::unique_ptr<Operation>> get_fusionopfixedshape_operations();

}  // namespace fusion
}  // namespace runtime
}  // namespace tfcc
