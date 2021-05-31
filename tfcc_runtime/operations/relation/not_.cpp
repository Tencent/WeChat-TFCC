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

#include "not_.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/relation.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace relation {

static const char* _wrapper_not_v1(
    const tfcc::Tensor<uint8_t>* a, tfcc::Variable<uint8_t>* result) noexcept {
  try {
    *result = tfcc::relation::equal(*a, static_cast<uint8_t>(0));
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

static const char* _wrapper_not_v2(const uint8_t* a, uint8_t* result) noexcept {
  *result = (*a) ? 0 : 1;
  return nullptr;
}

std::string Not::getOperationTypeUrl() const {
  tfcc::runtime::operations::relation::Not operation;
  return get_protobuf_type_url(operation);
}

std::set<unsigned> Not::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::relation::Not::VERSION_1};
  return versions;
}

std::unique_ptr<OperationResource> Not::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 1 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::relation::Not>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  const SymbolInfo& inputInfo = symbolManager.getSymbolInfo(node.inputs(0));
  const SymbolInfo& outputInfo = symbolManager.getSymbolInfo(node.outputs(0));
  if (outputInfo.dtype != tfcc::runtime::common::BOOL ||
      inputInfo.dtype != tfcc::runtime::common::BOOL) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource);
  tfcc::runtime::operations::relation::Not operation;
  node.operation().UnpackTo(&operation);

  auto& jit = getJIT(graph);

  if (is_tensor(inputInfo.stype)) {
    const tfcc::Tensor<uint8_t>* inputSymbol = symbolManager.getTensor(node.inputs(0), uint8_t());
    tfcc::Variable<uint8_t>* outputSymbol = symbolManager.getVariable(node.outputs(0), uint8_t());
    callFunction(jit, _wrapper_not_v1, inputSymbol, outputSymbol);
  } else if (is_value(inputInfo.stype)) {
    const uint8_t* inputSymbol = symbolManager.getValue(node.inputs(0), uint8_t());
    uint8_t* outputSymbol = symbolManager.getValue(node.outputs(0), uint8_t());
    callFunction(jit, _wrapper_not_v2, inputSymbol, outputSymbol);
  } else {
    // error
    return nullptr;
  }

  return resource;
}

std::vector<std::unique_ptr<Operation>> get_not_operations() {
  std::vector<std::unique_ptr<Operation>> operations;
  operations.emplace_back(std::unique_ptr<Operation>(new Not()));
  return operations;
}

}  // namespace relation
}  // namespace runtime
}  // namespace tfcc
