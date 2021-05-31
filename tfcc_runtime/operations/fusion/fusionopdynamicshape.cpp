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

#include "fusionopdynamicshape.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/fusion.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace fusion {

template <class T>
static const char* _wrapper_fusionopdynamicshape(
    const std::vector<tfcc::fusionop::OperationType>* opTypes,
    const std::vector<const tfcc::Tensor<float>*>* inputs, tfcc::Variable<float>* result,
    tfcc::fusionop::FusionHandler* handler) noexcept {
  try {
    *result = tfcc::fusionop::dynamicShapeFusion(*opTypes, *inputs, *handler);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE>
std::string FusionOpDynamicShape<DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::fusion::FusionOpDynamicShape operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE>
std::set<unsigned> FusionOpDynamicShape<DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {
      tfcc::runtime::operations::fusion::FusionOpDynamicShape::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE>
std::unique_ptr<OperationResource> FusionOpDynamicShape<DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() == 0 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::fusion::FusionOpDynamicShape>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  if (symbolManager.getSymbolInfo(node.outputs(0)).dtype != DTYPE) {
    return nullptr;
  }
  for (const std::string& name : node.inputs()) {
    if (symbolManager.getSymbolInfo(name).dtype != DTYPE) {
      return nullptr;
    }
  }

  std::unique_ptr<FusionOpDynamicShapeResource<T>> resource(new FusionOpDynamicShapeResource<T>());
  tfcc::runtime::operations::fusion::FusionOpDynamicShape operation;
  node.operation().UnpackTo(&operation);
  for (auto r : operation.op_types()) {
    resource->opTypes.push_back((tfcc::fusionop::OperationType)r);
  }

  auto& jit = getJIT(graph);

  for (const std::string& name : node.inputs()) {
    const tfcc::Tensor<T>* symbol = symbolManager.getTensor(name, T());
    resource->symbols.push_back(symbol);
  }

  tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());

  resource->handler = tfcc::fusionop::getDynamicShapeFusionHandler(resource->opTypes);
  callFunction(
      jit, _wrapper_fusionopdynamicshape<T>, &resource->opTypes, &resource->symbols, outputSymbol,
      &resource->handler);

  return std::unique_ptr<OperationResource>(std::move(resource));
}

std::vector<std::unique_ptr<Operation>> get_fusionopdynamicshape_operations() {
#define DEFINE_FUNC(dtype) \
  operations.emplace_back( \
      std::unique_ptr<Operation>(new FusionOpDynamicShape<tfcc::runtime::common::FLOAT>()));

  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace fusion
}  // namespace runtime
}  // namespace tfcc
