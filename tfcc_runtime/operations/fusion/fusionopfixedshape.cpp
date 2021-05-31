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

#include "fusionopfixedshape.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/fusion.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace fusion {

template <class T>
static const char* _wrapper_fusionopfixedshape(
    const std::vector<tfcc::fusionop::OperationType>* opTypes,
    const std::vector<unsigned>* resultShape, const std::vector<std::vector<bool>>* broadcastMarks,
    const std::vector<const tfcc::Tensor<float>*>* inputs, tfcc::Variable<float>* result,
    tfcc::fusionop::FusionHandler* handler) noexcept {
  try {
    *result = tfcc::fusionop::fixedShapeFusion(
        *opTypes, *resultShape, *broadcastMarks, *inputs, *handler);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE>
std::string FusionOpFixedShape<DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::fusion::FusionOpFixedShape operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE>
std::set<unsigned> FusionOpFixedShape<DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::fusion::FusionOpFixedShape::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE>
std::unique_ptr<OperationResource> FusionOpFixedShape<DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() == 0 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::fusion::FusionOpFixedShape>()) {
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

  std::unique_ptr<FusionOpFixedShapeResource<T>> resource(new FusionOpFixedShapeResource<T>());
  tfcc::runtime::operations::fusion::FusionOpFixedShape operation;
  node.operation().UnpackTo(&operation);
  for (auto r : operation.op_types()) {
    resource->opTypes.push_back((tfcc::fusionop::OperationType)r);
  }
  for (auto r : operation.result_shape()) {
    resource->resultShape.push_back(r);
  }
  int index = 0;
  resource->broadcastMarks.resize(node.inputs_size());
  for (auto& v : resource->broadcastMarks) {
    for (int i = 0; i < operation.result_shape_size(); ++i) {
      v.push_back(operation.broadcast_marks(index++));
    }
  }

  auto& jit = getJIT(graph);

  for (const std::string& name : node.inputs()) {
    const tfcc::Tensor<T>* symbol = symbolManager.getTensor(name, T());
    resource->symbols.push_back(symbol);
  }

  tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());

  resource->handler = tfcc::fusionop::getFixedShapeFusionHandler(
      resource->opTypes, resource->resultShape, resource->broadcastMarks);
  callFunction(
      jit, _wrapper_fusionopfixedshape<T>, &resource->opTypes, &resource->resultShape,
      &resource->broadcastMarks, &resource->symbols, outputSymbol, &resource->handler);

  return std::unique_ptr<OperationResource>(std::move(resource));
}

std::vector<std::unique_ptr<Operation>> get_fusionopfixedshape_operations() {
#define DEFINE_FUNC(dtype) \
  operations.emplace_back( \
      std::unique_ptr<Operation>(new FusionOpFixedShape<tfcc::runtime::common::FLOAT>()));

  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace fusion
}  // namespace runtime
}  // namespace tfcc
