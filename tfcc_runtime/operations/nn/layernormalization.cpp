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

#include "layernormalization.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/nn.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace nn {

template <class T>
static const char* _wrapper_layer_normalization(
    const tfcc::Tensor<T>* a, const tfcc::Tensor<T>* gamma, const tfcc::Tensor<T>* beta, T epsilon,
    tfcc::Variable<T>* result) noexcept {
  try {
    *result = tfcc::nn::layer_normalization(*a, *gamma, *beta, epsilon, a->shape().size() - 1);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE>
std::string LayerNormalization<DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::nn::LayerNormalization operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE>
std::set<unsigned> LayerNormalization<DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::nn::LayerNormalization::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE>
std::unique_ptr<OperationResource> LayerNormalization<DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 3 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::nn::LayerNormalization>()) {
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

  std::unique_ptr<OperationResource> resource(new OperationResource);
  tfcc::runtime::operations::nn::LayerNormalization operation;
  node.operation().UnpackTo(&operation);

  auto& jit = getJIT(graph);

  const tfcc::Tensor<T>* aSymbol = symbolManager.getTensor(node.inputs(0), T());
  const tfcc::Tensor<T>* gammaSymbol = symbolManager.getTensor(node.inputs(1), T());
  const tfcc::Tensor<T>* betaSymbol = symbolManager.getTensor(node.inputs(2), T());
  tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());
  T epsilon = static_cast<T>(0);
  if (!get_value(operation.epsilon(), epsilon)) {
    return nullptr;
  }

  callFunction(
      jit, _wrapper_layer_normalization<T>, aSymbol, gammaSymbol, betaSymbol, epsilon,
      outputSymbol);
  return resource;
}

std::vector<std::unique_ptr<Operation>> get_layer_normalization_operations() {
#define DEFINE_FUNC(dtype) \
  operations.emplace_back(std::unique_ptr<Operation>(new LayerNormalization<dtype>));
  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace nn
}  // namespace runtime
}  // namespace tfcc
