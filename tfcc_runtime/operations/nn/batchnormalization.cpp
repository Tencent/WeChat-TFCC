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

#include "batchnormalization.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/nn.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace nn {

template <class T>
static const char* _wrapper_batch_normalization(
    const tfcc::Tensor<T>* a, size_t axis, const tfcc::Tensor<T>* scale,
    const tfcc::Tensor<T>* offset, const tfcc::Tensor<T>* mean, const tfcc::Tensor<T>* var,
    T epsilon, tfcc::Variable<T>* result) noexcept {
  try {
    *result = tfcc::nn::batch_normalization(*a, axis, *scale, *offset, *mean, *var, epsilon);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE>
std::string BatchNormalization<DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::nn::BatchNormalization operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE>
std::set<unsigned> BatchNormalization<DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {
      tfcc::runtime::operations::nn::BatchNormalization::VERSION_1,
      tfcc::runtime::operations::nn::BatchNormalization::VERSION_2};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE>
std::unique_ptr<OperationResource> BatchNormalization<DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 5 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::nn::BatchNormalization>()) {
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
  tfcc::runtime::operations::nn::BatchNormalization operation;
  node.operation().UnpackTo(&operation);
  size_t axis = 1;
  if (node.version() != tfcc::runtime::operations::nn::BatchNormalization::VERSION_1) {
    axis = operation.axis();
  }

  auto& jit = getJIT(graph);

  const tfcc::Tensor<T>* a = symbolManager.getTensor(node.inputs(0), T());
  const tfcc::Tensor<T>* scale = symbolManager.getTensor(node.inputs(1), T());
  const tfcc::Tensor<T>* offset = symbolManager.getTensor(node.inputs(2), T());
  const tfcc::Tensor<T>* mean = symbolManager.getTensor(node.inputs(3), T());
  const tfcc::Tensor<T>* var = symbolManager.getTensor(node.inputs(4), T());
  tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());
  T epsilon = static_cast<T>(0);
  if (!get_value(operation.epsilon(), epsilon)) {
    return nullptr;
  }

  callFunction(
      jit, _wrapper_batch_normalization<T>, a, axis, scale, offset, mean, var, epsilon,
      outputSymbol);
  return resource;
}

std::vector<std::unique_ptr<Operation>> get_batch_normalization_operations() {
#define DEFINE_FUNC(dtype) \
  operations.emplace_back(std::unique_ptr<Operation>(new BatchNormalization<dtype>));
  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace nn
}  // namespace runtime
}  // namespace tfcc
