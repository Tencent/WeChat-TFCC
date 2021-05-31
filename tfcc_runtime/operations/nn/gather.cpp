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

#include "gather.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/nn.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace nn {

template <class T, class IDX>
static const char* _wrapper_gather(
    const tfcc::Tensor<T>* a, const tfcc::Tensor<IDX>* indices, size_t axis,
    tfcc::Variable<T>* result) noexcept {
  try {
    if (axis != 0) {
      return "Gather: invalid axis";
    }
    *result = tfcc::nn::gather(*a, *indices);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType IDX_DTYPE>
std::string Gather<DTYPE, IDX_DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::nn::Gather operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType IDX_DTYPE>
std::set<unsigned> Gather<DTYPE, IDX_DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::nn::Gather::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType IDX_DTYPE>
std::unique_ptr<OperationResource> Gather<DTYPE, IDX_DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 2 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::nn::Gather>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  if (symbolManager.getSymbolInfo(node.outputs(0)).dtype != DTYPE ||
      symbolManager.getSymbolInfo(node.inputs(0)).dtype != DTYPE) {
    return nullptr;
  }
  const SymbolInfo& indicesSymbolInfo = symbolManager.getSymbolInfo(node.inputs(1));
  if (indicesSymbolInfo.dtype != IDX_DTYPE) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource);
  tfcc::runtime::operations::nn::Gather operation;
  node.operation().UnpackTo(&operation);

  auto& jit = getJIT(graph);

  const tfcc::Tensor<T>* inputSymbol = symbolManager.getTensor(node.inputs(0), T());
  const tfcc::Tensor<IDX>* indicesSymbol = symbolManager.getTensor(node.inputs(1), IDX());
  tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());

  callFunction(
      jit, _wrapper_gather<T, IDX>, inputSymbol, indicesSymbol, operation.axis(), outputSymbol);

  return resource;
}

std::vector<std::unique_ptr<Operation>> get_gather_operations() {
#define DEFINE_FUNC(dtype)                                                             \
  operations.emplace_back(                                                             \
      std::unique_ptr<Operation>(new Gather<dtype, tfcc::runtime::common::INT32>()));  \
  operations.emplace_back(                                                             \
      std::unique_ptr<Operation>(new Gather<dtype, tfcc::runtime::common::UINT32>())); \
  operations.emplace_back(                                                             \
      std::unique_ptr<Operation>(new Gather<dtype, tfcc::runtime::common::INT64>()));  \
  operations.emplace_back(                                                             \
      std::unique_ptr<Operation>(new Gather<dtype, tfcc::runtime::common::UINT64>()));

  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace nn
}  // namespace runtime
}  // namespace tfcc
