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

#include "onehot.h"

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/nn.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace nn {

template <class T, class IDX>
static const char* _wrapper_one_hot(
    const tfcc::Tensor<IDX>* indices, unsigned depth, T offValue, T onValue,
    tfcc::Variable<T>* result) {
  try {
    std::vector<unsigned> ids;
    for (int64_t idx : tfcc::data::get(*indices)) {
      ids.push_back(idx);
    }
    *result = tfcc::nn::one_hot(ids, indices->shape(), depth, offValue, onValue);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType IDX_DTYPE>
std::string OneHot<DTYPE, IDX_DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::nn::OneHot operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType IDX_DTYPE>
std::set<unsigned> OneHot<DTYPE, IDX_DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::nn::OneHot::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType IDX_DTYPE>
std::unique_ptr<OperationResource> OneHot<DTYPE, IDX_DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 1 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::nn::OneHot>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  if (symbolManager.getSymbolInfo(node.outputs(0)).dtype != DTYPE) {
    return nullptr;
  }
  if (symbolManager.getSymbolInfo(node.inputs(0)).dtype != IDX_DTYPE) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource());
  tfcc::runtime::operations::nn::OneHot operation;
  node.operation().UnpackTo(&operation);

  if (operation.data_type() != DTYPE) {
    return nullptr;
  }

  T offValue;
  if (!get_value(operation.off_value(), offValue)) {
    return nullptr;
  }
  T onValue;
  if (!get_value(operation.on_value(), onValue)) {
    return nullptr;
  }

  unsigned depth = operation.depth();

  auto& jit = getJIT(graph);

  const tfcc::Tensor<IDX>* inputSymbol = symbolManager.getTensor(node.inputs(0), IDX());
  tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());

  callFunction(jit, _wrapper_one_hot<T, IDX>, inputSymbol, depth, offValue, onValue, outputSymbol);
  return resource;
}

std::vector<std::unique_ptr<Operation>> get_one_hot_operations() {
#define DEFINE_FUNC(dtype)                                                             \
  operations.emplace_back(                                                             \
      std::unique_ptr<Operation>(new OneHot<dtype, tfcc::runtime::common::INT8>()));   \
  operations.emplace_back(                                                             \
      std::unique_ptr<Operation>(new OneHot<dtype, tfcc::runtime::common::UINT8>()));  \
  operations.emplace_back(                                                             \
      std::unique_ptr<Operation>(new OneHot<dtype, tfcc::runtime::common::INT16>()));  \
  operations.emplace_back(                                                             \
      std::unique_ptr<Operation>(new OneHot<dtype, tfcc::runtime::common::UINT16>())); \
  operations.emplace_back(                                                             \
      std::unique_ptr<Operation>(new OneHot<dtype, tfcc::runtime::common::INT32>()));  \
  operations.emplace_back(                                                             \
      std::unique_ptr<Operation>(new OneHot<dtype, tfcc::runtime::common::UINT32>())); \
  operations.emplace_back(                                                             \
      std::unique_ptr<Operation>(new OneHot<dtype, tfcc::runtime::common::INT64>()));  \
  operations.emplace_back(                                                             \
      std::unique_ptr<Operation>(new OneHot<dtype, tfcc::runtime::common::UINT64>()));

  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace nn
}  // namespace runtime
}  // namespace tfcc
