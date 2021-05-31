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

#include "at.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/base.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace base {

template <class T, class IDX>
static const char* _wrapper_at(const std::vector<T>* a, const IDX* pos, T* result) noexcept {
  if (*pos < 0) {
    return "At: invalid pos";
  }
  if (static_cast<size_t>(*pos) >= a->size()) {
    return "At: invalid pos";
  }
  *result = (*a)[*pos];
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType IDX_DTYPE>
std::string At<DTYPE, IDX_DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::base::At operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType IDX_DTYPE>
std::set<unsigned> At<DTYPE, IDX_DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::base::At::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType IDX_DTYPE>
std::unique_ptr<OperationResource> At<DTYPE, IDX_DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 2 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::base::At>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  if (symbolManager.getSymbolInfo(node.outputs(0)).dtype != DTYPE) {
    return nullptr;
  }
  if (symbolManager.getSymbolInfo(node.inputs(0)).dtype != DTYPE ||
      symbolManager.getSymbolInfo(node.inputs(1)).dtype != IDX_DTYPE) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource);
  tfcc::runtime::operations::base::At operation;
  node.operation().UnpackTo(&operation);

  auto& jit = getJIT(graph);

  const std::vector<T>* inputSymbol = symbolManager.getVector(node.inputs(0), T());
  const IDX* indicesSymbol = symbolManager.getValue(node.inputs(1), IDX());
  T* outputSymbol = symbolManager.getValue(node.outputs(0), T());

  callFunction(jit, _wrapper_at<T, IDX>, inputSymbol, indicesSymbol, outputSymbol);

  return resource;
}

std::vector<std::unique_ptr<Operation>> get_at_operations() {
#define DEFINE_FUNC(dtype)                                                         \
  operations.emplace_back(                                                         \
      std::unique_ptr<Operation>(new At<dtype, tfcc::runtime::common::INT8>()));   \
  operations.emplace_back(                                                         \
      std::unique_ptr<Operation>(new At<dtype, tfcc::runtime::common::UINT16>())); \
  operations.emplace_back(                                                         \
      std::unique_ptr<Operation>(new At<dtype, tfcc::runtime::common::INT16>()));  \
  operations.emplace_back(                                                         \
      std::unique_ptr<Operation>(new At<dtype, tfcc::runtime::common::UINT32>())); \
  operations.emplace_back(                                                         \
      std::unique_ptr<Operation>(new At<dtype, tfcc::runtime::common::INT32>()));  \
  operations.emplace_back(                                                         \
      std::unique_ptr<Operation>(new At<dtype, tfcc::runtime::common::UINT64>())); \
  operations.emplace_back(                                                         \
      std::unique_ptr<Operation>(new At<dtype, tfcc::runtime::common::INT64>()));  \
  operations.emplace_back(                                                         \
      std::unique_ptr<Operation>(new At<dtype, tfcc::runtime::common::UINT64>()));

  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace base
}  // namespace runtime
}  // namespace tfcc
