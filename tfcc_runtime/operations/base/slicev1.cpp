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

#include "slicev1.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/base.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace base {

template <class T, class POS>
static const char* _wrapper_slice_v1(
    const tfcc::Tensor<T>* a, const POS* pos, size_t axis, unsigned length,
    tfcc::Variable<T>* result) noexcept {
  if (*pos < 0) {
    return "SliceV1: invalid pos";
  }
  try {
    *result = tfcc::base::slice(
        *a, axis, static_cast<unsigned>(*pos), static_cast<unsigned>(*pos) + length);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType POS_DTYPE>
std::string SliceV1<DTYPE, POS_DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::base::SliceV1 operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType POS_DTYPE>
std::set<unsigned> SliceV1<DTYPE, POS_DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::base::SliceV1::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType POS_DTYPE>
std::unique_ptr<OperationResource> SliceV1<DTYPE, POS_DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 2 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::base::SliceV1>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  if (symbolManager.getSymbolInfo(node.outputs(0)).dtype != DTYPE) {
    return nullptr;
  }
  if (symbolManager.getSymbolInfo(node.inputs(0)).dtype != DTYPE) {
    return nullptr;
  }
  if (symbolManager.getSymbolInfo(node.inputs(1)).dtype != POS_DTYPE) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource);
  tfcc::runtime::operations::base::SliceV1 operation;
  node.operation().UnpackTo(&operation);

  auto& jit = getJIT(graph);

  const tfcc::Tensor<T>* inputSymbol = symbolManager.getTensor(node.inputs(0), T());
  const POS* posSymbol = symbolManager.getValue(node.inputs(1), POS());
  tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());

  callFunction(
      jit, _wrapper_slice_v1<T, POS>, inputSymbol, posSymbol, operation.axis(), operation.length(),
      outputSymbol);

  return resource;
}

std::vector<std::unique_ptr<Operation>> get_slice_v1_operations() {
#define DEFINE_FUNC(dtype)                                                              \
  operations.emplace_back(                                                              \
      std::unique_ptr<Operation>(new SliceV1<dtype, tfcc::runtime::common::UINT8>()));  \
  operations.emplace_back(                                                              \
      std::unique_ptr<Operation>(new SliceV1<dtype, tfcc::runtime::common::INT8>()));   \
  operations.emplace_back(                                                              \
      std::unique_ptr<Operation>(new SliceV1<dtype, tfcc::runtime::common::UINT16>())); \
  operations.emplace_back(                                                              \
      std::unique_ptr<Operation>(new SliceV1<dtype, tfcc::runtime::common::INT16>()));  \
  operations.emplace_back(                                                              \
      std::unique_ptr<Operation>(new SliceV1<dtype, tfcc::runtime::common::UINT32>())); \
  operations.emplace_back(                                                              \
      std::unique_ptr<Operation>(new SliceV1<dtype, tfcc::runtime::common::INT32>()));  \
  operations.emplace_back(                                                              \
      std::unique_ptr<Operation>(new SliceV1<dtype, tfcc::runtime::common::UINT64>())); \
  operations.emplace_back(                                                              \
      std::unique_ptr<Operation>(new SliceV1<dtype, tfcc::runtime::common::INT64>()));

  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace base
}  // namespace runtime
}  // namespace tfcc
