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

#include "slicev2.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/base.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace base {

template <class T, class POS>
static const char* _wrapper_slice_v2_signed(
    const tfcc::Tensor<T>* a, const POS* start, const POS* end, size_t axis,
    tfcc::Variable<T>* result) noexcept {
  try {
    if (axis > a->shape().size()) {
      return "SliceV2: invalid axis";
    }
    POS realStart = *start;
    POS realEnd = *end;
    while (realStart < 0) {
      realStart += static_cast<int64_t>(a->shape(axis));
    }
    while (realEnd < 0) {
      realEnd += static_cast<int64_t>(a->shape(axis));
    }
    *result = tfcc::base::slice(
        *a, axis, static_cast<unsigned>(realStart), static_cast<unsigned>(realEnd));
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <class T, class POS>
static const char* _wrapper_slice_v2_unsigned(
    const tfcc::Tensor<T>* a, const POS* start, const POS* end, size_t axis,
    tfcc::Variable<T>* result) noexcept {
  try {
    *result =
        tfcc::base::slice(*a, axis, static_cast<unsigned>(*start), static_cast<unsigned>(*end));
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType POS_DTYPE>
std::string SliceV2<DTYPE, POS_DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::base::SliceV2 operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType POS_DTYPE>
std::set<unsigned> SliceV2<DTYPE, POS_DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::base::SliceV2::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType POS_DTYPE>
std::unique_ptr<OperationResource> SliceV2<DTYPE, POS_DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 3 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::base::SliceV2>()) {
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
  if (symbolManager.getSymbolInfo(node.inputs(2)).dtype != POS_DTYPE) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource);
  tfcc::runtime::operations::base::SliceV2 operation;
  node.operation().UnpackTo(&operation);

  auto& jit = getJIT(graph);

  const tfcc::Tensor<T>* inputSymbol = symbolManager.getTensor(node.inputs(0), T());
  const POS* startSymbol = symbolManager.getValue(node.inputs(1), POS());
  const POS* endSymbol = symbolManager.getValue(node.inputs(2), POS());
  tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());

  if (is_signed(POS_DTYPE)) {
    callFunction(
        jit, _wrapper_slice_v2_signed<T, POS>, inputSymbol, startSymbol, endSymbol,
        operation.axis(), outputSymbol);
  } else {
    callFunction(
        jit, _wrapper_slice_v2_unsigned<T, POS>, inputSymbol, startSymbol, endSymbol,
        operation.axis(), outputSymbol);
  }

  return resource;
}

std::vector<std::unique_ptr<Operation>> get_slice_v2_operations() {
#define DEFINE_FUNC(dtype)                                                              \
  operations.emplace_back(                                                              \
      std::unique_ptr<Operation>(new SliceV2<dtype, tfcc::runtime::common::UINT8>()));  \
  operations.emplace_back(                                                              \
      std::unique_ptr<Operation>(new SliceV2<dtype, tfcc::runtime::common::INT8>()));   \
  operations.emplace_back(                                                              \
      std::unique_ptr<Operation>(new SliceV2<dtype, tfcc::runtime::common::UINT16>())); \
  operations.emplace_back(                                                              \
      std::unique_ptr<Operation>(new SliceV2<dtype, tfcc::runtime::common::INT16>()));  \
  operations.emplace_back(                                                              \
      std::unique_ptr<Operation>(new SliceV2<dtype, tfcc::runtime::common::UINT32>())); \
  operations.emplace_back(                                                              \
      std::unique_ptr<Operation>(new SliceV2<dtype, tfcc::runtime::common::INT32>()));  \
  operations.emplace_back(                                                              \
      std::unique_ptr<Operation>(new SliceV2<dtype, tfcc::runtime::common::UINT64>())); \
  operations.emplace_back(                                                              \
      std::unique_ptr<Operation>(new SliceV2<dtype, tfcc::runtime::common::INT64>()));

  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace base
}  // namespace runtime
}  // namespace tfcc
