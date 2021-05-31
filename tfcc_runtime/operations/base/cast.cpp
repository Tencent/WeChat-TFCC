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

#include "cast.h"

#include <algorithm>
#include <type_traits>

#include "tfcc.h"

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/base.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace base {

// cast to other
template <tfcc::runtime::common::DataType SRC_DTYPE, tfcc::runtime::common::DataType DST_DTYPE>
static typename std::enable_if<DST_DTYPE != tfcc::runtime::common::BOOL, const char*>::type
_wrapper_cast_v1(
    const tfcc::Tensor<typename DataTypeInfo<SRC_DTYPE>::CPPType>* a,
    tfcc::Variable<typename DataTypeInfo<DST_DTYPE>::CPPType>* result) noexcept {
  using DST = typename DataTypeInfo<DST_DTYPE>::CPPType;
  try {
    *result = tfcc::data::cast<DST>(*a);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType SRC_DTYPE, tfcc::runtime::common::DataType DST_DTYPE>
static typename std::enable_if<DST_DTYPE != tfcc::runtime::common::BOOL, const char*>::type
_wrapper_cast_v2(
    const typename DataTypeInfo<SRC_DTYPE>::CPPType* a,
    typename DataTypeInfo<DST_DTYPE>::CPPType* result) noexcept {
  using DST = typename DataTypeInfo<DST_DTYPE>::CPPType;
  *result = static_cast<DST>(*a);
  return nullptr;
}

// cast to boolean
template <tfcc::runtime::common::DataType SRC_DTYPE, tfcc::runtime::common::DataType DST_DTYPE>
static typename std::enable_if<DST_DTYPE == tfcc::runtime::common::BOOL, const char*>::type
_wrapper_cast_v1(
    const tfcc::Tensor<typename DataTypeInfo<SRC_DTYPE>::CPPType>* a,
    tfcc::Variable<uint8_t>* result) noexcept {
  try {
    *result = tfcc::data::cast_to_boolean(*a);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType SRC_DTYPE, tfcc::runtime::common::DataType DST_DTYPE>
static typename std::enable_if<DST_DTYPE == tfcc::runtime::common::BOOL, const char*>::type
_wrapper_cast_v2(const typename DataTypeInfo<SRC_DTYPE>::CPPType* a, uint8_t* result) noexcept {
  *result = (*a) ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
  return nullptr;
}

template <tfcc::runtime::common::DataType SRC_DTYPE, tfcc::runtime::common::DataType DST_DTYPE>
std::string Cast<SRC_DTYPE, DST_DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::base::Cast operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType SRC_DTYPE, tfcc::runtime::common::DataType DST_DTYPE>
std::set<unsigned> Cast<SRC_DTYPE, DST_DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::base::Cast::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType SRC_DTYPE, tfcc::runtime::common::DataType DST_DTYPE>
std::unique_ptr<OperationResource> Cast<SRC_DTYPE, DST_DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 1 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::base::Cast>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  const SymbolInfo& inputInfo = symbolManager.getSymbolInfo(node.inputs(0));
  const SymbolInfo& outputInfo = symbolManager.getSymbolInfo(node.outputs(0));
  if (outputInfo.dtype != DST_DTYPE || inputInfo.dtype != SRC_DTYPE) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource);
  tfcc::runtime::operations::base::Cast operation;
  node.operation().UnpackTo(&operation);
  if (operation.data_type() != DST_DTYPE) {
    return nullptr;
  }

  auto& jit = getJIT(graph);

  if (is_tensor(inputInfo.stype) && is_tensor(outputInfo.stype)) {
    const tfcc::Tensor<SRC>* inputSymbol = symbolManager.getTensor(node.inputs(0), SRC());
    tfcc::Variable<DST>* outputSymbol = symbolManager.getVariable(node.outputs(0), DST());
    callFunction(jit, _wrapper_cast_v1<SRC_DTYPE, DST_DTYPE>, inputSymbol, outputSymbol);
  } else if (is_value(inputInfo.stype) && is_value(outputInfo.stype)) {
    const SRC* inputSymbol = symbolManager.getValue(node.inputs(0), SRC());
    DST* outputSymbol = symbolManager.getValue(node.outputs(0), DST());
    callFunction(jit, _wrapper_cast_v2<SRC_DTYPE, DST_DTYPE>, inputSymbol, outputSymbol);
  } else {
    // stype error
    return nullptr;
  }

  return resource;
}

std::vector<std::unique_ptr<Operation>> get_cast_operations() {
#define DEFINE_FUNC(src_dtype, dst_dtype) \
  operations.emplace_back(std::unique_ptr<Operation>(new Cast<src_dtype, dst_dtype>()));
  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE_ARG2(DEFINE_FUNC);
  return operations;
}

}  // namespace base
}  // namespace runtime
}  // namespace tfcc
