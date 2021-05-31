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

#include "unequal.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/relation.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace relation {

template <class T>
static const char* _wrapper_unequal_v1(
    const tfcc::Tensor<T>* a, const tfcc::Tensor<T>* b, tfcc::Variable<uint8_t>* result) noexcept {
  try {
    *result = tfcc::relation::unequal(*a, *b);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <class T>
static const char* _wrapper_unequal_v2(
    const tfcc::Tensor<T>* a, const T* b, tfcc::Variable<uint8_t>* result) noexcept {
  try {
    *result = tfcc::relation::unequal(*a, *b);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <class T>
static const char* _wrapper_unequal_v3(
    const T* a, const tfcc::Tensor<T>* b, tfcc::Variable<uint8_t>* result) noexcept {
  try {
    *result = tfcc::relation::unequal(*a, *b);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <class T>
static const char* _wrapper_unequal_v4(const T* a, const T* b, uint8_t* result) noexcept {
  *result = *a == *b ? 0 : 1;
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE>
std::string UnEqual<DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::relation::UnEqual operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE>
std::set<unsigned> UnEqual<DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::relation::UnEqual::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE>
std::unique_ptr<OperationResource> UnEqual<DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 2 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::relation::UnEqual>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  const SymbolInfo& input0Info = symbolManager.getSymbolInfo(node.inputs(0));
  const SymbolInfo& input1Info = symbolManager.getSymbolInfo(node.inputs(1));
  const SymbolInfo& outputInfo = symbolManager.getSymbolInfo(node.outputs(0));
  if (outputInfo.dtype != tfcc::runtime::common::BOOL || input0Info.dtype != DTYPE ||
      input1Info.dtype != DTYPE) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource);
  tfcc::runtime::operations::relation::UnEqual operation;
  node.operation().UnpackTo(&operation);

  auto& jit = getJIT(graph);

  if (is_tensor(input0Info.stype) && is_tensor(input1Info.stype)) {
    // variable - variable
    const tfcc::Tensor<T>* input0Symbol = symbolManager.getTensor(node.inputs(0), T());
    const tfcc::Tensor<T>* input1Symbol = symbolManager.getTensor(node.inputs(1), T());
    tfcc::Variable<uint8_t>* outputSymbol = symbolManager.getVariable(node.outputs(0), uint8_t());
    callFunction(jit, _wrapper_unequal_v1<T>, input0Symbol, input1Symbol, outputSymbol);
  } else if (is_tensor(input0Info.stype) && is_value(input1Info.stype)) {
    // variable - value
    const tfcc::Tensor<T>* input0Symbol = symbolManager.getTensor(node.inputs(0), T());
    const T* input1Symbol = symbolManager.getValue(node.inputs(1), T());
    tfcc::Variable<uint8_t>* outputSymbol = symbolManager.getVariable(node.outputs(0), uint8_t());
    callFunction(jit, _wrapper_unequal_v2<T>, input0Symbol, input1Symbol, outputSymbol);
  } else if (is_value(input0Info.stype) && is_tensor(input1Info.stype)) {
    // variable - value
    const T* input0Symbol = symbolManager.getValue(node.inputs(0), T());
    const tfcc::Tensor<T>* input1Symbol = symbolManager.getTensor(node.inputs(1), T());
    tfcc::Variable<uint8_t>* outputSymbol = symbolManager.getVariable(node.outputs(0), uint8_t());
    callFunction(jit, _wrapper_unequal_v3<T>, input0Symbol, input1Symbol, outputSymbol);
  } else if (is_value(input0Info.stype) && is_value(input1Info.stype)) {
    // variable - value
    const T* input0Symbol = symbolManager.getValue(node.inputs(0), T());
    const T* input1Symbol = symbolManager.getValue(node.inputs(1), T());
    uint8_t* outputSymbol = symbolManager.getValue(node.outputs(0), uint8_t());
    callFunction(jit, _wrapper_unequal_v4<T>, input0Symbol, input1Symbol, outputSymbol);
  } else {
    // error
    return nullptr;
  }

  return resource;
}

std::vector<std::unique_ptr<Operation>> get_unequal_operations() {
#define DEFINE_FUNC(dtype) \
  operations.emplace_back(std::unique_ptr<Operation>(new UnEqual<dtype>()));
  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace relation
}  // namespace runtime
}  // namespace tfcc
