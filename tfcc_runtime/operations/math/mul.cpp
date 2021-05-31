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

#include "mul.h"

#include <algorithm>
#include <type_traits>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/math.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace math {

template <class T>
static const char* _wrapper_mul_v1(
    const tfcc::Tensor<T>* a, const tfcc::Tensor<T>* b, tfcc::Variable<T>* result) noexcept {
  try {
    *result = *a * *b;
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <class T>
static const char* _wrapper_mul_v2(
    const tfcc::Tensor<T>* a, const T* b, tfcc::Variable<T>* result) noexcept {
  try {
    *result = *a * *b;
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <class T>
static const char* _wrapper_mul_v3(
    const T* a, const tfcc::Tensor<T>* b, tfcc::Variable<T>* result) noexcept {
  try {
    *result = *a * *b;
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <class T>
static const char* _wrapper_mul_v4(const T* a, const T* b, T* result) noexcept {
  *result = *a * *b;
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE>
std::string Mul<DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::math::Mul operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE>
std::set<unsigned> Mul<DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::math::Mul::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE>
std::unique_ptr<OperationResource> Mul<DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 2 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::math::Mul>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  const SymbolInfo& input0Info = symbolManager.getSymbolInfo(node.inputs(0));
  const SymbolInfo& input1Info = symbolManager.getSymbolInfo(node.inputs(1));
  const SymbolInfo& outputInfo = symbolManager.getSymbolInfo(node.outputs(0));
  if (outputInfo.dtype != DTYPE || input0Info.dtype != DTYPE || input1Info.dtype != DTYPE) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource);
  tfcc::runtime::operations::math::Mul operation;
  node.operation().UnpackTo(&operation);

  auto& jit = getJIT(graph);

  if (is_tensor(input0Info.stype) && is_tensor(input1Info.stype)) {
    // variable - variable
    const tfcc::Tensor<T>* input0Symbol = symbolManager.getTensor(node.inputs(0), T());
    const tfcc::Tensor<T>* input1Symbol = symbolManager.getTensor(node.inputs(1), T());
    tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());
    callFunction(jit, _wrapper_mul_v1<T>, input0Symbol, input1Symbol, outputSymbol);
  } else if (is_tensor(input0Info.stype) && is_value(input1Info.stype)) {
    // variable - value
    const tfcc::Tensor<T>* input0Symbol = symbolManager.getTensor(node.inputs(0), T());
    const T* input1Symbol = symbolManager.getValue(node.inputs(1), T());
    tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());
    callFunction(jit, _wrapper_mul_v2<T>, input0Symbol, input1Symbol, outputSymbol);
  } else if (is_value(input0Info.stype) && is_tensor(input1Info.stype)) {
    // variable - value
    const T* input0Symbol = symbolManager.getValue(node.inputs(0), T());
    const tfcc::Tensor<T>* input1Symbol = symbolManager.getTensor(node.inputs(1), T());
    tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());
    callFunction(jit, _wrapper_mul_v3<T>, input0Symbol, input1Symbol, outputSymbol);
  } else if (is_value(input0Info.stype) && is_value(input1Info.stype)) {
    // variable - value
    const T* input0Symbol = symbolManager.getValue(node.inputs(0), T());
    const T* input1Symbol = symbolManager.getValue(node.inputs(1), T());
    T* outputSymbol = symbolManager.getValue(node.outputs(0), T());
    callFunction(jit, _wrapper_mul_v4<T>, input0Symbol, input1Symbol, outputSymbol);
  } else {
    // error
    return nullptr;
  }

  return resource;
}

template <>
std::unique_ptr<OperationResource> Mul<tfcc::runtime::common::COMPLEX64>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 2 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::math::Mul>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  const SymbolInfo& input0Info = symbolManager.getSymbolInfo(node.inputs(0));
  const SymbolInfo& input1Info = symbolManager.getSymbolInfo(node.inputs(1));
  const SymbolInfo& outputInfo = symbolManager.getSymbolInfo(node.outputs(0));
  if (outputInfo.dtype != tfcc::runtime::common::COMPLEX64) {
    return nullptr;
  }
  if (input0Info.dtype != tfcc::runtime::common::COMPLEX64 ||
      input1Info.dtype != tfcc::runtime::common::COMPLEX64) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource);
  tfcc::runtime::operations::math::Mul operation;
  node.operation().UnpackTo(&operation);

  auto& jit = getJIT(graph);

  if (is_tensor(input0Info.stype) && is_tensor(input1Info.stype)) {
    // variable - variable
    const tfcc::Tensor<T>* input0Symbol = symbolManager.getTensor(node.inputs(0), T());
    const tfcc::Tensor<T>* input1Symbol = symbolManager.getTensor(node.inputs(1), T());
    tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());
    callFunction(jit, _wrapper_mul_v1<T>, input0Symbol, input1Symbol, outputSymbol);
  } else {
    // error
    return nullptr;
  }

  return resource;
}

std::vector<std::unique_ptr<Operation>> get_mul_operations() {
#define DEFINE_FUNC(dtype) operations.emplace_back(std::unique_ptr<Operation>(new Mul<dtype>()));
  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  TFCC_RUNTIME_FOR_COMPLEX_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace math
}  // namespace runtime
}  // namespace tfcc
