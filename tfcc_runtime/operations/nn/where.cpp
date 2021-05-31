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

#include "where.h"

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/nn.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace nn {

template <class T>
static const char* _wrapper_where_v1(
    const tfcc::Tensor<uint8_t>* condition, const tfcc::Tensor<T>* x, const tfcc::Tensor<T>* y,
    tfcc::Variable<T>* result) noexcept {
  try {
    *result = tfcc::nn::where(*condition, *x, *y);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <class T>
static const char* _wrapper_where_v2(
    const tfcc::Tensor<uint8_t>* condition, const T* x, const tfcc::Tensor<T>* y,
    tfcc::Variable<T>* result) noexcept {
  try {
    *result = tfcc::nn::where(*condition, *x, *y);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <class T>
static const char* _wrapper_where_v3(
    const tfcc::Tensor<uint8_t>* condition, const tfcc::Tensor<T>* x, const T* y,
    tfcc::Variable<T>* result) noexcept {
  try {
    *result = tfcc::nn::where(*condition, *x, *y);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE>
std::string Where<DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::nn::Where operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE>
std::set<unsigned> Where<DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::nn::Where::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE>
std::unique_ptr<OperationResource> Where<DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 3 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::nn::Where>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  const SymbolInfo& xInfo = symbolManager.getSymbolInfo(node.inputs(1));
  const SymbolInfo& yInfo = symbolManager.getSymbolInfo(node.inputs(2));
  if (symbolManager.getSymbolInfo(node.outputs(0)).dtype != DTYPE) {
    return nullptr;
  }
  if (symbolManager.getSymbolInfo(node.inputs(0)).dtype != tfcc::runtime::common::BOOL) {
    return nullptr;
  }
  if (xInfo.dtype != DTYPE || yInfo.dtype != DTYPE) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource());
  tfcc::runtime::operations::nn::Where operation;
  node.operation().UnpackTo(&operation);

  auto& jit = getJIT(graph);

  const tfcc::Tensor<uint8_t>* conditionSymbol = symbolManager.getTensor(node.inputs(0), uint8_t());
  tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());
  if (is_tensor(xInfo.stype) && is_tensor(yInfo.stype)) {
    const tfcc::Tensor<T>* xSymbol = symbolManager.getTensor(node.inputs(1), T());
    const tfcc::Tensor<T>* ySymbol = symbolManager.getTensor(node.inputs(2), T());

    callFunction(jit, _wrapper_where_v1<T>, conditionSymbol, xSymbol, ySymbol, outputSymbol);
  } else if (is_value(xInfo.stype) && is_tensor(yInfo.stype)) {
    const T* xSymbol = symbolManager.getValue(node.inputs(1), T());
    const tfcc::Tensor<T>* ySymbol = symbolManager.getTensor(node.inputs(2), T());

    callFunction(jit, _wrapper_where_v2<T>, conditionSymbol, xSymbol, ySymbol, outputSymbol);
  } else if (is_tensor(xInfo.stype) && is_value(yInfo.stype)) {
    const tfcc::Tensor<T>* xSymbol = symbolManager.getTensor(node.inputs(1), T());
    const T* ySymbol = symbolManager.getValue(node.inputs(2), T());

    callFunction(jit, _wrapper_where_v3<T>, conditionSymbol, xSymbol, ySymbol, outputSymbol);
  } else {
    // invalid stype
    return nullptr;
  }

  return resource;
}

std::vector<std::unique_ptr<Operation>> get_where_operations() {
#define DEFINE_FUNC(dtype) operations.emplace_back(std::unique_ptr<Operation>(new Where<dtype>()));
  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace nn
}  // namespace runtime
}  // namespace tfcc
