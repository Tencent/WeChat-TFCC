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

#include "reciprocal.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/math.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace math {

template <class T>
static const char* _wrapper_reciprocal_v1(
    const tfcc::Tensor<T>* a, tfcc::Variable<T>* result) noexcept {
  try {
    *result = static_cast<T>(1) / *a;
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <class T>
static const char* _wrapper_reciprocal_v2(const T* a, T* result) noexcept {
  try {
    *result = static_cast<T>(1) / *a;
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE>
std::string Reciprocal<DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::math::Reciprocal operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE>
std::set<unsigned> Reciprocal<DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::math::Reciprocal::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE>
std::unique_ptr<OperationResource> Reciprocal<DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 1 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::math::Reciprocal>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  const SymbolInfo& inputInfo = symbolManager.getSymbolInfo(node.outputs(0));
  const SymbolInfo& outputInfo = symbolManager.getSymbolInfo(node.inputs(0));
  if (inputInfo.dtype != DTYPE || outputInfo.dtype != DTYPE) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource);
  tfcc::runtime::operations::math::Reciprocal operation;
  node.operation().UnpackTo(&operation);

  auto& jit = getJIT(graph);

  if (is_tensor(inputInfo.stype) && is_tensor(outputInfo.stype)) {
    const tfcc::Tensor<T>* inputSymbol = symbolManager.getTensor(node.inputs(0), T());
    tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());
    callFunction(jit, _wrapper_reciprocal_v1<T>, inputSymbol, outputSymbol);
  } else if (is_value(inputInfo.stype) && is_value(outputInfo.stype)) {
    const T* inputSymbol = symbolManager.getValue(node.inputs(0), T());
    T* outputSymbol = symbolManager.getValue(node.outputs(0), T());
    callFunction(jit, _wrapper_reciprocal_v2<T>, inputSymbol, outputSymbol);
  } else {
    // stype error
    return nullptr;
  }

  return resource;
}

std::vector<std::unique_ptr<Operation>> get_reciprocal_operations() {
#define DEFINE_FUNC(dtype) \
  operations.emplace_back(std::unique_ptr<Operation>(new Reciprocal<dtype>()));
  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace math
}  // namespace runtime
}  // namespace tfcc
