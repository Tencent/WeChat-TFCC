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

#include "flatten.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/base.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace base {

template <class T>
static const char* _wrapper_flatten(
    const tfcc::Tensor<T>* a, size_t axis, tfcc::View<T>* result) noexcept {
  if (axis >= a->shape().size()) {
    return "Flatten: axis error";
  }
  try {
    unsigned s1 = 1;
    unsigned s2 = 1;
    size_t i = 0;
    for (; i < axis; ++i) {
      s1 *= a->shape(i);
    }
    for (; i < a->shape().size(); ++i) {
      s2 *= a->shape(i);
    }
    *result = tfcc::View<T>(*a, {s1, s2});
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE>
std::string Flatten<DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::base::Flatten operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE>
std::set<unsigned> Flatten<DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::base::Flatten::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE>
std::unique_ptr<OperationResource> Flatten<DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 1 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::base::Flatten>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  if (symbolManager.getSymbolInfo(node.outputs(0)).dtype != DTYPE ||
      symbolManager.getSymbolInfo(node.inputs(0)).dtype != DTYPE) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource);
  tfcc::runtime::operations::base::Flatten operation;
  node.operation().UnpackTo(&operation);

  auto& jit = getJIT(graph);

  const tfcc::Tensor<T>* inputSymbol = symbolManager.getTensor(node.inputs(0), T());
  tfcc::View<T>* outputSymbol = symbolManager.getView(node.outputs(0), T());

  callFunction(jit, _wrapper_flatten<T>, inputSymbol, operation.axis(), outputSymbol);

  return resource;
}

std::vector<std::unique_ptr<Operation>> get_flatten_operations() {
#define DEFINE_FUNC(dtype) \
  operations.emplace_back(std::unique_ptr<Operation>(new Flatten<dtype>()));
  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace base
}  // namespace runtime
}  // namespace tfcc
