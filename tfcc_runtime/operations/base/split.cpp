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

#include "split.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/base.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace base {

template <class T>
static const char* _wrapper_split(
    const tfcc::Tensor<T>* a, size_t axis, std::vector<tfcc::Variable<T>*>* result) {
  try {
    auto tmp = tfcc::base::split(*a, result->size(), axis);
    for (size_t i = 0; i < tmp.size(); ++i) {
      *((*result)[i]) = std::move(tmp[i]);
    }
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE>
std::string Split<DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::base::Split operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE>
std::set<unsigned> Split<DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::base::Split::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE>
std::unique_ptr<OperationResource> Split<DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 1 || node.outputs_size() == 0) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::base::Split>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  if (symbolManager.getSymbolInfo(node.inputs(0)).dtype != DTYPE) {
    return nullptr;
  }
  for (const std::string& name : node.outputs()) {
    if (symbolManager.getSymbolInfo(name).dtype != DTYPE) {
      return nullptr;
    }
  }

  std::unique_ptr<SplitResource<T>> resource(new SplitResource<T>());

  tfcc::runtime::operations::base::Split operation;
  node.operation().UnpackTo(&operation);

  const tfcc::Tensor<T>* inputSymbol = symbolManager.getTensor(node.inputs(0), T());
  for (const std::string& name : node.outputs()) {
    tfcc::Variable<T>* symbol = symbolManager.getVariable(name, T());
    resource->symbols.push_back(symbol);
  }

  auto& jit = getJIT(graph);
  callFunction(jit, _wrapper_split<T>, inputSymbol, operation.axis(), &resource->symbols);
  return std::unique_ptr<OperationResource>(std::move(resource));
}

std::vector<std::unique_ptr<Operation>> get_split_operations() {
#define DEFINE_FUNC(dtype) operations.emplace_back(std::unique_ptr<Operation>(new Split<dtype>));
  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace base
}  // namespace runtime
}  // namespace tfcc
