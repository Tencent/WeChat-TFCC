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

#include "squeeze.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/base.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace base {

template <class T>
static const char* _wrapper_squeeze(
    const tfcc::Tensor<T>* a, const std::vector<size_t>* axes, tfcc::View<T>* result) noexcept {
  try {
    std::vector<unsigned> shape;
    size_t pos = 0;
    for (size_t i = 0; i < a->shape().size(); ++i) {
      if (pos < axes->size() && i == (*axes)[pos]) {
        if (a->shape(i) != 1) {
          return "Squeeze: invalid axes";
        }
        ++pos;
        continue;
      }
      shape.push_back(a->shape(i));
    }
    if (pos != axes->size()) {
      return "Squeeze: invalid axes";
    }
    *result = tfcc::View<T>(*a, shape);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <class T>
static const char* _wrapper_set_view(const tfcc::Tensor<T>* a, tfcc::View<T>* result) noexcept {
  try {
    *result = tfcc::View<T>(*a);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE>
std::string Squeeze<DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::base::Squeeze operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE>
std::set<unsigned> Squeeze<DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::base::Squeeze::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE>
std::unique_ptr<OperationResource> Squeeze<DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 1 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::base::Squeeze>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  if (symbolManager.getSymbolInfo(node.outputs(0)).dtype != DTYPE ||
      symbolManager.getSymbolInfo(node.inputs(0)).dtype != DTYPE) {
    return nullptr;
  }

  std::unique_ptr<SqueezeResource> resource(new SqueezeResource);
  tfcc::runtime::operations::base::Squeeze operation;
  node.operation().UnpackTo(&operation);
  for (unsigned axis : operation.axes()) {
    resource->axes.push_back(axis);
  }

  std::sort(resource->axes.begin(), resource->axes.end());

  auto& jit = getJIT(graph);

  const tfcc::Tensor<T>* inputSymbol = symbolManager.getTensor(node.inputs(0), T());
  tfcc::View<T>* outputSymbol = symbolManager.getView(node.outputs(0), T());

  if (resource->axes.size() > 0) {
    callFunction(jit, _wrapper_squeeze<T>, inputSymbol, &resource->axes, outputSymbol);
  } else {
    callFunction(jit, _wrapper_set_view<T>, inputSymbol, outputSymbol);
  }

  return std::unique_ptr<OperationResource>(std::move(resource));
}

std::vector<std::unique_ptr<Operation>> get_squeeze_operations() {
#define DEFINE_FUNC(dtype) \
  operations.emplace_back(std::unique_ptr<Operation>(new Squeeze<dtype>()));
  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace base
}  // namespace runtime
}  // namespace tfcc
