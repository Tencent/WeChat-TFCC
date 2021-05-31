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

#include "reduceprod.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/math.pb.h"
#include "tfcc_runtime/utils/commutils.h"
#include "tfcc_runtime/utils/reduceutils.h"

namespace tfcc {
namespace runtime {
namespace math {

template <class T>
static const char* _wrapper_reduce_prod(
    const tfcc::Tensor<T>* a, const std::vector<size_t>* axes, tfcc::Variable<T>* result) noexcept {
  try {
    std::vector<size_t> perm;
    tfcc::Shape targetShape;
    bool needTranspose;
    std::tie(perm, targetShape, needTranspose) =
        get_reduce_perm_and_target_shape(a->shape(), *axes);
    if (needTranspose) {
      *result = tfcc::base::transpose(*a, perm);
      *result = tfcc::math::reduce_prod(*result, perm.size() - axes->size());
      result->reshape(targetShape);
    } else {
      *result = tfcc::math::reduce_prod(*a, perm.size() - axes->size());
    }
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE>
std::string ReduceProd<DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::math::ReduceProd operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE>
std::set<unsigned> ReduceProd<DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::math::ReduceProd::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE>
std::unique_ptr<OperationResource> ReduceProd<DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 1 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::math::ReduceProd>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  if (symbolManager.getSymbolInfo(node.outputs(0)).dtype != DTYPE ||
      symbolManager.getSymbolInfo(node.inputs(0)).dtype != DTYPE) {
    return nullptr;
  }

  std::unique_ptr<ReduceProdResource> resource(new ReduceProdResource);
  tfcc::runtime::operations::math::ReduceProd operation;
  node.operation().UnpackTo(&operation);
  for (unsigned axis : operation.axes()) {
    resource->axes.push_back(axis);
  }

  std::sort(resource->axes.begin(), resource->axes.end());

  auto& jit = getJIT(graph);

  const tfcc::Tensor<T>* inputSymbol = symbolManager.getTensor(node.inputs(0), T());
  tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());

  callFunction(jit, _wrapper_reduce_prod<T>, inputSymbol, &resource->axes, outputSymbol);

  return std::unique_ptr<OperationResource>(std::move(resource));
}

std::vector<std::unique_ptr<Operation>> get_reduce_prod_operations() {
#define DEFINE_FUNC(dtype) \
  operations.emplace_back(std::unique_ptr<Operation>(new ReduceProd<dtype>()));
  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace math
}  // namespace runtime
}  // namespace tfcc
