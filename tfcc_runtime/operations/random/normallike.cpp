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

#include "normallike.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/random.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace random {

template <class T>
static const char* _wrapper_normal_like(
    const std::vector<uint32_t>* shape, T mean, T scale, tfcc::Variable<T>* result) noexcept {
  try {
    *result = tfcc::random::normal(tfcc::Shape(*shape), mean, scale);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE>
std::string NormalLike<DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::random::NormalLike operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE>
std::set<unsigned> NormalLike<DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::random::NormalLike::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE>
std::unique_ptr<OperationResource> NormalLike<DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 1 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::random::NormalLike>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  if (symbolManager.getSymbolInfo(node.outputs(0)).dtype != DTYPE) {
    return nullptr;
  }
  if (symbolManager.getSymbolInfo(node.inputs(0)).dtype != tfcc::runtime::common::UINT32) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource);
  tfcc::runtime::operations::random::NormalLike operation;
  node.operation().UnpackTo(&operation);

  auto& jit = getJIT(graph);

  const std::vector<uint32_t>* shape = symbolManager.getVector(node.inputs(0), uint32_t());
  tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());
  T mean = static_cast<T>(0);
  if (!get_value(operation.mean(), mean)) {
    return nullptr;
  }
  T scale = static_cast<T>(0);
  if (!get_value(operation.scale(), scale)) {
    return nullptr;
  }

  callFunction(jit, _wrapper_normal_like<T>, shape, mean, scale, outputSymbol);
  return resource;
}

std::vector<std::unique_ptr<Operation>> get_normal_like_operations() {
#define DEFINE_FUNC(dtype) \
  operations.emplace_back(std::unique_ptr<Operation>(new NormalLike<dtype>));
  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_FLOATING_POINT_DATA_TYPES(DEFINE_FUNC);
  return operations;
}

}  // namespace random
}  // namespace runtime
}  // namespace tfcc
