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

#include "uniformlike.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/random.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace random {

template <class T>
static const char* _wrapper_uniform_like(
    const std::vector<uint32_t>* shape, T low, T high, tfcc::Variable<T>* result) noexcept {
  try {
    *result = tfcc::random::uniform(tfcc::Shape(*shape), low, high);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE>
std::string UniformLike<DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::random::UniformLike operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE>
std::set<unsigned> UniformLike<DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::random::UniformLike::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE>
std::unique_ptr<OperationResource> UniformLike<DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 1 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::random::UniformLike>()) {
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
  tfcc::runtime::operations::random::UniformLike operation;
  node.operation().UnpackTo(&operation);

  auto& jit = getJIT(graph);

  const std::vector<uint32_t>* shape = symbolManager.getVector(node.inputs(0), uint32_t());
  tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());
  T low = std::numeric_limits<T>::lowest();
  if (!get_value(operation.low(), low)) {
    return nullptr;
  }
  T high = std::numeric_limits<T>::max();
  if (!get_value(operation.high(), high)) {
    return nullptr;
  }

  callFunction(jit, _wrapper_uniform_like<T>, shape, low, high, outputSymbol);
  return resource;
}

std::vector<std::unique_ptr<Operation>> get_uniform_like_operations() {
#define DEFINE_FUNC(dtype) \
  operations.emplace_back(std::unique_ptr<Operation>(new UniformLike<dtype>));
  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_FLOATING_POINT_DATA_TYPES(DEFINE_FUNC);
  return operations;
}

}  // namespace random
}  // namespace runtime
}  // namespace tfcc
