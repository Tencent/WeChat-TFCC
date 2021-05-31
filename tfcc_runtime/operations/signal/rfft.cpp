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

#include "rfft.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/signal.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace signal {

template <class T>
static const char* _wrapper_rfft(
    const tfcc::Tensor<T>* a, unsigned length, tfcc::Variable<tfcc::Complex<T>>* result) noexcept {
  try {
    *result = tfcc::signal::rfft(*a, length);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE>
std::string RFFT<DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::signal::RFFT operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE>
std::set<unsigned> RFFT<DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::signal::RFFT::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE>
std::unique_ptr<OperationResource> RFFT<DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 1 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::signal::RFFT>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  if (symbolManager.getSymbolInfo(node.inputs(0)).dtype != DTYPE) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource);
  tfcc::runtime::operations::signal::RFFT operation;
  node.operation().UnpackTo(&operation);

  auto& jit = getJIT(graph);

  const tfcc::Tensor<T>* a = symbolManager.getTensor(node.inputs(0), T());
  tfcc::Variable<tfcc::Complex<T>>* outputSymbol =
      symbolManager.getVariable(node.outputs(0), tfcc::Complex<T>());

  callFunction(jit, _wrapper_rfft<T>, a, operation.length(), outputSymbol);
  return resource;
}

std::vector<std::unique_ptr<Operation>> get_rfft_operations() {
  std::vector<std::unique_ptr<Operation>> operations;
  operations.emplace_back(
      std::unique_ptr<Operation>(new RFFT<tfcc::runtime::common::DataType::FLOAT>));
  return operations;
}

}  // namespace signal
}  // namespace runtime
}  // namespace tfcc
