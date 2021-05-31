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

#include "irfft.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/signal.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace signal {

template <class T>
static const char* _wrapper_irfft(
    const tfcc::Tensor<tfcc::Complex<T>>* a, unsigned length, tfcc::Variable<T>* result) noexcept {
  try {
    *result = tfcc::signal::irfft(*a, length);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE>
std::string IRFFT<DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::signal::IRFFT operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE>
std::set<unsigned> IRFFT<DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::signal::IRFFT::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE>
std::unique_ptr<OperationResource> IRFFT<DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 1 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::signal::IRFFT>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  if (symbolManager.getSymbolInfo(node.outputs(0)).dtype != DTYPE) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource);
  tfcc::runtime::operations::signal::IRFFT operation;
  node.operation().UnpackTo(&operation);

  auto& jit = getJIT(graph);

  const tfcc::Tensor<tfcc::Complex<T>>* a =
      symbolManager.getTensor(node.inputs(0), tfcc::Complex<T>());
  tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());

  callFunction(jit, _wrapper_irfft<T>, a, operation.length(), outputSymbol);
  return resource;
}

std::vector<std::unique_ptr<Operation>> get_irfft_operations() {
  std::vector<std::unique_ptr<Operation>> operations;
  operations.emplace_back(
      std::unique_ptr<Operation>(new IRFFT<tfcc::runtime::common::DataType::FLOAT>));
  return operations;
}

}  // namespace signal
}  // namespace runtime
}  // namespace tfcc
