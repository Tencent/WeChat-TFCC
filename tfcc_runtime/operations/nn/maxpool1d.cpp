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

#include "maxpool1d.h"

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/nn.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace nn {

template <class T>
static const char* _wrapper_max_pool1d_ncw(
    const tfcc::Tensor<T>* a, unsigned kernel, unsigned padding, unsigned stride,
    tfcc::Variable<T>* result) noexcept {
  try {
    *result = tfcc::nn::max_pool1d(*a, false, kernel, padding, stride);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <class T>
static const char* _wrapper_max_pool1d_nwc(
    const tfcc::Tensor<T>* a, unsigned kernel, unsigned padding, unsigned stride,
    tfcc::Variable<T>* result) noexcept {
  try {
    *result = tfcc::nn::max_pool1d(*a, true, kernel, padding, stride);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE>
std::string MaxPool1D<DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::nn::MaxPool1D operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE>
std::set<unsigned> MaxPool1D<DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {
      tfcc::runtime::operations::nn::MaxPool1D::VERSION_1,
      tfcc::runtime::operations::nn::MaxPool1D::VERSION_2};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE>
std::unique_ptr<OperationResource> MaxPool1D<DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 1 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::nn::MaxPool1D>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  if (symbolManager.getSymbolInfo(node.outputs(0)).dtype != DTYPE ||
      symbolManager.getSymbolInfo(node.inputs(0)).dtype != DTYPE) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource());
  tfcc::runtime::operations::nn::MaxPool1D operation;
  node.operation().UnpackTo(&operation);

  bool ncw = true;
  if (node.version() != tfcc::runtime::operations::nn::MaxPool1D::VERSION_1) {
    ncw = operation.ncw();
  }

  auto& jit = getJIT(graph);

  const tfcc::Tensor<T>* inputSymbol = symbolManager.getTensor(node.inputs(0), T());
  tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());

  if (ncw) {
    callFunction(
        jit, _wrapper_max_pool1d_ncw<T>, inputSymbol, operation.kernel(), operation.padding(),
        operation.stride(), outputSymbol);
  } else {
    callFunction(
        jit, _wrapper_max_pool1d_nwc<T>, inputSymbol, operation.kernel(), operation.padding(),
        operation.stride(), outputSymbol);
  }
  return resource;
}

std::vector<std::unique_ptr<Operation>> get_max_pool1d_operations() {
#define DEFINE_FUNC(dtype) \
  operations.emplace_back(std::unique_ptr<Operation>(new MaxPool1D<dtype>()));
  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace nn
}  // namespace runtime
}  // namespace tfcc
