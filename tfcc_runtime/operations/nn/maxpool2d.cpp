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

#include "maxpool2d.h"

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/nn.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace nn {

template <class T>
static const char* _wrapper_max_pool2d_nchw(
    const tfcc::Tensor<T>* a, unsigned kernelHeight, unsigned kernelWidth, unsigned paddingHeight,
    unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth,
    tfcc::Variable<T>* result) noexcept {
  try {
    *result = tfcc::nn::max_pool2d(
        *a, false, kernelHeight, kernelWidth, paddingHeight, paddingWidth, strideHeight,
        strideWidth);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <class T>
static const char* _wrapper_max_pool2d_nhwc(
    const tfcc::Tensor<T>* a, unsigned kernelHeight, unsigned kernelWidth, unsigned paddingHeight,
    unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth,
    tfcc::Variable<T>* result) noexcept {
  try {
    *result = tfcc::nn::max_pool2d(
        *a, true, kernelHeight, kernelWidth, paddingHeight, paddingWidth, strideHeight,
        strideWidth);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE>
std::string MaxPool2D<DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::nn::MaxPool2D operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE>
std::set<unsigned> MaxPool2D<DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {
      tfcc::runtime::operations::nn::MaxPool2D::VERSION_1,
      tfcc::runtime::operations::nn::MaxPool2D::VERSION_2};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE>
std::unique_ptr<OperationResource> MaxPool2D<DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 1 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::nn::MaxPool2D>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  if (symbolManager.getSymbolInfo(node.outputs(0)).dtype != DTYPE ||
      symbolManager.getSymbolInfo(node.inputs(0)).dtype != DTYPE) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource());
  tfcc::runtime::operations::nn::MaxPool2D operation;
  node.operation().UnpackTo(&operation);

  bool nchw = true;
  if (node.version() != tfcc::runtime::operations::nn::MaxPool1D::VERSION_1) {
    nchw = operation.nchw();
  }

  auto& jit = getJIT(graph);

  const tfcc::Tensor<T>* inputSymbol = symbolManager.getTensor(node.inputs(0), T());
  tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());

  if (nchw) {
    callFunction(
        jit, _wrapper_max_pool2d_nchw<T>, inputSymbol, operation.kernel_height(),
        operation.kernel_width(), operation.padding_height(), operation.padding_width(),
        operation.stride_height(), operation.stride_width(), outputSymbol);
  } else {
    callFunction(
        jit, _wrapper_max_pool2d_nhwc<T>, inputSymbol, operation.kernel_height(),
        operation.kernel_width(), operation.padding_height(), operation.padding_width(),
        operation.stride_height(), operation.stride_width(), outputSymbol);
  }
  return resource;
}

std::vector<std::unique_ptr<Operation>> get_max_pool2d_operations() {
#define DEFINE_FUNC(dtype) \
  operations.emplace_back(std::unique_ptr<Operation>(new MaxPool2D<dtype>()));
  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace nn
}  // namespace runtime
}  // namespace tfcc
