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

#include "conv2d.h"

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/nn.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace nn {

template <class T>
static const char* _wrapper_conv2d_nchw(
    const tfcc::Tensor<T>* a, const tfcc::Tensor<T>* kernel, unsigned paddingHeight,
    unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth, unsigned dilateHeight,
    unsigned dilateWidth, tfcc::Variable<T>* result) noexcept {
  try {
    *result = tfcc::nn::conv2d(
        *a, false, *kernel, paddingHeight, paddingWidth, strideHeight, strideWidth, dilateHeight,
        dilateWidth);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <class T>
static const char* _wrapper_conv2d_nhwc(
    const tfcc::Tensor<T>* a, const tfcc::Tensor<T>* kernel, unsigned paddingHeight,
    unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth, unsigned dilateHeight,
    unsigned dilateWidth, tfcc::Variable<T>* result) noexcept {
  try {
    *result = tfcc::nn::conv2d(
        *a, true, *kernel, paddingHeight, paddingWidth, strideHeight, strideWidth, dilateHeight,
        dilateWidth);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <class T>
static const char* _wrapper_conv2d_nchw_with_group(
    const tfcc::Tensor<T>* a, const tfcc::Tensor<T>* kernel, unsigned paddingHeight,
    unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth, unsigned dilateHeight,
    unsigned dilateWidth, unsigned group, tfcc::Variable<T>* result) noexcept {
  try {
    *result = tfcc::nn::conv2d(
        *a, false, *kernel, paddingHeight, paddingWidth, strideHeight, strideWidth, dilateHeight,
        dilateWidth, group);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <class T>
static const char* _wrapper_conv2d_nhwc_with_group(
    const tfcc::Tensor<T>* a, const tfcc::Tensor<T>* kernel, unsigned paddingHeight,
    unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth, unsigned dilateHeight,
    unsigned dilateWidth, unsigned group, tfcc::Variable<T>* result) noexcept {
  try {
    *result = tfcc::nn::conv2d(
        *a, true, *kernel, paddingHeight, paddingWidth, strideHeight, strideWidth, dilateHeight,
        dilateWidth, group);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE>
std::string Conv2D<DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::nn::Conv2D operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE>
std::set<unsigned> Conv2D<DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {
      tfcc::runtime::operations::nn::Conv2D::VERSION_1,
      tfcc::runtime::operations::nn::Conv2D::VERSION_2,
      tfcc::runtime::operations::nn::Conv2D::VERSION_3,
      tfcc::runtime::operations::nn::Conv2D::VERSION_4,
  };
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE>
std::unique_ptr<OperationResource> Conv2D<DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 2 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::nn::Conv2D>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  if (symbolManager.getSymbolInfo(node.outputs(0)).dtype != DTYPE) {
    return nullptr;
  }
  if (symbolManager.getSymbolInfo(node.inputs(0)).dtype != DTYPE ||
      symbolManager.getSymbolInfo(node.inputs(1)).dtype != DTYPE) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource());
  tfcc::runtime::operations::nn::Conv2D operation;
  node.operation().UnpackTo(&operation);

  auto& jit = getJIT(graph);

  const tfcc::Tensor<T>* inputSymbol = symbolManager.getTensor(node.inputs(0), T());
  const tfcc::Tensor<T>* kernelSymbol = symbolManager.getTensor(node.inputs(1), T());
  tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());

  unsigned dilate_height = operation.dilate_height();
  unsigned dilate_width = operation.dilate_width();
  dilate_height = dilate_height == 0 ? 1 : dilate_height;
  dilate_width = dilate_width == 0 ? 1 : dilate_width;
  bool nchw = true;
  if (node.version() != tfcc::runtime::operations::nn::Conv2D::VERSION_1 &&
      node.version() != tfcc::runtime::operations::nn::Conv2D::VERSION_2) {
    nchw = operation.nchw();
  }
  uint32_t group = 1;
  if (operation.group() != 0) {
    group = operation.group();
  }
  if (group == 1) {
    if (nchw) {
      callFunction(
          jit, _wrapper_conv2d_nchw<T>, inputSymbol, kernelSymbol, operation.padding_height(),
          operation.padding_width(), operation.stride_height(), operation.stride_width(),
          dilate_height, dilate_width, outputSymbol);
    } else {
      callFunction(
          jit, _wrapper_conv2d_nhwc<T>, inputSymbol, kernelSymbol, operation.padding_height(),
          operation.padding_width(), operation.stride_height(), operation.stride_width(),
          dilate_height, dilate_width, outputSymbol);
    }
  } else {
    if (nchw) {
      callFunction(
          jit, _wrapper_conv2d_nchw_with_group<T>, inputSymbol, kernelSymbol,
          operation.padding_height(), operation.padding_width(), operation.stride_height(),
          operation.stride_width(), dilate_height, dilate_width, group, outputSymbol);
    } else {
      callFunction(
          jit, _wrapper_conv2d_nhwc_with_group<T>, inputSymbol, kernelSymbol,
          operation.padding_height(), operation.padding_width(), operation.stride_height(),
          operation.stride_width(), dilate_height, dilate_width, group, outputSymbol);
    }
  }
  return resource;
}

std::vector<std::unique_ptr<Operation>> get_conv2d_operations() {
#define DEFINE_FUNC(dtype) operations.emplace_back(std::unique_ptr<Operation>(new Conv2D<dtype>()));
  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace nn
}  // namespace runtime
}  // namespace tfcc
