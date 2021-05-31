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

#include "tfcc_conv2dlayer.h"

#include <utility>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_invaliddataerror.h"
#include "framework/tfcc_scope.h"
#include "framework/tfcc_types.h"
#include "operations/tfcc_nn.h"
#include "operations/tfcc_operator.h"

namespace tfcc {
namespace layer {

template <class T>
Conv2D<T>::Conv2D(
    unsigned filters, unsigned kernelHeight, unsigned kernelWidth, unsigned strideHeight,
    unsigned strideWidth, bool paddingSame, bool channelsLast, bool useBias)
    : Conv2D(
          filters, kernelHeight, kernelWidth, strideHeight, strideWidth, paddingSame, channelsLast,
          useBias, "conv2d") {}

template <class T>
Conv2D<T>::Conv2D(
    unsigned filters, unsigned kernelHeight, unsigned kernelWidth, unsigned strideHeight,
    unsigned strideWidth, bool paddingSame, bool channelsLast, bool useBias, std::string name)
    : Conv2D(
          filters, kernelHeight, kernelWidth, strideHeight, strideWidth, paddingSame, channelsLast,
          useBias, Conv2DKernelFormat::HWIO, std::move(name)) {}

template <class T>
Conv2D<T>::Conv2D(
    unsigned filters, unsigned kernelHeight, unsigned kernelWidth, unsigned strideHeight,
    unsigned strideWidth, bool paddingSame, bool channelsLast, bool useBias,
    Conv2DKernelFormat kernelFormat, std::string name)
    : _initialized(false),
      _kernel(nullptr),
      _bias(nullptr),
      _filters(filters),
      _kernelHeight(kernelHeight),
      _kernelWidth(kernelWidth),
      _strideHeight(strideHeight),
      _strideWidth(strideWidth),
      _paddingSame(paddingSame),
      _channelsLast(channelsLast),
      _useBias(useBias),
      _kernelFormat(kernelFormat),
      _name(std::move(name)) {}

template <class T>
Variable<T> Conv2D<T>::operator()(const Tensor<T>& inputs) {
  auto scopeG = Scope::scope(_name);

  // initialize
  if (!_initialized) {
    _kernel = _kernelFormat == Conv2DKernelFormat::HWIO
                  ? &Constant<T>::getConstantWithTranspose("kernel", {3, 2, 0, 1})
                  : &Constant<T>::getConstant("kernel");

    // check kernel
    if (_kernel->shape().size() != 4) {
      throw InvalidDataError("invalid kernel");
    }
    if (_filters != 0 && _filters != _kernel->shape(0)) {
      throw InvalidDataError("kernel and filters don't match");
    }
    if (_kernelHeight != 0 && _kernelHeight != _kernel->shape(2)) {
      throw InvalidDataError("kernel and kernel height don't match");
    }
    if (_kernelWidth != 0 && _kernelWidth != _kernel->shape(3)) {
      throw InvalidDataError("kernel and kernel width don't match");
    }

    if (_useBias) {
      _bias = &Constant<T>::getConstant("bias", [this](Shape shape, std::vector<T> data) {
        return this->biasInitialize(std::move(shape), std::move(data));
      });

      // check bias
      if (_bias->shape().size() != 4) {
        throw InvalidDataError("invalid bias");
      }
      if (_bias->size() != _kernel->shape(0)) {
        throw InvalidDataError("kernel and bias don't match");
      }
      if (_channelsLast && _bias->shape(3) != _kernel->shape(0)) {
        throw InvalidDataError("kernel and bias don't match");
      }
      if (!_channelsLast && _bias->shape(1) != _kernel->shape(0)) {
        throw InvalidDataError("kernel and bias don't match");
      }
    }

    _initialized = true;
  }

  Variable<T> result;
  if (_paddingSame) {
    result = nn::conv2d_same(inputs, _channelsLast, *_kernel, _strideHeight, _strideWidth);
  } else {
    result = nn::conv2d_valid(inputs, _channelsLast, *_kernel, _strideHeight, _strideWidth);
  }
  if (_useBias) {
    result = result + *_bias;
  }
  return result;
}

template <class T>
std::pair<Shape, std::vector<T>> Conv2D<T>::biasInitialize(Shape shape, std::vector<T> data) {
  if (shape.size() != 1) {
    return std::make_pair(std::move(shape), std::move(data));
  }
  if (_channelsLast) {
    return std::make_pair(Shape({1, 1, 1, shape[0]}), std::move(data));
  } else {
    return std::make_pair(Shape({1, shape[0], 1, 1}), std::move(data));
  }
}

template <class T>
Variable<T> conv2d(
    const Tensor<T>& inputs, unsigned filters, unsigned kernelHeight, unsigned kernelWidth,
    unsigned strideHeight, unsigned strideWidth, bool paddingSame, bool channelsLast,
    bool useBias) {
  Conv2D<T> convLayer(
      filters, kernelHeight, kernelWidth, strideHeight, strideWidth, paddingSame, channelsLast,
      useBias);
  return convLayer(inputs);
}

template <class T>
Variable<T> conv2d(
    const Tensor<T>& inputs, unsigned filters, unsigned kernelHeight, unsigned kernelWidth,
    unsigned strideHeight, unsigned strideWidth, bool paddingSame, bool channelsLast, bool useBias,
    std::string name) {
  Conv2D<T> convLayer(
      filters, kernelHeight, kernelWidth, strideHeight, strideWidth, paddingSame, channelsLast,
      useBias, name);
  return convLayer(inputs);
}

template <class T>
Variable<T> conv2d(
    const Tensor<T>& inputs, unsigned filters, unsigned kernelHeight, unsigned kernelWidth,
    unsigned strideHeight, unsigned strideWidth, bool paddingSame, bool channelsLast, bool useBias,
    Conv2DKernelFormat kernelFormat, std::string name) {
  Conv2D<T> convLayer(
      filters, kernelHeight, kernelWidth, strideHeight, strideWidth, paddingSame, channelsLast,
      useBias, kernelFormat, name);
  return convLayer(inputs);
}

#define DEFINE_FUNC(type)                                                                        \
  template class Conv2D<type>;                                                                   \
  template Variable<type> conv2d(                                                                \
      const Tensor<type>& inputs, unsigned filters, unsigned kernelHeight, unsigned kernelWidth, \
      unsigned strideHeight, unsigned strideWidth, bool paddingSame, bool channelsLast,          \
      bool useBias);                                                                             \
  template Variable<type> conv2d(                                                                \
      const Tensor<type>& inputs, unsigned filters, unsigned kernelHeight, unsigned kernelWidth, \
      unsigned strideHeight, unsigned strideWidth, bool paddingSame, bool channelsLast,          \
      bool useBias, std::string name);                                                           \
  template Variable<type> conv2d(                                                                \
      const Tensor<type>& inputs, unsigned filters, unsigned kernelHeight, unsigned kernelWidth, \
      unsigned strideHeight, unsigned strideWidth, bool paddingSame, bool channelsLast,          \
      bool useBias, Conv2DKernelFormat kernelFormat, std::string name);

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace layer
}  // namespace tfcc
