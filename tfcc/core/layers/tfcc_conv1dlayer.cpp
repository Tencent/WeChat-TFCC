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

#include "tfcc_conv1dlayer.h"

#include <utility>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_invaliddataerror.h"
#include "framework/tfcc_scope.h"
#include "framework/tfcc_types.h"
#include "operations/tfcc_nn.h"
#include "operations/tfcc_operator.h"
#include "utils/tfcc_commutils.h"

namespace tfcc {
namespace layer {

template <class T>
Conv1D<T>::Conv1D(
    unsigned filters, unsigned kernelSize, unsigned stride, bool paddingSame, bool channelsLast,
    bool useBias)
    : Conv1D(filters, kernelSize, stride, paddingSame, channelsLast, useBias, "conv1d") {}

template <class T>
Conv1D<T>::Conv1D(
    unsigned filters, unsigned kernelSize, unsigned stride, bool paddingSame, bool channelsLast,
    bool useBias, std::string name)
    : Conv1D(
          filters, kernelSize, stride, paddingSame, channelsLast, useBias, Conv1DKernelFormat::WIO,
          std::move(name)) {}

template <class T>
Conv1D<T>::Conv1D(
    unsigned filters, unsigned kernelSize, unsigned stride, bool paddingSame, bool channelsLast,
    bool useBias, Conv1DKernelFormat kernelFormat, std::string name)
    : _initialized(false),
      _kernel(nullptr),
      _bias(nullptr),
      _filters(filters),
      _kernelSize(kernelSize),
      _stride(stride),
      _paddingSame(paddingSame),
      _channelsLast(channelsLast),
      _useBias(useBias),
      _kernelFormat(kernelFormat),
      _name(name) {}

template <class T>
Variable<T> Conv1D<T>::operator()(const Tensor<T>& inputs) {
  auto scopeG = Scope::scope(_name);

  // initialize
  if (!_initialized) {
    _kernel = &Constant<T>::getConstant("kernel", [this](Shape shape, std::vector<T> data) {
      return this->kernelInitialize(std::move(shape), std::move(data));
    });

    // check kernel
    if (_kernel->shape().size() != 3) {
      throw InvalidDataError("invalid kernel");
    }
    if (_filters != 0 && _filters != _kernel->shape(0)) {
      throw InvalidDataError("kernel and filters don't match");
    }
    if (_kernelSize != 0 && _kernelSize != _kernel->shape(2)) {
      throw InvalidDataError("kernel and kernel size don't match");
    }

    if (_useBias) {
      _bias = &Constant<T>::getConstant("bias", [this](Shape shape, std::vector<T> data) {
        return this->biasInitialize(std::move(shape), std::move(data));
      });

      // check bias
      if (_bias->shape().size() != 3) {
        throw InvalidDataError("invalid bias");
      }
      if (_bias->size() != _kernel->shape(0)) {
        throw InvalidDataError("kernel and bias don't match");
      }
      if (_channelsLast && _bias->shape(2) != _kernel->shape(0)) {
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
    result = nn::conv1d_same(inputs, _channelsLast, *_kernel, _stride);
  } else {
    result = nn::conv1d_valid(inputs, _channelsLast, *_kernel, _stride);
  }
  if (_useBias) {
    result = result + *_bias;
  }
  return result;
}

template <class T>
std::pair<Shape, std::vector<T>> Conv1D<T>::kernelInitialize(Shape shape, std::vector<T> data) {
  if (_kernelFormat != Conv1DKernelFormat::WIO) {
    return std::make_pair(std::move(shape), std::move(data));
  }
  if (shape.size() != 3) {
    return std::make_pair(std::move(shape), std::move(data));
  }
  return std::make_pair(
      Shape({shape[2], shape[1], shape[0]}), host_transpose(data, shape, {2, 1, 0}));
}

template <class T>
std::pair<Shape, std::vector<T>> Conv1D<T>::biasInitialize(Shape shape, std::vector<T> data) {
  if (shape.size() != 1) {
    return std::make_pair(std::move(shape), std::move(data));
  }
  if (_channelsLast) {
    return std::make_pair(Shape({1, 1, shape[0]}), std::move(data));
  } else {
    return std::make_pair(Shape({1, shape[0], 1}), std::move(data));
  }
}

template <class T>
Variable<T> conv1d(
    const Tensor<T>& inputs, unsigned filters, unsigned kernelSize, unsigned stride,
    bool paddingSame, bool channelsLast, bool useBias) {
  Conv1D<T> convLayer(filters, kernelSize, stride, paddingSame, channelsLast, useBias);
  return convLayer(inputs);
}

template <class T>
Variable<T> conv1d(
    const Tensor<T>& inputs, unsigned filters, unsigned kernelSize, unsigned stride,
    bool paddingSame, bool channelsLast, bool useBias, std::string name) {
  Conv1D<T> convLayer(
      filters, kernelSize, stride, paddingSame, channelsLast, useBias, std::move(name));
  return convLayer(inputs);
}

template <class T>
Variable<T> conv1d(
    const Tensor<T>& inputs, unsigned filters, unsigned kernelSize, unsigned stride,
    bool paddingSame, bool channelsLast, bool useBias, Conv1DKernelFormat kernelFormat,
    std::string name) {
  Conv1D<T> convLayer(
      filters, kernelSize, stride, paddingSame, channelsLast, useBias, kernelFormat,
      std::move(name));
  return convLayer(inputs);
}

#define DEFINE_FUNC(type)                                                                 \
  template class Conv1D<type>;                                                            \
  template Variable<type> conv1d(                                                         \
      const Tensor<type>& inputs, unsigned filters, unsigned kernelSize, unsigned stride, \
      bool paddingSame, bool channelsLast, bool useBias);                                 \
  template Variable<type> conv1d(                                                         \
      const Tensor<type>& inputs, unsigned filters, unsigned kernelSize, unsigned stride, \
      bool paddingSame, bool channelsLast, bool useBias, std::string name);               \
  template Variable<type> conv1d(                                                         \
      const Tensor<type>& inputs, unsigned filters, unsigned kernelSize, unsigned stride, \
      bool paddingSame, bool channelsLast, bool useBias, Conv1DKernelFormat kernelFormat, \
      std::string name);

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace layer
}  // namespace tfcc
