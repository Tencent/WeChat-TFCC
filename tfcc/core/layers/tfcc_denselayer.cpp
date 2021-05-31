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

#include "tfcc_denselayer.h"

#include <utility>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_invaliddataerror.h"
#include "framework/tfcc_scope.h"
#include "framework/tfcc_types.h"
#include "operations/tfcc_blas.h"
#include "operations/tfcc_operator.h"
#include "utils/tfcc_debugutils.h"

namespace tfcc {
namespace layer {

template <class T>
Dense<T>::Dense(unsigned units, bool useBlas) : Dense(units, useBlas, "dense") {}

template <class T>
Dense<T>::Dense(unsigned units, bool useBlas, std::string name)
    : _initialized(false),
      _kernel(nullptr),
      _bias(nullptr),
      _units(units),
      _useBias(useBlas),
      _name(std::move(name)),
      _kernelName("kernel"),
      _biasName("bias") {}

template <class T>
Variable<T> Dense<T>::operator()(const Tensor<T>& inputs) {
  auto scopeG = Scope::scope(_name);

  // get constant
  if (!_initialized) {
    _kernel = &Constant<T>::getConstant(_kernelName);
    // check kernel
    if (_kernel->shape().size() != 2) {
      throw InvalidDataError("invalid kernel");
    }
    if (_units != 0 && _units != _kernel->shape(1)) {
      throw InvalidDataError("kernel and units don't match");
    }

    if (_useBias) {
      _bias = &Constant<T>::getConstant(_biasName);
      // check bias
      if (_bias->shape().size() != 1) {
        throw InvalidDataError("invalid bias");
      }
      if (_units != 0 && _units != _bias->shape(0)) {
        throw InvalidDataError("bias and units don't match");
      }
      if (_bias->shape(0) != _kernel->shape(1)) {
        throw InvalidDataError("bias and kernel don't match");
      }
    }
    _initialized = true;
  }

  // check input
  if (inputs.shape().size() < 2) {
    throw InvalidArgumentError("invalid inputs");
  }
  if (inputs.shape(inputs.shape().size() - 1) != _kernel->shape(0)) {
    throw InvalidArgumentError("inputs and kernel don't match");
  }

  Variable<T> result = blas::matmul(inputs, *_kernel);
  if (_useBias) {
    result = result + *_bias;
  }
  return result;
}

template <class T>
void Dense<T>::setKernelName(std::string name) {
  _kernelName = std::move(name);
}

template <class T>
void Dense<T>::setBiasName(std::string name) {
  _biasName = std::move(name);
}

template <class T>
Variable<T> dense(const Tensor<T>& inputs, unsigned units, bool useBias) {
  Dense<T> denseLayer(units, useBias);
  return denseLayer(inputs);
}

template <class T>
Variable<T> dense(const Tensor<T>& inputs, unsigned units, bool useBias, std::string name) {
  Dense<T> denseLayer(units, useBias, std::move(name));
  return denseLayer(inputs);
}

template <class T>
Variable<T> dense(
    const Tensor<T>& inputs, unsigned units, bool useBias, std::string name, std::string kernelName,
    std::string biasName) {
  Dense<T> denseLayer(units, useBias, std::move(name));
  denseLayer.setKernelName(std::move(kernelName));
  denseLayer.setBiasName(std::move(biasName));
  return denseLayer(inputs);
}

#define DEFINE_FUNC(type)                                                                  \
  template class Dense<type>;                                                              \
  template Variable<type> dense(const Tensor<type>& inputs, unsigned units, bool useBias); \
  template Variable<type> dense(                                                           \
      const Tensor<type>& inputs, unsigned units, bool useBias, std::string name);         \
  template Variable<type> dense(                                                           \
      const Tensor<type>& inputs, unsigned units, bool useBias, std::string name,          \
      std::string kernelName, std::string biasName);

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace layer
}  // namespace tfcc
