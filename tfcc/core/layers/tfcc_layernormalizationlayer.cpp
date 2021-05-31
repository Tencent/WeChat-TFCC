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

#include "tfcc_layernormalizationlayer.h"

#include <utility>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_invaliddataerror.h"
#include "framework/tfcc_scope.h"
#include "framework/tfcc_types.h"
#include "interfaces/tfcc_normalizationinterface.h"
#include "operations/tfcc_math.h"
#include "operations/tfcc_operation.h"
#include "operations/tfcc_operator.h"

namespace tfcc {
namespace layer {

template <class T>
LayerNormalization<T>::LayerNormalization(size_t beginNormAxis, bool center, bool scale)
    : LayerNormalization(beginNormAxis, center, scale, "LayerNorm") {}

template <class T>
LayerNormalization<T>::LayerNormalization(
    size_t beginNormAxis, bool center, bool scale, std::string name)
    : _initialized(false),
      _beta(nullptr),
      _gamma(nullptr),
      _beginNormAxis(beginNormAxis),
      _center(center),
      _scale(scale),
      _name(std::move(name)) {}

template <class T>
Variable<T> LayerNormalization<T>::operator()(const Tensor<T>& inputs) {
  auto scopeG = Scope::scope(_name);

  if (_beginNormAxis >= inputs.shape().size()) {
    throw InvalidArgumentError("invalid input");
  }

  if (!_initialized) {
    if (_center) {
      _beta = &Constant<T>::getConstant("beta");
    } else {
      _beta = Constant<T>::tryGetRuntimeConstant("beta");
    }
    if (_scale) {
      _gamma = &Constant<T>::getConstant("gamma");
    } else {
      _gamma = Constant<T>::tryGetRuntimeConstant("gamma");
    }
    if (_beta == nullptr || _gamma == nullptr) {
      std::vector<unsigned> s;
      for (size_t i = _beginNormAxis; i < inputs.shape().size(); ++i) {
        s.push_back(inputs.shape(i));
      }
      Shape shape(s);
      if (_beta == nullptr) {
        std::vector<T> data(shape.area(), static_cast<T>(0));
        Constant<T>::setRuntimeConstantIfNotExist("beta", shape, data);
        _beta = Constant<T>::tryGetRuntimeConstant("beta");
      }
      if (_gamma == nullptr) {
        std::vector<T> data(shape.area(), static_cast<T>(1));
        Constant<T>::setRuntimeConstantIfNotExist("gamma", shape, data);
        _gamma = Constant<T>::tryGetRuntimeConstant("gamma");
      }
    }
    _initialized = true;
  }

  for (size_t i = 0; i < inputs.shape().size() - _beginNormAxis; ++i) {
    if (_beta != nullptr && inputs.shape(i + _beginNormAxis) != _beta->shape(i)) {
      throw InvalidDataError("inputs and beta don't match");
    }
    if (_gamma != nullptr && inputs.shape(i + _beginNormAxis) != _gamma->shape(i)) {
      throw InvalidDataError("inputs and gamma don't match");
    }
  }

  return Operation<T>::getCurrentInterface().getNormalizationInterface().layerNormalize(
      inputs, *_gamma, *_beta, static_cast<T>(1e-12), _beginNormAxis);
}

template <class T>
Variable<T> layer_normalization(
    const Tensor<T>& inputs, size_t beginNormAxis, bool center, bool scale) {
  LayerNormalization<T> layerNormalizationLayer(beginNormAxis, center, scale);
  return layerNormalizationLayer(inputs);
}

template <class T>
Variable<T> layer_normalization(
    const Tensor<T>& inputs, size_t beginNormAxis, bool center, bool scale, std::string name) {
  LayerNormalization<T> layerNormalizationLayer(beginNormAxis, center, scale, std::move(name));
  return layerNormalizationLayer(inputs);
}

#define DEFINE_FUNC(type)                                                         \
  template class LayerNormalization<type>;                                        \
  template Variable<type> layer_normalization(                                    \
      const Tensor<type>& inputs, size_t beginNormAxis, bool center, bool scale); \
  template Variable<type> layer_normalization(                                    \
      const Tensor<type>& inputs, size_t beginNormAxis, bool center, bool scale,  \
      std::string name);

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace layer

}  // namespace tfcc
