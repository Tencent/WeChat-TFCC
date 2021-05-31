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

#include "tfcc_batchnormalizationlayer.h"

#include <utility>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_invaliddataerror.h"
#include "framework/tfcc_scope.h"
#include "framework/tfcc_types.h"
#include "framework/tfcc_view.h"
#include "operations/tfcc_math.h"
#include "operations/tfcc_nn.h"
#include "operations/tfcc_operator.h"

namespace tfcc {
namespace layer {

template <class T>
BatchNormalization<T>::BatchNormalization()
    : BatchNormalization(static_cast<T>(0.001), "batch_normalization") {}

template <class T>
BatchNormalization<T>::BatchNormalization(T epsilon)
    : BatchNormalization(epsilon, {}, "batch_normalization") {}

template <class T>
BatchNormalization<T>::BatchNormalization(std::string name)
    : BatchNormalization(static_cast<T>(0.001), {}, std::move(name)) {}

template <class T>
BatchNormalization<T>::BatchNormalization(T epsilon, std::string name)
    : BatchNormalization(epsilon, {}, name) {}

template <class T>
BatchNormalization<T>::BatchNormalization(
    T epsilon, std::vector<unsigned> axisList, std::string name)
    : _initialized(false),
      _mean(nullptr),
      _variance(nullptr),
      _offset(nullptr),
      _scale(nullptr),
      _epsilon(epsilon),
      _axisList(std::move(axisList)),
      _name(std::move(name)) {}

template <class T>
Variable<T> BatchNormalization<T>::operator()(const Tensor<T>& inputs) {
  auto scopeG = Scope::scope(_name);

  // initialize
  if (!_initialized) {
    _mean = &Constant<T>::getConstant("moving_mean");
    if (_axisList.size() == 0 && _mean->shape().size() != 1) {
      throw InvalidDataError("invalid mean");
    }
    if (_axisList.size() > 1) {
      unsigned area = 1;
      for (unsigned axis : _axisList) {
        if (axis >= _mean->shape().size()) {
          throw InvalidDataError("invalid mean");
        }
        area *= _mean->shape(axis);
      }
      if (_axisList.size() != 0 && area != _mean->size()) {
        throw InvalidDataError("invalid mean");
      }
    }

    if (_mean->shape().size() == 1 && _axisList.size() > 1) {
      throw InvalidDataError("invalid mean");
    }

    _variance = &Constant<T>::getConstant("moving_variance");
    if (_variance->shape() != _mean->shape()) {
      throw InvalidDataError("mean and variance don't match");
    }

    _offset = &Constant<T>::getConstant("beta");
    if (_offset->shape() != _mean->shape()) {
      throw InvalidDataError("mean and offset don't match");
    }

    _scale = &Constant<T>::getConstant("gamma");
    if (_scale->shape() != _mean->shape()) {
      throw InvalidDataError("mean and scale don't match");
    }

    _initialized = true;
  }

  View<T> mean(*_mean);
  View<T> variance(*_variance);
  View<T> offset(*_offset);
  View<T> scale(*_scale);

  if (_mean->shape().size() == 1 && _axisList.size() == 1) {
    std::vector<unsigned> s(inputs.shape().size(), 1);
    s[_axisList[0]] = _mean->size();

    mean.reshape(s);
    variance.reshape(s);
    offset.reshape(s);
    scale.reshape(s);
  }

  Variable<T> inv = math::rsqrt(variance + _epsilon);
  inv = inv * scale;
  return inputs * inv + offset - mean * inv;
}

template <class T>
Variable<T> batch_normalization(const Tensor<T>& inputs) {
  BatchNormalization<T> batchNormalizationLayer;
  return batchNormalizationLayer(inputs);
}

template <class T>
Variable<T> batch_normalization(const Tensor<T>& inputs, T epsilon) {
  BatchNormalization<T> batchNormalizationLayer(epsilon);
  return batchNormalizationLayer(inputs);
}

template <class T>
Variable<T> batch_normalization(const Tensor<T>& inputs, std::string name) {
  BatchNormalization<T> batchNormalizationLayer(std::move(name));
  return batchNormalizationLayer(inputs);
}

template <class T>
Variable<T> batch_normalization(const Tensor<T>& inputs, T epsilon, std::string name) {
  BatchNormalization<T> batchNormalizationLayer(epsilon, std::move(name));
  return batchNormalizationLayer(inputs);
}

template <class T>
Variable<T> batch_normalization(
    const Tensor<T>& inputs, T epsilon, std::vector<unsigned> axisList, std::string name) {
  BatchNormalization<T> batchNormalizationLayer(epsilon, std::move(axisList), std::move(name));
  return batchNormalizationLayer(inputs);
}

#define DEFINE_FUNC(type)                                                                    \
  template class BatchNormalization<type>;                                                   \
  template Variable<type> batch_normalization(const Tensor<type>& inputs);                   \
  template Variable<type> batch_normalization(const Tensor<type>& inputs, type epsilon);     \
  template Variable<type> batch_normalization(const Tensor<type>& inputs, std::string name); \
  template Variable<type> batch_normalization(                                               \
      const Tensor<type>& inputs, type epsilon, std::string name);                           \
  template Variable<type> batch_normalization(                                               \
      const Tensor<type>& inputs, type epsilon, std::vector<unsigned> axisList, std::string name);

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace layer
}  // namespace tfcc
