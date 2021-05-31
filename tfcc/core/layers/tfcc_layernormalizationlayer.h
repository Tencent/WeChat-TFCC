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

#pragma once

#include <string>

#include "framework/tfcc_constant.h"
#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"

namespace tfcc {
namespace layer {

template <class T>
class LayerNormalization {
  bool _initialized;
  Constant<T>* _beta;
  Constant<T>* _gamma;
  size_t _beginNormAxis;
  bool _center;
  bool _scale;
  std::string _name;

 public:
  LayerNormalization(size_t beginNormAxis, bool center, bool scale);
  LayerNormalization(size_t beginNormAxis, bool center, bool scale, std::string name);

  Variable<T> operator()(const Tensor<T>& inputs);
};

template <class T>
Variable<T> layer_normalization(
    const Tensor<T>& inputs, size_t beginNormAxis, bool center, bool scale);

template <class T>
Variable<T> layer_normalization(
    const Tensor<T>& inputs, size_t beginNormAxis, bool center, bool scale, std::string name);

}  // namespace layer

}  // namespace tfcc
