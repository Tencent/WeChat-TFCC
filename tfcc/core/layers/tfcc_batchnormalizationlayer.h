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

#include "framework/tfcc_constant.h"
#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"

namespace tfcc {
namespace layer {

template <class T>
class BatchNormalization {
  bool _initialized;
  Constant<T>* _mean;
  Constant<T>* _variance;
  Constant<T>* _offset;
  Constant<T>* _scale;
  T _epsilon;
  std::vector<unsigned> _axisList;
  std::string _name;

 public:
  /**
   * @see BatchNormalization(T epsilon, std::vector<unsigned> axisList, std::string name)
   */
  BatchNormalization();

  /**
   * @see BatchNormalization(T epsilon, std::vector<unsigned> axisList, std::string name)
   */
  explicit BatchNormalization(T epsilon);

  /**
   * @see BatchNormalization(T epsilon, std::vector<unsigned> axisList, std::string name)
   */
  explicit BatchNormalization(std::string name);

  /**
   * @see BatchNormalization(T epsilon, std::vector<unsigned> axisList, std::string name)
   */
  BatchNormalization(T epsilon, std::string name);

  /**
   * @param epsilon Small float added to variance to avoid dividing by zero. The default value is
   * 0.001.
   * @param axisList The list of axis should be normalized. Specially, the last axis should be
   * normalized while it's empty.
   * @param name Name of the layer. The default value is 'batch_normalization'.
   */
  BatchNormalization(T epsilon, std::vector<unsigned> axisList, std::string name);

  Variable<T> operator()(const Tensor<T>& inputs);
};

template <class T>
Variable<T> batch_normalization(const Tensor<T>& inputs);

template <class T>
Variable<T> batch_normalization(const Tensor<T>& inputs, T epsilon);

template <class T>
Variable<T> batch_normalization(const Tensor<T>& inputs, std::string name);

template <class T>
Variable<T> batch_normalization(const Tensor<T>& inputs, T epsilon, std::string name);

template <class T>
Variable<T> batch_normalization(
    const Tensor<T>& inputs, T epsilon, std::vector<unsigned> axisList, std::string name);

}  // namespace layer
}  // namespace tfcc
