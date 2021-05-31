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

#include <tuple>
#include <vector>

#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"

namespace tfcc {

template <class T>
class ScatterInterface {
 public:
  ScatterInterface() {}
  ScatterInterface(const ScatterInterface&) = delete;
  virtual ~ScatterInterface() {}

  ScatterInterface& operator=(const ScatterInterface&) = delete;

  /**
   * Scatter updates into a new tensor according to indices.
   * @param indices A Tensor. Index tensor.
   * @param updates A Tensor. Updates to scatter into output.
   * @param shape A Shape.
   * @return A Tensor.
   */
  virtual Variable<T> scatterND(
      const Tensor<uint32_t>& indices, const Tensor<T>& updates, const Shape& shape);
  virtual Variable<T> scatterND(
      const Tensor<int32_t>& indices, const Tensor<T>& updates, const Shape& shape);
  virtual Variable<T> scatterND(
      const Tensor<uint64_t>& indices, const Tensor<T>& updates, const Shape& shape);
  virtual Variable<T> scatterND(
      const Tensor<int64_t>& indices, const Tensor<T>& updates, const Shape& shape);

  /**
   * Scatter updates into a new tensor according to indices.
   * @param data A Tensor. Base tensor.
   * @param indices A Tensor. Index tensor.
   * @param updates A Tensor. Updates to scatter into output.
   * @return A Tensor.
   */
  virtual Variable<T> scatterND(
      const Tensor<T>& data, const Tensor<uint32_t>& indices, const Tensor<T>& updates);
  virtual Variable<T> scatterND(
      const Tensor<T>& data, const Tensor<int32_t>& indices, const Tensor<T>& updates);
  virtual Variable<T> scatterND(
      const Tensor<T>& data, const Tensor<uint64_t>& indices, const Tensor<T>& updates);
  virtual Variable<T> scatterND(
      const Tensor<T>& data, const Tensor<int64_t>& indices, const Tensor<T>& updates);
};

}  // namespace tfcc
