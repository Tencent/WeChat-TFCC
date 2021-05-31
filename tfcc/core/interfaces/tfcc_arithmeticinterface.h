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

#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"

namespace tfcc {

template <class T>
class ArithmeticInterface {
 public:
  ArithmeticInterface() {}
  ArithmeticInterface(const ArithmeticInterface&) = delete;
  virtual ~ArithmeticInterface() {}

  ArithmeticInterface& operator=(const ArithmeticInterface&) = delete;

  /**
   * Attention: a and b must has same dimension.
   *      In each dimensions, they must have same length or one of them equal to 1.
   * Example: a.shape: [3, 3], b.shape: [3, 1]    => Valid
   *          a.shape: [3, 3], b.shape: [3]       => Invalid
   */
  virtual Variable<T> add(const Tensor<T>& a, const Tensor<T>& b);

  /**
   * @see add
   */
  virtual Variable<T> sub(const Tensor<T>& a, const Tensor<T>& b);

  /**
   * @see add
   */
  virtual Variable<T> mul(const Tensor<T>& a, const Tensor<T>& b);

  /**
   * @see add
   */
  virtual Variable<T> div(const Tensor<T>& a, const Tensor<T>& b);
};

}  // namespace tfcc
