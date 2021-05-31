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
class MinMaxInterface {
 public:
  MinMaxInterface() {}
  MinMaxInterface(const MinMaxInterface&) = delete;
  virtual ~MinMaxInterface() {}

  MinMaxInterface& operator=(const MinMaxInterface&) = delete;

  /**
   * Calculate min(a, b) for earch elements.
   * NOTE: This function support broadcasting.
   * @param a A tensor.
   * @param b A tensor.
   * @return A variable.
   */
  virtual Variable<T> min(const Tensor<T>& a, const Tensor<T>& b);

  /**
   * Calculate min(a, b) for earch elements.
   * NOTE: This function support broadcasting.
   * @param a A tensor.
   * @param b A tensor.
   * @return A variable.
   */
  virtual Variable<T> min(const Tensor<T>& a, T b);

  /**
   * Calculate max(a, b) for earch elements.
   * NOTE: This function support broadcasting.
   * @param a A tensor.
   * @param b A tensor.
   * @return A variable.
   */
  virtual Variable<T> max(const Tensor<T>& a, const Tensor<T>& b);

  /**
   * Calculate max(a, b) for earch elements.
   * NOTE: This function support broadcasting.
   * @param a A tensor.
   * @param b A tensor.
   * @return A variable.
   */
  virtual Variable<T> max(const Tensor<T>& a, T b);
};

}  // namespace tfcc
