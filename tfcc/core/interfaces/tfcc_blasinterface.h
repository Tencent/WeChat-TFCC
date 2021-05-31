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
class BlasInterface {
 public:
  BlasInterface() {}
  BlasInterface(const BlasInterface&) = delete;
  virtual ~BlasInterface() {}

  BlasInterface& operator=(const BlasInterface&) = delete;

  /**
   * Attention: this function support batch matmul.
   * Example: a.shape: [2, 3], b.shape[3, 4]                              => Valid
   *          a.shape: [N1, ..., NK, 2, 3], b.shape[N1, ..., NK, 3, 4]    => Valid
   *          a.shape: [N1, ..., NK, 2, 3], b.shape[3, 4]                 => Valid
   *          a.shape: [2, 3], b.shape[N1, ..., NK, 3, 4]                 => Valid
   * @param a A tensor.
   * @param b A tensor.
   * @return a .* b.
   */
  virtual Variable<T> matmul(const Tensor<T>& a, const Tensor<T>& b);

  /**
   * Attention: this function support batch matmul.
   * Example: a.shape: [2, 3], b.shape[3, 4]                              => Valid
   *          a.shape: [N1, ..., NK, 2, 3], b.shape[N1, ..., NK, 3, 4]    => Valid
   *          a.shape: [N1, ..., NK, 2, 3], b.shape[3, 4]                 => Valid
   *          a.shape: [2, 3], b.shape[N1, ..., NK, 3, 4]                 => Valid
   * @param a A tensor.
   * @param b A tensor.
   * @param c A one dimension tensor.
   * @return a .* b + c.
   */

  virtual Variable<T> matmul(const Tensor<T>& a, const Tensor<T>& b, const Tensor<T>& c);
};

}  // namespace tfcc
