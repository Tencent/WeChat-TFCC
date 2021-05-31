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
namespace blas {

/**
 * Multiplies matrix a by matrix b, producing a .* b.
 * Two tensors must have same rank or one of them's rank equal to 2.
 * @param a A tensor.
 * @param b A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> matmul(const Tensor<T>& a, const Tensor<T>& b);

/**
 * Multiplies matrix a by matrix b and add blas c, producing a .* b + c.
 * tensors a and b must have same rank or one of them's rank equal to 2.
 * tensor c's rank must equal to 1.
 * @param a A tensor.
 * @param b A tensor.
 * @param c A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> matmul(const Tensor<T>& a, const Tensor<T>& b, const Tensor<T>& c);

}  // namespace blas
}  // namespace tfcc
