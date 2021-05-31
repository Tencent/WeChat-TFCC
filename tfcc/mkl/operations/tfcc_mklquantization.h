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

#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"

namespace tfcc {
namespace mkl {
namespace quantization {

/**
 * Multiplies int8 matrix a by matrix b, producing a .* b.
 * Two tensors must have same rank or one of them's rank equal to 2.
 * @param a A tensor.
 * @param reduceSumA A tensor which reduced by a along row.
 * @param minA A tensor with shape [1,].
 * @param maxA A tensor with shape [1,].
 * @param b A tensor.
 * @param reduceSumB A tensor which reduced by b along column.
 * @param minB A tensor with shape [1,].
 * @param maxB A tensor with shape [1,].
 * @return A tuple of quantized variable, min value variable and max value variable.
 */
std::tuple<Variable<int32_t>, Variable<float>, Variable<float>> matmul(
    const Tensor<uint8_t>& a, const Tensor<int32_t>& reduceSumA, const Tensor<float>& minA,
    const Tensor<float>& maxA, const Tensor<int8_t>& b, const Tensor<int32_t>& reduceSumB,
    const Tensor<float>& minB, const Tensor<float>& maxB);

/**
 * Reduces a along row or column.
 * @param a A uint8_t tensor or int8_t tensor with 2 or more dimensions.
 * @param row Reduces a along row if true or along column.
 * @return A variable.
 */
template <class T>
Variable<int32_t> reduce_sum(const Tensor<T>& a, bool row);

}  // namespace quantization
}  // namespace mkl
}  // namespace tfcc
