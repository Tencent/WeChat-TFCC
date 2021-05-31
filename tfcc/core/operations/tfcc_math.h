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

#include <cstdint>
#include <initializer_list>

#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"

namespace tfcc {
namespace math {

/**
 * Calculate a + b.
 * NOTE: This function support broadcasting.
 * @param a A tensor.
 * @param b A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> add(const Tensor<T>& a, const Tensor<T>& b);

/**
 * Calculate a - b.
 * NOTE: This function support broadcasting.
 * @param a A tensor.
 * @param b A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> sub(const Tensor<T>& a, const Tensor<T>& b);

/**
 * Calculate a * b.
 * NOTE: This function support broadcasting.
 * @param a A tensor.
 * @param b A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> mul(const Tensor<T>& a, const Tensor<T>& b);

/**
 * Calculate a / b.
 * NOTE: This function support broadcasting.
 * @param a A tensor.
 * @param b A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> div(const Tensor<T>& a, const Tensor<T>& b);

/**
 * Calculate values[0] + values[1] + values[2] + ...
 * NOTE: Values must have same shape.
 * @param values A list of tensor pointer.
 * @return A variable.
 */
template <class T>
Variable<T> batch_add(const std::vector<const Tensor<T>*>& values);

/**
 * Calculate values[0] + values[1] + values[2] + ...
 * NOTE: Values must have same shape.
 * @param values A list of tensor pointer.
 * @return A variable.
 */
template <class T>
Variable<T> batch_add(std::initializer_list<const Tensor<T>*> values);

/**
 * Calculate values[0] * values[1] * values[2] * ...
 * NOTE: Values must have same shape.
 * @param values A list of tensor pointer.
 * @return A variable.
 */
template <class T>
Variable<T> batch_mul(const std::vector<const Tensor<T>*>& values);

/**
 * Calculate values[0] * values[1] * values[2] * ...
 * NOTE: Values must have same shape.
 * @param values A list of tensor pointer.
 * @return A variable.
 */
template <class T>
Variable<T> batch_mul(std::initializer_list<const Tensor<T>*> values);

/**
 * Calculate a * alpha + beta.
 * @param a A tensor.
 * @param alpha Specifies the scalar alpha.
 * @param beta Specifies the scalar beta.
 * @return A variable.
 */
template <class T>
Variable<T> transform(const Tensor<T>& a, T alpha, T beta);

/**
 * Calculate a / alpha + beta.
 * @param a A tensor.
 * @param alpha Specifies the scalar alpha.
 * @param beta Specifies the scalar beta.
 * @return A variable.
 */
template <class T>
Variable<T> transform2(const Tensor<T>& a, T alpha, T beta);

/**
 * Calculate alpha / a + beta.
 * @param a A tensor.
 * @param alpha Specifies the scalar alpha.
 * @param beta Specifies the scalar beta.
 * @return A variable.
 */
template <class T>
Variable<T> transform3(const Tensor<T>& a, T alpha, T beta);

/**
 * Calculate beta - a * alpha.
 * @param a A tensor.
 * @param alpha Specifies the scalar alpha.
 * @param beta Specifies the scalar beta.
 * @return A variable.
 */
template <class T>
Variable<T> transform4(const Tensor<T>& a, T alpha, T beta);

/**
 * Calculate beta - a / alpha.
 * @param a A tensor.
 * @param alpha Specifies the scalar alpha.
 * @param beta Specifies the scalar beta.
 * @return A variable.
 */
template <class T>
Variable<T> transform5(const Tensor<T>& a, T alpha, T beta);

/**
 * Calculate beta - alpha / a.
 * @param a A tensor.
 * @param alpha Specifies the scalar alpha.
 * @param beta Specifies the scalar beta.
 * @return A variable.
 */
template <class T>
Variable<T> transform6(const Tensor<T>& a, T alpha, T beta);

/**
 * Calculate min(a, b).
 * NOTE: This function support broadcasting.
 * @param a A tensor.
 * @param b A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> min(const Tensor<T>& a, const Tensor<T>& b);

/**
 * Calculate min(a, b).
 * NOTE: This function support broadcasting.
 * @param a A tensor.
 * @param b A value.
 * @return A variable.
 */
template <class T>
Variable<T> min(const Tensor<T>& a, T b);

/**
 * Calculate min(a, b).
 * NOTE: This function support broadcasting.
 * @param a A value.
 * @param b A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> min(T a, const Tensor<T>& b);

/**
 * Calculate max(a, b).
 * NOTE: This function support broadcasting.
 * @param a A tensor.
 * @param b A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> max(const Tensor<T>& a, const Tensor<T>& b);

/**
 * Calculate max(a, b).
 * NOTE: This function support broadcasting.
 * @param a A tensor.
 * @param b A value.
 * @return A variable.
 */
template <class T>
Variable<T> max(const Tensor<T>& a, T b);

/**
 * Calculate max(a, b).
 * NOTE: This function support broadcasting.
 * @param a A value.
 * @param b A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> max(T a, const Tensor<T>& b);

/**
 * Calculate 1 / (1 + exp(-a)).
 * @param a A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> sigmoid(const Tensor<T>& a);

/**
 * Calculate max(0, a).
 * @param a A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> relu(const Tensor<T>& a);

/**
 * Calculate max(alpha * a, a).
 * @param a A tensor.
 * @param alpha Specifies the scalar alpha.
 * @return A variable.
 */
template <class T>
Variable<T> leaky_relu(const Tensor<T>& a, T alpha);

/**
 * Calculate log(exp(a) + 1).
 * @param a A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> softplus(const Tensor<T>& a);

/**
 * Calculate log(a).
 * @param a A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> log(const Tensor<T>& a);

/**
 * Calculate rsqrt(a).
 * @param a A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> rsqrt(const Tensor<T>& a);

/**
 * Calculate erf(a).
 * @param a A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> erf(const Tensor<T>& a);

/**
 * Calculate tanh(a).
 * @param a A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> tanh(const Tensor<T>& a);

/**
 * Calculate sin(a).
 * @param a A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> sin(const Tensor<T>& a);

/**
 * Calculate cos(a).
 * @param a A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> cos(const Tensor<T>& a);

/**
 * Calculate pow(a, exponent).
 * @param a A tensor.
 * @param exponent Exponent value.
 * @return A variable.
 */
template <class T>
Variable<T> pow(const Tensor<T>& a, T exponent);

/**
 * Calculate pow(a, exponent).
 * @param a A tensor.
 * @param exponent Exponent value.
 * @return A variable.
 */
template <class T>
Variable<T> pow(const Tensor<T>& a, const Tensor<T>& exponent);

/**
 * Calculate pow(a, exponent).
 * @param a A value.
 * @param exponent Exponent value.
 * @return A variable.
 */
template <class T>
Variable<T> pow(T a, const Tensor<T>& exponent);

/**
 * Calculate softmax activations.
 * @param a A tensor.
 * @param axis The dimension softmax would be performed on.
 * @return A variable.
 */
template <class T>
Variable<T> softmax(const Tensor<T>& a, size_t axis);

/**
 * Reduces a along the dimensions greater than keep.
 * Example: a.shape() => [5, 4, 3, 2], reduce_sum(a, 2).shape() => [5, 4, 1, 1]
 * @param a A tensor.
 * @param keep The number of dimensions to keep.
 * @return A variable.
 */
template <class T>
Variable<T> reduce_sum(const Tensor<T>& a, size_t keep);

/**
 * Reduces a along the dimensions greater than keep.
 * Example: a.shape() => [5, 4, 3, 2], reduce_mean(a, 2).shape() => [5, 4, 1, 1]
 * @param a A tensor.
 * @param keep The number of dimensions to keep.
 * @return A variable.
 */
template <class T>
Variable<T> reduce_mean(const Tensor<T>& a, size_t keep);

/**
 * Reduces a along the dimensions greater than keep.
 * Example: a.shape() => [5, 4, 3, 2], reduce_prod(a, 2).shape() => [5, 4, 1, 1]
 * @param a A tensor.
 * @param keep The number of dimensions to keep.
 * @return A variable.
 */
template <class T>
Variable<T> reduce_prod(const Tensor<T>& a, size_t keep);

/**
 * Reduces a along the dimensions greater than keep.
 * Example: a.shape() => [5, 4, 3, 2], reduce_prod(a, 2).shape() => [5, 4, 1, 1]
 * @param a A tensor.
 * @param keep The number of dimensions to keep.
 * @return A variable.
 */
template <class T>
Variable<T> reduce_max(const Tensor<T>& a, size_t keep);

/**
 * Reduces a along the dimensions greater than keep.
 * Example: a.shape() => [5, 4, 3, 2], reduce_prod(a, 2).shape() => [5, 4, 1, 1]
 * @param a A tensor.
 * @param keep The number of dimensions to keep.
 * @return A variable.
 */
template <class T>
Variable<T> reduce_min(const Tensor<T>& a, size_t keep);

/**
 * Reduces a along the dimensions greater than keep.
 * Example: a.shape() => [5, 4, 3, 2], reduce_prod(a, 2).shape() => [5, 4, 1, 1]
 * @param a A tensor.
 * @param keep The number of dimensions to keep.
 * @return A variable.
 */
template <class T>
Variable<T> reduce_any(const Tensor<T>& a, size_t keep);

/**
 * Clips tensor values to a specified min and max.
 * @param a A tensor.
 * @param minValue The minimum value to clip by.
 * @param maxValue The maximum value to clip by.
 * @return A variable.
 */
template <class T>
Variable<T> clip(const Tensor<T>& a, T minValue, T maxValue);

/**
 * Finds values and indices of the k largest entries for the last dimension.
 * @param a A tensor.
 * @param k Number of top elements to look for along the last dimension.
 * @return A tuple of top k entries and indices.
 * @deprecated
 */
template <class T>
std::tuple<Variable<T>, Variable<uint32_t>> top_k(const Tensor<T>& a, unsigned k);

/**
 * Computes the sum along segments of a tensor.
 * Like tf.math.unsorted_segment_sum.
 * @param a A tensor.
 * @param ids A tensor.
 * @param num The number of distinct segment ids.
 * @return A variable.
 */
template <class T>
Variable<T> unsorted_segment_sum(const Tensor<T>& a, const Tensor<int>& ids, unsigned num);

/**
 * Calculate gelu(a) which equal to a * (0.5 * (1.0 + tanh(sqrt(2 / PI) * (a + 0.044715 * pow(a,
 * 3))))).
 * @param a A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> gelu(const Tensor<T>& a);

/**
 * Calculate gelu_accurate(a) which equal to 0.5 * a * (1 + erf(a / sqrt(2))).
 * @param a A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> gelu_accurate(const Tensor<T>& a);

/**
 * Absolute takes one input data (Tensor) and produces one output data (Tensor)
 * where the absolute is, y = abs(x), is applied to the tensor elementwise.
 * @param a A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> abs(const Tensor<T>& a);

/**
 * Calculates the arcsine (inverse of sine) of the given input tensor, element-wise.
 * @param a A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> asin(const Tensor<T>& a);

/**
 * Calculates the hyperbolic arcsine of the given input tensor element-wise.
 * @param a A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> asinh(const Tensor<T>& a);

/**
 * Calculates the arccosine (inverse of cosine) of the given input tensor, element-wise.
 * @param a A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> acos(const Tensor<T>& a);

/**
 * Calculates the hyperbolic arccosine of the given input tensor element-wise.
 * @param a A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> acosh(const Tensor<T>& a);

/**
 * Calculates the arctangent (inverse of tangent) of the given input tensor, element-wise.
 * @param a A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> atan(const Tensor<T>& a);

/**
 * Calculates the hyperbolic arctangent of the given input tensor element-wise.
 * @param a A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> atanh(const Tensor<T>& a);

/**
 * Calculate the sign of the given input tensor element-wise. If input > 0, output 1. if input < 0,
 * output -1. if input == 0, output 0.
 * @param a A tensor.
 * @return A variable.
 */
template <class T>
Variable<T> sign(const Tensor<T>& a);

/**
 * Returns the index with the largest value across axes of a tensor.
 * @param a A tensor.
 * @param axis An integer, the axis to reduce across.
 * @return A variable.
 */
template <class T>
Variable<int64_t> argmax(const Tensor<T>& a, size_t axis);

}  // namespace math
}  // namespace tfcc
