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

#include <initializer_list>
#include <vector>

#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"

namespace tfcc {
namespace base {

/**
 * Extracts a slice from a tensor.
 * @param a A tensor.
 * @param axis The axis to slice along.
 * @param start The start offset at specify dimension.
 * @param end The end location at specify dimension.
 * @return A variable.
 */
template <class T>
Variable<T> slice(const Tensor<T>& a, size_t axis, unsigned start, unsigned end);

/**
 * Splits a tensor into sub variables.
 * @param a A tensor.
 * @param num Count of the tensor is split along dimension axis.
 * @param axis The axis to split along.
 * @return A list of variable.
 */
template <class T>
std::vector<Variable<T>> split(const Tensor<T>& a, size_t num, size_t axis);

/**
 * Splits a tensor into sub variables.
 * @param a A tensor.
 * @param sizes Each size along dimension axis of result.
 * @param axis The axis to split along.
 * @return A list of variable.
 */
template <class T>
std::vector<Variable<T>> split(const Tensor<T>& a, const std::vector<unsigned>& sizes, size_t axis);

/**
 * Concatenates tensors along one dimension.
 * @param values A list of tensor pointer.
 * @param axis Dimension along which to concatenate.
 * @return A variable resulting from concatenation of the input tensors.
 */
template <class T>
Variable<T> concat(const std::vector<const Tensor<T>*>& values, size_t axis);

/**
 * @see concat(const std::vector<const Tensor<T>*>& values, size_t axis)
 */
template <class T>
Variable<T> concat(std::initializer_list<const Tensor<T>*> values, size_t axis);

/**
 * @see concat(const std::vector<const Tensor<T>*>& values, size_t axis)
 */
template <class T>
Variable<T> concat(const std::vector<Variable<T>>& values, size_t axis);

/**
 * Pads a tensor with zero.
 * @param a A Tensor.
 * @param axis The axis to pad.
 * @param paddingHead The padding number before the start of specify dimension.
 * @param paddingEnd The padding number before the end of4 specify dimension.
 * @return A variable.
 */
template <class T>
Variable<T> pad(const Tensor<T>& a, size_t axis, unsigned paddingHead, unsigned paddingEnd);

/**
 * Assign tensor a to variable b.
 * @param a A tensor.
 * @param axis The axis to assign.
 * @param start The start of specify dimension.
 * @param b A variable
 */
template <class T>
void assign_to(const Tensor<T>& a, size_t axis, unsigned start, Variable<T>& b);

/**
 * Transpose a tensor.
 * @param a A tensor.
 * @param perm A permutation of the dimensions of the tensor.
 * @return A transposed variable.
 */
template <class T>
Variable<T> transpose(const Tensor<T>& a, const std::vector<size_t>& perm);

/**
 * Stacks a list of rank-R tensors into one rank-(R+1) tensor.
 * @param values A list of tensor objects with the same shape.
 * @param axis The axis to stack along.
 * @return A stacked variable.
 */
template <class T>
Variable<T> stack(const std::vector<const Tensor<T>*>& values, size_t axis);

/**
 * Stacks a list of rank-R tensors into one rank-(R+1) tensor.
 * @param values A list of tensor objects with the same shape.
 * @param axis The axis to stack along.
 * @return A stacked variable.
 */
template <class T>
Variable<T> stack(const std::initializer_list<const Tensor<T>*>& values, size_t axis);

/**
 * Stacks a list of rank-R tensors into one rank-(R+1) tensor.
 * @param values A list of tensor objects with the same shape.
 * @param axis The axis to stack along.
 * @return A stacked variable.
 */
template <class T>
Variable<T> stack(const std::vector<Variable<T>>& values, size_t axis);

/**
 * Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
 * @param a A rank R > 0 tensor to be unstacked.
 * @param axis The axis to unstack along.
 * @return The list of variable objects unstacked from a.
 */
template <class T>
std::vector<Variable<T>> unstack(const Tensor<T>& a, size_t axis);

/**
 * Lower triangle of an array.
 * Return a copy of a matrix with the elements below the k-th diagonal zeroed.
 * @see https://numpy.org/doc/stable/reference/generated/numpy.tril.html#numpy.tril
 * @param a A 2-dims tensor.
 * @param k Diagonal above which to zero elements. k = 0 (the default) is the main diagonal, k < 0
 * is below it and k > 0 is above.
 * @return Lower triangle of a, of same shape as a.
 */
template <class T>
Variable<T> tril(const Tensor<T>& a, int64_t k);

/**
 * Upper triangle of an array.
 * Return a copy of a matrix with the elements below the k-th diagonal zeroed.
 * @see https://numpy.org/doc/stable/reference/generated/numpy.tril.html#numpy.tril
 * @param a A 2-dims tensor.
 * @param k Diagonal above which to zero elements. k = 0 (the default) is the main diagonal, k < 0
 * is below it and k > 0 is above.
 * @return Upper triangle of a, of same shape as a.
 */
template <class T>
Variable<T> triu(const Tensor<T>& a, int64_t k);

}  // namespace base

}  // namespace tfcc
