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
class BasicInterface {
 public:
  BasicInterface() {}
  BasicInterface(const BasicInterface&) = delete;
  virtual ~BasicInterface() {}

  BasicInterface& operator=(const BasicInterface&) = delete;

  /**
   * return a[,...,start:end,...]
   */
  virtual Variable<T> slice(const Tensor<T>& a, size_t axis, unsigned start, unsigned end);

  /**
   * b[,...,start:start + a.shape(axis),...,] = a
   */
  virtual void assignTo(const Tensor<T>& a, size_t axis, unsigned start, Variable<T>& b);

  /**
   * equal to tf.transpose(a, perm)
   */
  virtual Variable<T> transpose(const Tensor<T>& a, const std::vector<size_t>& perm);

  /**
   * replace values which are greater than max to max, and
   * replace values which are less than min to min
   */
  virtual Variable<T> clip(const Tensor<T>& a, T minValue, T maxValue);

  /**
   * Concatenates tensors along one dimension.
   * @param values A list of tensor pointer.
   * @param axis Dimension along which to concatenate.
   * @return A variable resulting from concatenation of the input tensors.
   */
  virtual Variable<T> concat(const std::vector<const Tensor<T>*>& values, size_t axis);

  /**
   * Return the elements where condition is True (multiplexing x and y). Where behaves like
   * numpy.where .
   * @param condition A uint8_t tensor.
   * @param x A tensor.
   * @param y A tensor.
   * @return A tensor.
   */
  virtual Variable<T> where(
      const Tensor<uint8_t>& condition, const Tensor<T>& x, const Tensor<T>& y);

  /**
   * Return the elements where condition is True (multiplexing x and y). Where behaves like
   * numpy.where .
   * @param condition A uint8_t tensor.
   * @param x A value.
   * @param y A tensor.
   * @return A tensor.
   */
  virtual Variable<T> where(const Tensor<uint8_t>& condition, T x, const Tensor<T>& y);

  /**
   * Return the elements where condition is True (multiplexing x and y). Where behaves like
   * numpy.where .
   * @param condition A uint8_t tensor.
   * @param x A tensor.
   * @param y A value.
   * @return A tensor.
   */
  virtual Variable<T> where(const Tensor<uint8_t>& condition, const Tensor<T>& x, T y);

  /**
   * Absolute takes one input data (Tensor) and produces one output data (Tensor)
   * where the absolute is, y = abs(x), is applied to the tensor elementwise.
   * @param a A tensor.
   * @return A variable.
   */
  virtual Variable<T> abs(const Tensor<T>& a);

  /**
   * Lower triangle of an array.
   * Return a copy of a matrix with the elements below the k-th diagonal zeroed.
   * @see https://numpy.org/doc/stable/reference/generated/numpy.tril.html#numpy.tril
   * @param a A 2-dims tensor.
   * @param k Diagonal above which to zero elements. k = 0 (the default) is the main diagonal, k < 0
   * is below it and k > 0 is above.
   * @return Lower triangle of a, of same shape as a.
   */
  virtual Variable<T> tril(const Tensor<T>& a, int64_t k);

  /**
   * Upper triangle of an array.
   * Return a copy of a matrix with the elements below the k-th diagonal zeroed.
   * @see https://numpy.org/doc/stable/reference/generated/numpy.triu.html
   * @param a A 2-dims tensor.
   * @param k Diagonal above which to zero elements. k = 0 (the default) is the main diagonal, k < 0
   * is below it and k > 0 is above.
   * @return Upper triangle of a, of same shape as a.
   */
  virtual Variable<T> triu(const Tensor<T>& a, int64_t k);

  /**
   * Returns the index with the largest value across axes of a tensor.
   * @param a A tensor.
   * @param axis An integer, the axis to reduce across.
   * @return A variable.
   */
  virtual Variable<int64_t> argmax(const Tensor<T>& a, size_t axis);

  /**
   * @deprecated
   */
  virtual std::tuple<Variable<T>, Variable<uint32_t>> topK(const Tensor<T>& a, unsigned k);
};

}  // namespace tfcc
