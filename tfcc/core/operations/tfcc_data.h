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

#include <vector>

#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"

namespace tfcc {
namespace data {

/**
 * Set the value of specify variable.
 * @param a A Variable.
 * @param data The value to set.
 */
template <class T>
void set(Variable<T>& a, const T* data);

/**
 * Set the value of specify variable.
 * @param a A Variable.
 * @param data The data to set.
 */
template <class T>
void set(Variable<T>& a, const std::vector<T>& data);

/**
 * Set the value of specify variable.
 * @param a A Variable.
 * @param data The data to set.
 */
template <class T>
void set(Variable<T>& a, std::vector<T>&& data);

/**
 * Set the value of specify variable.
 * @param data The data to set.
 * @param shape The shape of data
 */
template <class T>
tfcc::Variable<T> set(const std::vector<T>& data, Shape shape);

/**
 * Get the value of specify tensor.
 * @param a A tensor.
 * @param data The value where to store.
 */
template <class T>
void get(const Tensor<T>& a, T* data);

/**
 * Get the value of specify tensor.
 * @param a A tensor.
 * @return The value of the tensor.
 */
template <class T>
std::vector<T> get(const Tensor<T>& a);

/**
 * Set a variable's value to zero.
 * @param a A variable.
 */
template <class T>
void zeros(Variable<T>& a);

/**
 * Set a variable's value to one.
 * @param a A variable.
 */
template <class T>
void ones(Variable<T>& a);

/**
 * Copy a tensor to a new variable.
 * @param a A tensor.
 * @return The new variable.
 */
template <class T>
Variable<T> copy(const Tensor<T>& a);

/**
 * Cast tensor to type T
 * @param a A tensor.
 * @return A vairable.
 */
template <class T>
Variable<T> cast(const Tensor<float>& a);

/**
 * Cast tensor to type T
 * @param a A tensor.
 * @return A vairable.
 */
template <class T>
Variable<T> cast(const Tensor<double>& a);

/**
 * Cast tensor to type T
 * @param a A tensor.
 * @return A vairable.
 */
template <class T>
Variable<T> cast(const Tensor<int8_t>& a);

/**
 * Cast tensor to type T
 * @param a A tensor.
 * @return A vairable.
 */
template <class T>
Variable<T> cast(const Tensor<uint8_t>& a);

/**
 * Cast tensor to type T
 * @param a A tensor.
 * @return A vairable.
 */
template <class T>
Variable<T> cast(const Tensor<int16_t>& a);

/**
 * Cast tensor to type T
 * @param a A tensor.
 * @return A vairable.
 */
template <class T>
Variable<T> cast(const Tensor<uint16_t>& a);

/**
 * Cast tensor to type T
 * @param a A tensor.
 * @return A vairable.
 */
template <class T>
Variable<T> cast(const Tensor<int32_t>& a);

/**
 * Cast tensor to type T
 * @param a A tensor.
 * @return A vairable.
 */
template <class T>
Variable<T> cast(const Tensor<uint32_t>& a);

/**
 * Cast tensor to type T
 * @param a A tensor.
 * @return A vairable.
 */
template <class T>
Variable<T> cast(const Tensor<int64_t>& a);

/**
 * Cast tensor to type T
 * @param a A tensor.
 * @return A vairable.
 */
template <class T>
Variable<T> cast(const Tensor<uint64_t>& a);

/**
 * Cast tensor to type T
 * @param a A tensor.
 * @return A vairable.
 */
template <class T>
Variable<uint8_t> cast_to_boolean(const Tensor<T>& a);

}  // namespace data
}  // namespace tfcc
