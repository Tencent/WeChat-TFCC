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

#include "framework/tfcc_shape.h"
#include "framework/tfcc_variable.h"

namespace tfcc {
namespace random {

/**
 * Create a random variable from a normal distribution.
 * This operation just support floating-point type.
 * @param s The shape of output variable.
 * @param mean The mean of the normal distribution.
 * @param stddev The standard deviation of the normal distribution.
 * @return A variable.
 */
template <class T>
Variable<T> normal(Shape s, T mean, T stddev);

/**
 * Create a random variable from a logistics distribution.
 * This operation just support floating-point type.
 * @param s The shape of output variable.
 * @param minVal The min value of the logistics distribution.
 * @param maxVal The max value of the logistics distribution.
 * @return A variable.
 */
template <class T>
Variable<T> logistics(Shape s, T minVal, T maxVal);

/**
 * Create a random variable from a binary distribution.
 * This operation just support floating-point type.
 * @param s The shape of output variable.
 * @param a The first value of the binary distribution.
 * @param b The second value of the binary distribution.
 * @return A variable.
 */
template <class T>
Variable<T> binary(Shape s, T a, T b);

/**
 * Create a random variable from a uniform distribution.
 * This operation just support floating-point type.
 * @param s The shape of output variable.
 * @param minVal The min value of the uniform distribution.
 * @param maxVal The max value of the uniform distribution.
 * @return A variable.
 */
template <class T>
Variable<T> uniform(Shape s, T minVal, T maxVal);

}  // namespace random
}  // namespace tfcc
