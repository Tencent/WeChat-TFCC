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
#include "framework/tfcc_types.h"
#include "framework/tfcc_variable.h"

namespace tfcc {
namespace signal {

/**
 * Computes the 1-dimensional discrete Fourier transform of a real-valued signal over the inner-most
 * dimension of a. NOTE: This function support broadcasting.
 * @param a A tensor.
 * @param length fft length.
 * @return A variable with shape [a.shape(0), a.shape(1), ..., 2].
 */
template <class T>
Variable<Complex<T>> rfft(const Tensor<T>& a, unsigned length);

/**
 * Computes the inverse 1-dimensional discrete Fourier transform of a real-valued signal over the
 * inner-most dimension of a. NOTE: This function support broadcasting.
 * @param a A tensor.
 * @param length fft length.
 * @return A variable with shape [a.shape(0), a.shape(1), ..., 2].
 */
template <class T>
Variable<T> irfft(const Tensor<Complex<T>>& a, unsigned length);

}  // namespace signal
}  // namespace tfcc
