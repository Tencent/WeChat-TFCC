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
namespace rnn {

/**
 * LSTM forward with ICFO kernel format.
 * @param a Input tensor with shape [seq_length, batch_size, hidden_size]
 * @param inputKernel Input kernel tensor with shape [input_size, 4 * hidden_size]
 * @param stateKernel State kernel tensor with shape [hidden_size, 4 * hidden_size]
 * @param bias Bias tensor with shape [4 * hidden_size]
 * @param ih Initial h tensor with shape [batch_size, hidden_size]
 * @param ic Initial c tensor with shape [batch_size, hidden_size]
 * @return Tuple of (output, h, c), output with shape [seq_length, batch_size, hidden_size] and h, c
 * with shape [batch_size, hidden_size]
 */
template <class T>
std::tuple<Variable<T>, Variable<T>, Variable<T>> lstm_forward(
    const Tensor<T>& a, const Tensor<T>& inputKernel, const Tensor<T>& stateKernel,
    const Tensor<T>& bias, const Tensor<T>& ih, const Tensor<T>& ic);

/**
 * LSTM backward with ICFO kernel format.
 * @param a Input tensor with shape [seq_length, batch_size, hidden_size]
 * @param inputKernel Input kernel tensor with shape [input_size, 4 * hidden_size]
 * @param stateKernel State kernel tensor with shape [hidden_size, 4 * hidden_size]
 * @param bias Bias tensor with shape [4 * hidden_size]
 * @param ih Initial h tensor with shape [batch_size, hidden_size]
 * @param ic Initial c tensor with shape [batch_size, hidden_size]
 * @return Tuple of (output, h, c), output with shape [seq_length, batch_size, hidden_size] and h, c
 * with shape [batch_size, hidden_size]
 */
template <class T>
std::tuple<Variable<T>, Variable<T>, Variable<T>> lstm_backward(
    const Tensor<T>& a, const Tensor<T>& inputKernel, const Tensor<T>& stateKernel,
    const Tensor<T>& bias, const Tensor<T>& ih, const Tensor<T>& ic);

}  // namespace rnn
}  // namespace tfcc
