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
#include "tfcc.h"

namespace tfcc {
namespace helper {
namespace rnn {

template <class T>
std::tuple<
    tfcc::Variable<T>, tfcc::Variable<T>, tfcc::Variable<T>, tfcc::Variable<T>, tfcc::Variable<T>,
    tfcc::Variable<T>>
bidirectional_lstm(
    const Tensor<T>& a, const Tensor<T>& inputKernel, const Tensor<T>& stateKernel,
    const Tensor<T>& bias, const Tensor<T>& ih, const Tensor<T>& ic);

}  // namespace rnn
}  // namespace helper
}  // namespace tfcc
