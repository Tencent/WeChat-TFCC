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
namespace rnn {

template <class Cell, class T>
std::tuple<Variable<T>, typename Cell::State> dynamic_rnn(
    Cell& cell, const Tensor<T>& inputs, const std::vector<unsigned> lengths,
    typename Cell::State&& initState, size_t inputsTimeAxis, size_t outputsTimeAxis,
    const std::string& name);

template <class CFW, class CBW, class T>
std::tuple<Variable<T>, Variable<T>, typename CFW::State, typename CBW::State>
bidirectional_dynamic_rnn(
    CFW& cellFW, CBW& cellBW, const Tensor<T>& inputs, const std::vector<unsigned> lengths,
    typename CFW::State&& initStateFW, typename CBW::State&& initStateBW, size_t inputsTimeAxis,
    size_t outputsTimeAxis, const std::string& name);

template <class Cell, class T>
std::tuple<Variable<T>, typename Cell::State> dynamic_rnn(
    Cell& cell, const Tensor<T>& inputs, typename Cell::State&& initState, size_t inputsTimeAxis,
    size_t outputsTimeAxis, const std::string& name);

template <class CFW, class CBW, class T>
std::tuple<Variable<T>, Variable<T>, typename CFW::State, typename CBW::State>
bidirectional_dynamic_rnn(
    CFW& cellFW, CBW& cellBW, const Tensor<T>& inputs, typename CFW::State&& initStateFW,
    typename CBW::State&& initStateBW, size_t inputsTimeAxis, size_t outputsTimeAxis,
    const std::string& name);

}  // namespace rnn
}  // namespace tfcc

#include "tfcc_dynamicrnn.hpp"
