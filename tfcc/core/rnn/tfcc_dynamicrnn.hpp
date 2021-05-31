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

#include "framework/tfcc_scope.h"
#include "operations/tfcc_base.h"
#include "operations/tfcc_data.h"

namespace tfcc {
namespace rnn {

template <class Cell, class T>
std::tuple<Variable<T>, typename Cell::State> dynamic_rnn(
    Cell& cell, const Tensor<T>& inputs, const std::vector<unsigned> lengths,
    typename Cell::State&& initState, size_t inputsTimeAxis, size_t outputsTimeAxis,
    const std::string& name) {
  // TODO check arguments.
  // TODO optimize
  auto scopeG = Scope::scope(name);
  std::vector<unsigned> inputShape = inputs.shape().toVector();
  inputShape.erase(inputShape.begin() + inputsTimeAxis);
  std::vector<Variable<T>> outputs;
  auto state = std::move(initState);
  for (unsigned i = 0; i < inputs.shape(inputsTimeAxis); ++i) {
    Variable<T> in = base::slice(inputs, inputsTimeAxis, i, i + 1);
    in.reshape(Shape(inputShape));
    auto outputAndState = cell(in, std::move(state), i, lengths);
    state = std::move(outputAndState.second);
    Variable<T> ot = std::move(outputAndState.first);
    // TODO check outputs
    std::vector<unsigned> outputShape = ot.shape().toVector();
    outputShape.insert(outputShape.begin() + outputsTimeAxis, 1);
    ot.reshape(Shape(outputShape));
    outputs.emplace_back(std::move(ot));
  }

  // merge
  std::vector<const Tensor<T>*> values;
  for (auto& ot : outputs) values.push_back(&ot);
  Variable<T> result = base::concat(values, outputsTimeAxis);
  return std::make_tuple(std::move(result), std::move(state));
}

template <class CFW, class CBW, class T>
std::tuple<Variable<T>, Variable<T>, typename CFW::State, typename CBW::State>
bidirectional_dynamic_rnn(
    CFW& cellFW, CBW& cellBW, const Tensor<T>& inputs, const std::vector<unsigned> lengths,
    typename CFW::State&& initStateFW, typename CBW::State&& initStateBW, size_t inputsTimeAxis,
    size_t outputsTimeAxis, const std::string& name) {
  // TODO check arguments.
  // TODO optimize
  auto scopeG = Scope::scope(name);
  std::vector<unsigned> inputShape = inputs.shape().toVector();
  inputShape.erase(inputShape.begin() + inputsTimeAxis);
  std::vector<Variable<T>> outputsFW, outputsBW;
  auto stateFW = std::move(initStateFW);
  auto stateBW = std::move(initStateBW);
  {
    auto scopeG = Scope::scope("fw");
    for (unsigned i = 0; i < inputs.shape(inputsTimeAxis); ++i) {
      Variable<T> in = base::slice(inputs, inputsTimeAxis, i, i + 1);
      in.reshape(Shape(inputShape));
      auto outputAndState = cellFW(in, std::move(stateFW), i, lengths);
      stateFW = std::move(outputAndState.second);
      Variable<T> ot = std::move(outputAndState.first);
      // TODO check outputs
      std::vector<unsigned> outputShape = ot.shape().toVector();
      outputShape.insert(outputShape.begin() + outputsTimeAxis, 1);
      ot.reshape(Shape(outputShape));
      outputsFW.emplace_back(std::move(ot));
    }
  }

  {
    auto scopeG = Scope::scope("bw");
    for (unsigned i = 0; i < inputs.shape(inputsTimeAxis); ++i) {
      unsigned pos = inputs.shape(inputsTimeAxis) - i - 1;
      Variable<T> in = base::slice(inputs, inputsTimeAxis, pos, pos + 1);
      in.reshape(Shape(inputShape));
      auto outputAndState = cellBW(in, std::move(stateBW), pos, lengths);
      stateBW = std::move(outputAndState.second);
      Variable<T> ot = std::move(outputAndState.first);
      // TODO check outputs
      std::vector<unsigned> outputShape = ot.shape().toVector();
      outputShape.insert(outputShape.begin() + outputsTimeAxis, 1);
      ot.reshape(Shape(outputShape));
      outputsBW.emplace_back(std::move(ot));
    }
  }

  // merge
  std::vector<const Tensor<T>*> values;
  for (auto& ot : outputsFW) values.push_back(&ot);
  Variable<T> resultFW = base::concat(values, outputsTimeAxis);
  values.clear();

  for (auto it = outputsBW.rbegin(); it != outputsBW.rend(); ++it) values.push_back(&(*it));
  Variable<T> resultBW = base::concat(values, outputsTimeAxis);
  return std::make_tuple(
      std::move(resultFW), std::move(resultBW), std::move(stateFW), std::move(stateBW));
}

template <class Cell, class T>
std::tuple<Variable<T>, typename Cell::State> dynamic_rnn(
    Cell& cell, const Tensor<T>& inputs, typename Cell::State&& initState, size_t inputsTimeAxis,
    size_t outputsTimeAxis, const std::string& name) {
  return dynamic_rnn(cell, inputs, {}, std::move(initState), inputsTimeAxis, outputsTimeAxis, name);
}

template <class CFW, class CBW, class T>
std::tuple<Variable<T>, Variable<T>, typename CFW::State, typename CBW::State>
bidirectional_dynamic_rnn(
    CFW& cellFW, CBW& cellBW, const Tensor<T>& inputs, typename CFW::State&& initStateFW,
    typename CBW::State&& initStateBW, size_t inputsTimeAxis, size_t outputsTimeAxis,
    const std::string& name) {
  return bidirectional_dynamic_rnn(
      cellFW, cellBW, inputs, {}, std::move(initStateFW), std::move(initStateBW), inputsTimeAxis,
      outputsTimeAxis, name);
}

}  // namespace rnn
}  // namespace tfcc
