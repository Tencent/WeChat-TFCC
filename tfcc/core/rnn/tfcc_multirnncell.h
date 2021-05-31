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
#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"

namespace tfcc {
namespace rnn {

template <class T, class Cell, class... Args>
class MultiRNNCell {
  MultiRNNCell<T, Args...> _otherCells;
  Cell& _cell;
  std::string _name;

 public:
  struct State {
    typename Cell::State currentState;
    typename MultiRNNCell<T, Args...>::State otherStates;
  };

  MultiRNNCell(size_t i, Cell& cell, Args&... otherCells)
      : _otherCells(i + 1, otherCells...), _cell(cell), _name("cell_" + std::to_string(i)) {}

  MultiRNNCell(Cell& cell, Args&... otherCells) : MultiRNNCell(0, cell, otherCells...) {}

  State zeroState(unsigned batch) {
    return State{_cell.zeroState(batch), _otherCells.zeroState(batch)};
  }

  std::pair<Variable<T>, State> operator()(
      const Tensor<T>& input, const State& state, size_t step,
      const std::vector<unsigned>& lengths) {
    auto result = processSelf(input, state, step, lengths);
    auto otherResult = _otherCells(std::get<0>(result), state.otherStates, step, lengths);
    State nextState{std::move(std::get<1>(result)), std::move(std::get<1>(otherResult))};
    return std::make_pair(std::move(std::get<0>(otherResult)), std::move(nextState));
  }

 private:
  auto processSelf(
      const Tensor<T>& input, const State& state, size_t step, const std::vector<unsigned>& lengths)
      -> decltype(_cell(input, state.currentState, step, lengths)) {
    auto scopeGuard = Scope::scope("multi_rnn_cell");
    auto scopeGuard2 = Scope::scope(_name);
    return _cell(input, state.currentState, step, lengths);
  }
};

template <class T, class Cell>
class MultiRNNCell<T, Cell> {
  Cell& _cell;
  std::string _name;

 public:
  typedef typename Cell::State State;

  MultiRNNCell(size_t i, Cell& cell) : _cell(cell), _name("cell_" + std::to_string(i)) {}

  MultiRNNCell(Cell& cell) : MultiRNNCell(0, cell) {}

  MultiRNNCell(MultiRNNCell&&) = default;
  MultiRNNCell(const MultiRNNCell&) = delete;

  State zeroState(unsigned batch) { return _cell.zeroState(batch); }
  std::pair<Variable<T>, State> operator()(
      const Tensor<T>& input, const State& state, size_t step,
      const std::vector<unsigned>& lengths) {
    auto scopeGuard = Scope::scope("multi_rnn_cell");
    auto scopeGuard2 = Scope::scope(_name);

    return _cell(input, state, step, lengths);
  }
};

}  // namespace rnn
}  // namespace tfcc
