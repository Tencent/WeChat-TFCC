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

#include <string>
#include <vector>

#include "framework/tfcc_constant.h"
#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"

namespace tfcc {
namespace rnn {

template <class T>
class LSTMCell {
  bool _initialized;
  Constant<T>* _kernel;
  Constant<T>* _bias;
  unsigned _units;
  std::string _name;

 public:
  struct State {
    Variable<T> cs, h;

    State() {}
    State(unsigned batch, unsigned units) : cs({batch, units}), h({batch, units}) {}
    State(State&&) noexcept = default;
    State(const State&) = delete;

    State& operator=(State&&) noexcept = default;
    State& operator=(const State&) = delete;
  };

  /**
   * @see LSTMCell(unsigned units, std::string name)
   */
  explicit LSTMCell(unsigned units);

  /**
   * @param units The number of units in the LSTM cell. If zero, units will be initialized by auto.
   * @param name Name of the layer. The default value is 'lstm_cell'
   */
  LSTMCell(unsigned units, std::string name);

  State zeroState(unsigned batch);
  std::pair<Variable<T>, State> operator()(
      const Tensor<T>& input, const State& state, size_t step,
      const std::vector<unsigned>& lengths);
};

}  // namespace rnn
}  // namespace tfcc
