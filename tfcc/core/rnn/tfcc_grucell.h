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

#include "framework/tfcc_constant.h"
#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"

namespace tfcc {
namespace rnn {

template <class T>
class GRUCell {
  bool _initialized;
  unsigned _units;
  std::string _name;

  Constant<T>* _gatesKernel;
  Constant<T>* _gatesBias;

  Constant<T>* _candidateKernel;
  Constant<T>* _candidateBias;

 public:
  typedef Variable<T> State;

  explicit GRUCell(unsigned units);
  GRUCell(unsigned units, const std::string& name);
  State zeroState(unsigned batch);

  std::pair<Variable<T>, State> operator()(
      const Tensor<T>& input, const State& state, size_t step,
      const std::vector<unsigned>& lengths);
};

}  // namespace rnn
}  // namespace tfcc
