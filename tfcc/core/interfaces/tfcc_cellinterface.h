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

template <class T>
class CellInterface {
 public:
  CellInterface() {}
  CellInterface(const CellInterface&) = delete;
  virtual ~CellInterface() {}

  CellInterface& operator=(const CellInterface&) = delete;

  /**
   * Calculate lstmcell.
   * xh = inputValue + stateHValue + inputBias + stateHBias
   * i, ci, f, o = split(xh, axis=1, count=4)
   * i = sigmoid(i)
   * f = sigmoid(f + forget)
   * ci = tanh(ci)
   * cs = ci * i + stateC * f
   * co = tanh(cs)
   * o = sigmoid(o)
   * h = co * o
   * return h, cs
   */
  virtual std::tuple<Variable<T>, Variable<T>> processLSTMCell(
      const Tensor<T>& stateC, const Tensor<T>& inputValue, const Tensor<T>& stateHValue,
      const Tensor<T>& bias, T forget);

  /**
   * Calculate grucell gates.
   * value = sigmoid(value + bias)
   * r = value[:,:units]
   * rState = r * state
   * return concat([inputs, rState], axis=1)
   */
  virtual Variable<T> processGRUCellGates(
      const Tensor<T>& state, const Tensor<T>& inputs, const Tensor<T>& value,
      const Tensor<T>& bias);

  /**
   * Calculate grucell candidate.
   * value = sigmoid(value + bias)
   * u = value[:,units:]
   * c = tanh(cValue + cBias)
   * return u * state + (1 - u) * c
   */
  virtual Variable<T> processGRUCellCandidate(
      const Tensor<T>& state, const Tensor<T>& value, const Tensor<T>& bias,
      const Tensor<T>& cValue, const Tensor<T>& cBias);

  /**
   * Calculate pytorch-version grucell.
   * units = candidateIBias.size()
   * tmp = sigmoid(inputValue[,:units * 2] + stateValue[,:units * 2] + gateBias)
   * r = tmp[,:units]
   * z = tmp[,units:]
   * ni = inputValue[,units * 2:] + candidateIBias
   * nh = stateValue[,units * 2:] + candidateHBias
   * n = tanh(ni + r * nh);
   * return n + z * (state - n)
   */
  virtual Variable<T> processPyTorchGRUCell(
      const Tensor<T>& state, const Tensor<T>& inputValue, const Tensor<T>& stateValue,
      const Tensor<T>& gateBias, const Tensor<T>& candidateIBias, const Tensor<T>& candidateHBias);
};

}  // namespace tfcc
