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

#include "tfcc_cellinterface.h"

#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_types.h"

namespace tfcc {

template <class T>
std::tuple<Variable<T>, Variable<T>> CellInterface<T>::processLSTMCell(
    const Tensor<T>& stateC, const Tensor<T>& inputValue, const Tensor<T>& stateHValue,
    const Tensor<T>& bias, T forget) {
  throw NotImplementedError();
}

template <class T>
Variable<T> CellInterface<T>::processGRUCellGates(
    const Tensor<T>& state, const Tensor<T>& inputs, const Tensor<T>& value,
    const Tensor<T>& bias) {
  throw NotImplementedError();
}

template <class T>
Variable<T> CellInterface<T>::processGRUCellCandidate(
    const Tensor<T>& state, const Tensor<T>& value, const Tensor<T>& bias, const Tensor<T>& c,
    const Tensor<T>& cBias) {
  throw NotImplementedError();
}

template <class T>
Variable<T> CellInterface<T>::processPyTorchGRUCell(
    const Tensor<T>& state, const Tensor<T>& inputValue, const Tensor<T>& stateValue,
    const Tensor<T>& gateBias, const Tensor<T>& candidateIBias, const Tensor<T>& candidateHBias) {
  throw NotImplementedError();
}

#define DEFINE_FUNC(type) template class CellInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);
TFCC_FOR_COMPLEX_TYPES(DEFINE_FUNC);

}  // namespace tfcc
