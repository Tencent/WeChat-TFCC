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

#include "tfcc_mklcellinterface.h"

#include "exceptions/tfcc_invalidargumenterror.h"
#include "framework/tfcc_mklsession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "interfaces/tfcc_mklinterfacehelper.h"
#include "kernel/tfcc_mklcellkernel.avx256.h"
#include "kernel/tfcc_mklcellkernel.avx512.h"
#include "kernel/tfcc_mklcellkernel.hpp"

namespace tfcc {

TFCC_MKL_HELPER_PRE_DEFINE(processLSTMCell);
TFCC_MKL_HELPER_PRE_DEFINE(processGRUCellGates);
TFCC_MKL_HELPER_PRE_DEFINE(processGRUCellCandidate);
TFCC_MKL_HELPER_PRE_DEFINE(processPyTorchGRUCell);

template <class T>
std::tuple<Variable<T>, Variable<T>> MKLCellInterface<T>::processLSTMCell(
    const Tensor<T>& stateC, const Tensor<T>& inputValue, const Tensor<T>& stateHValue,
    const Tensor<T>& bias, T forget) {
  if (stateC.shape().size() != 2) {
    throw InvalidArgumentError("invalid stateC");
  }
  unsigned batch = stateC.shape(0);
  unsigned units = stateC.shape(1);

  if (inputValue.shape().size() != 2 || inputValue.shape(0) != batch ||
      inputValue.shape(1) != units * 4) {
    throw InvalidArgumentError("invalid inputValue");
  }
  if (stateHValue.shape().size() != 2 || stateHValue.shape(0) != batch ||
      stateHValue.shape(1) != units * 4) {
    throw InvalidArgumentError("invalid stateHValue");
  }
  if (bias.shape().size() != 1 || bias.shape(0) != units * 4) {
    throw InvalidArgumentError("invalid bias");
  }

  Variable<T> result(stateC.shape());
  Variable<T> resultState(stateC.shape());

  mkl_async_auto_switch_wrapper(
      "lstm_kernel", TFCC_MKL_GET_RUNNER_HELPER(_MKLCellKernel, T, processLSTMCell)(), batch, units,
      stateC.data(), inputValue.data(), stateHValue.data(), bias.data(), forget, result.data(),
      resultState.data());
  return std::make_tuple(std::move(result), std::move(resultState));
}

template <class T>
Variable<T> MKLCellInterface<T>::processGRUCellGates(
    const Tensor<T>& state, const Tensor<T>& inputs, const Tensor<T>& value,
    const Tensor<T>& bias) {
  if (state.shape().size() != 2) {
    throw InvalidArgumentError("invalid state");
  }
  unsigned batch = state.shape(0);
  unsigned units = state.shape(1);
  if (inputs.shape().size() != 2 || inputs.shape(0) != batch) {
    throw InvalidArgumentError("invalid inputs");
  }
  unsigned inputSize = inputs.shape(1);
  if (value.shape().size() != 2 || value.shape(0) != batch || value.shape(1) != units * 2) {
    throw InvalidArgumentError("invalid value");
  }
  if (bias.shape().size() != 1 || bias.shape(0) != units * 2) {
    throw InvalidArgumentError("invalid bias");
  }
  Variable<T> result({batch, inputSize + units});

  mkl_async_auto_switch_wrapper(
      "gru_gates_kernel", TFCC_MKL_GET_RUNNER_HELPER(_MKLCellKernel, T, processGRUCellGates)(),
      batch, units, inputSize, state.data(), inputs.data(), value.data(), bias.data(),
      result.data());

  return result;
}

template <class T>
Variable<T> MKLCellInterface<T>::processGRUCellCandidate(
    const Tensor<T>& state, const Tensor<T>& value, const Tensor<T>& bias, const Tensor<T>& cValue,
    const Tensor<T>& cBias) {
  if (state.shape().size() != 2) {
    throw InvalidArgumentError("invalid state");
  }
  unsigned batch = state.shape(0);
  unsigned units = state.shape(1);
  if (value.shape().size() != 2 || value.shape(0) != batch || value.shape(1) != units * 2) {
    throw InvalidArgumentError("invalid value");
  }
  if (bias.shape().size() != 1 || bias.shape(0) != units * 2) {
    throw InvalidArgumentError("invalid bias");
  }
  if (cValue.shape().size() != 2 || cValue.shape(0) != batch || cValue.shape(1) != units) {
    throw InvalidArgumentError("invalid cValue");
  }
  if (cBias.shape().size() != 1 || cBias.shape(0) != units) {
    throw InvalidArgumentError("invalid cBias");
  }
  Variable<T> result({batch, units});

  mkl_async_auto_switch_wrapper(
      "gru_gates_kernel", TFCC_MKL_GET_RUNNER_HELPER(_MKLCellKernel, T, processGRUCellCandidate)(),
      batch, units, state.data(), value.data(), bias.data(), cValue.data(), cBias.data(),
      result.data());

  return result;
}

template <class T>
Variable<T> MKLCellInterface<T>::processPyTorchGRUCell(
    const Tensor<T>& state, const Tensor<T>& inputValue, const Tensor<T>& stateValue,
    const Tensor<T>& gateBias, const Tensor<T>& candidateIBias, const Tensor<T>& candidateHBias) {
  if (state.shape().size() != 2) {
    throw InvalidArgumentError("invalid state");
  }

  unsigned batch = state.shape(0);
  unsigned units = state.shape(1);

  if (inputValue.shape().size() != 2 || inputValue.shape(0) != batch ||
      inputValue.shape(1) != units * 3) {
    throw InvalidArgumentError("invalid inputValue");
  }
  if (stateValue.shape().size() != 2 || stateValue.shape(0) != batch ||
      stateValue.shape(1) != units * 3) {
    throw InvalidArgumentError("invalid stateValue");
  }
  if (gateBias.shape().size() != 1 || gateBias.shape(0) != units * 2) {
    throw InvalidArgumentError("invalid gateBias");
  }
  if (candidateIBias.shape().size() != 1 || candidateIBias.shape(0) != units) {
    throw InvalidArgumentError("invalid candidateIBias");
  }
  if (candidateHBias.shape().size() != 1 || candidateHBias.shape(0) != units) {
    throw InvalidArgumentError("invalid candidateHBias");
  }

  Variable<T> result(state.shape());

  mkl_async_auto_switch_wrapper(
      "pytorch_gru_kernel", TFCC_MKL_GET_RUNNER_HELPER(_MKLCellKernel, T, processPyTorchGRUCell)(),
      batch, units, state.data(), inputValue.data(), stateValue.data(), gateBias.data(),
      candidateIBias.data(), candidateHBias.data(), result.data());

  return result;
}

#define DEFINE_FUNC(type) template class MKLCellInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
