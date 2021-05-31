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

#include "tfcc_rnn.h"

#include "exceptions/tfcc_invalidargumenterror.h"
#include "framework/tfcc_types.h"
#include "framework/tfcc_view.h"
#include "interfaces/tfcc_basicinterface.h"
#include "interfaces/tfcc_blasinterface.h"
#include "interfaces/tfcc_cellinterface.h"
#include "interfaces/tfcc_datainterface.h"
#include "operations/tfcc_operation.h"

namespace tfcc {
namespace rnn {

template <class T>
std::tuple<Variable<T>, Variable<T>, Variable<T>> lstm_forward(
    const Tensor<T>& a, const Tensor<T>& inputKernel, const Tensor<T>& stateKernel,
    const Tensor<T>& bias, const Tensor<T>& ih, const Tensor<T>& ic) {
  if (a.shape().size() != 3) {
    throw InvalidArgumentError("invalid input");
  }
  unsigned batch = a.shape(1);
  unsigned length = a.shape(2);
  if (ih.shape() != ic.shape() || ih.shape().size() != 2 || ih.shape(0) != batch) {
    throw InvalidArgumentError("invalid ih or ic");
  }
  unsigned hidden = ih.shape(1);
  if (inputKernel.shape().size() != 2 || inputKernel.shape(0) != length ||
      inputKernel.shape(1) != hidden * 4) {
    throw InvalidArgumentError("invalid input kernel");
  }
  if (stateKernel.shape().size() != 2 || stateKernel.shape(0) != hidden ||
      stateKernel.shape(1) != hidden * 4) {
    throw InvalidArgumentError("invalid state kernel");
  }
  if (bias.shape().size() != 1 || bias.shape(0) != 4 * hidden) {
    throw InvalidArgumentError("invalid bias");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();

  std::vector<Variable<T>> outputs;
  Variable<T> h, c;
  {
    tfcc::View<T> input(a, a.shape(), 0, 1);
    input.reshape({batch, length});
    std::tie(h, c) = interface.getCellInterface().processLSTMCell(
        ic, interface.getBlasInterface().matmul(input, inputKernel),
        interface.getBlasInterface().matmul(ih, stateKernel), bias, 0.0f);
    outputs.emplace_back(interface.getDataInterface().copy(h));
  }
  for (unsigned i = 1; i < a.shape(0); ++i) {
    tfcc::View<T> input(a, a.shape(), i, i + 1);
    input.reshape({batch, length});
    std::tie(h, c) = interface.getCellInterface().processLSTMCell(
        c, interface.getBlasInterface().matmul(input, inputKernel),
        interface.getBlasInterface().matmul(h, stateKernel), bias, 0.0f);
    outputs.emplace_back(interface.getDataInterface().copy(h));
  }
  std::vector<const Tensor<T>*> values;
  for (auto& output : outputs) {
    values.emplace_back(&output);
  }
  Variable<T> result = interface.getBasicInterface().concat(values, 0);
  result.reshape({a.shape(0), batch, hidden});
  return std::make_tuple(std::move(result), std::move(h), std::move(c));
}

template <class T>
std::tuple<Variable<T>, Variable<T>, Variable<T>> lstm_backward(
    const Tensor<T>& a, const Tensor<T>& inputKernel, const Tensor<T>& stateKernel,
    const Tensor<T>& bias, const Tensor<T>& ih, const Tensor<T>& ic) {
  if (a.shape().size() != 3) {
    throw InvalidArgumentError("invalid input");
  }
  unsigned batch = a.shape(1);
  unsigned length = a.shape(2);
  if (ih.shape() != ic.shape() || ih.shape().size() != 2 || ih.shape(0) != batch) {
    throw InvalidArgumentError("invalid ih or ic");
  }
  unsigned hidden = ih.shape(1);
  if (inputKernel.shape().size() != 2 || inputKernel.shape(0) != length ||
      inputKernel.shape(1) != hidden * 4) {
    throw InvalidArgumentError("invalid input kernel");
  }
  if (stateKernel.shape().size() != 2 || stateKernel.shape(0) != hidden ||
      stateKernel.shape(1) != hidden * 4) {
    throw InvalidArgumentError("invalid state kernel");
  }
  if (bias.shape().size() != 1 || bias.shape(0) != 4 * hidden) {
    throw InvalidArgumentError("invalid bias");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();

  std::vector<Variable<T>> outputs;
  Variable<T> h, c;
  {
    tfcc::View<T> input(a, a.shape(), a.shape(0) - 1, a.shape(0));
    input.reshape({batch, length});
    std::tie(h, c) = interface.getCellInterface().processLSTMCell(
        ic, interface.getBlasInterface().matmul(input, inputKernel),
        interface.getBlasInterface().matmul(ih, stateKernel), bias, 0.0f);
    outputs.emplace_back(interface.getDataInterface().copy(h));
  }
  for (unsigned i = 1; i < a.shape(0); ++i) {
    tfcc::View<T> input(a, a.shape(), a.shape(0) - i - 1, a.shape(0) - i);
    input.reshape({batch, length});
    std::tie(h, c) = interface.getCellInterface().processLSTMCell(
        c, interface.getBlasInterface().matmul(input, inputKernel),
        interface.getBlasInterface().matmul(h, stateKernel), bias, 0.0f);
    outputs.emplace_back(interface.getDataInterface().copy(h));
  }
  std::vector<const Tensor<T>*> values;
  for (auto& output : outputs) {
    values.emplace_back(&output);
  }
  Variable<T> result = interface.getBasicInterface().concat(values, 0);
  result.reshape({a.shape(0), batch, hidden});
  return std::make_tuple(std::move(result), std::move(h), std::move(c));
}

#define DEFINE_FUNC(type)                                                                      \
  template std::tuple<Variable<type>, Variable<type>, Variable<type>> lstm_forward(            \
      const Tensor<type>& a, const Tensor<type>& inputKernel, const Tensor<type>& stateKernel, \
      const Tensor<type>& bias, const Tensor<type>& ih, const Tensor<type>& ic);               \
  template std::tuple<Variable<type>, Variable<type>, Variable<type>> lstm_backward(           \
      const Tensor<type>& a, const Tensor<type>& inputKernel, const Tensor<type>& stateKernel, \
      const Tensor<type>& bias, const Tensor<type>& ih, const Tensor<type>& ic);

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace rnn
}  // namespace tfcc
