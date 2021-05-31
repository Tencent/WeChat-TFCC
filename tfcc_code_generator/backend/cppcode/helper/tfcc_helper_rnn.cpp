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

#include "tfcc_helper_rnn.h"

#include "tfcc_mkl.h"

#include "tfcc_helper_inner.h"

namespace tfcc {
namespace helper {
namespace rnn {

template <class T>
std::tuple<
    tfcc::Variable<T>, tfcc::Variable<T>, tfcc::Variable<T>, tfcc::Variable<T>, tfcc::Variable<T>,
    tfcc::Variable<T>>
bidirectional_lstm(
    const Tensor<T>& a, const Tensor<T>& inputKernel, const Tensor<T>& stateKernel,
    const Tensor<T>& bias, const Tensor<T>& ih, const Tensor<T>& ic) {
  tfcc::View<T> forwardInputKernel(inputKernel, inputKernel.shape(), 0, 1);
  forwardInputKernel.reshape({inputKernel.shape(1), inputKernel.shape(2)});
  tfcc::View<T> backwardInputKernel(inputKernel, inputKernel.shape(), 1, 2);
  backwardInputKernel.reshape({inputKernel.shape(1), inputKernel.shape(2)});
  tfcc::View<T> forwardStateKernel(stateKernel, stateKernel.shape(), 0, 1);
  forwardStateKernel.reshape({stateKernel.shape(1), stateKernel.shape(2)});
  tfcc::View<T> backwardStateKernel(stateKernel, stateKernel.shape(), 1, 2);
  backwardStateKernel.reshape({stateKernel.shape(1), stateKernel.shape(2)});
  tfcc::View<T> forwardBias(bias, bias.shape(), 0, 1);
  forwardBias.reshape({bias.shape(1)});
  tfcc::View<T> backwardBias(bias, bias.shape(), 1, 2);
  backwardBias.reshape({bias.shape(1)});
  tfcc::View<T> forwardIH(ih, ih.shape(), 0, 1);
  forwardIH.reshape({ih.shape(1), ih.shape(2)});
  tfcc::View<T> backwardIH(ih, ih.shape(), 1, 2);
  backwardIH.reshape({ih.shape(1), ih.shape(2)});
  tfcc::View<T> forwardIC(ic, ic.shape(), 0, 1);
  forwardIC.reshape({ic.shape(1), ic.shape(2)});
  tfcc::View<T> backwardIC(ic, ic.shape(), 1, 2);
  backwardIC.reshape({ic.shape(1), ic.shape(2)});

  tfcc::Variable<T> forwardOutput, forwardH, forwardC;
  std::tie(forwardOutput, forwardH, forwardC) = tfcc::rnn::lstm_forward(
      a, forwardInputKernel, forwardStateKernel, forwardBias, forwardIH, forwardIC);
  tfcc::Variable<T> backwardOutput, backwardH, backwardC;
  std::tie(backwardOutput, backwardH, backwardC) = tfcc::rnn::lstm_backward(
      a, backwardInputKernel, backwardStateKernel, backwardBias, backwardIH, backwardIC);
  return std::make_tuple(
      std::move(forwardOutput), std::move(backwardOutput), std::move(forwardH),
      std::move(backwardH), std::move(forwardC), std::move(backwardC));
}

#define DEFINE_FUNC(type)                                                                      \
  template std::tuple<                                                                         \
      tfcc::Variable<type>, tfcc::Variable<type>, tfcc::Variable<type>, tfcc::Variable<type>,  \
      tfcc::Variable<type>, tfcc::Variable<type>>                                              \
  bidirectional_lstm(                                                                          \
      const Tensor<type>& a, const Tensor<type>& inputKernel, const Tensor<type>& stateKernel, \
      const Tensor<type>& bias, const Tensor<type>& ih, const Tensor<type>& ic);

TFCC_HELPER_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace rnn
}  // namespace helper
}  // namespace tfcc
