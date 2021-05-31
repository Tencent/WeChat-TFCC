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

#include "tfcc_lstmcell.h"

#include <utility>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "framework/tfcc_scope.h"
#include "framework/tfcc_types.h"
#include "framework/tfcc_view.h"
#include "interfaces/tfcc_cellinterface.h"
#include "operations/tfcc_base.h"
#include "operations/tfcc_blas.h"
#include "operations/tfcc_data.h"
#include "operations/tfcc_math.h"
#include "operations/tfcc_operation.h"
#include "operations/tfcc_operator.h"

namespace tfcc {
namespace rnn {

template <class T>
LSTMCell<T>::LSTMCell(unsigned units) : LSTMCell(units, "lstm_cell") {}

template <class T>
LSTMCell<T>::LSTMCell(unsigned units, std::string name)
    : _initialized(false),
      _kernel(nullptr),
      _bias(nullptr),
      _units(units),
      _name(std::move(name)) {}

template <class T>
typename LSTMCell<T>::State LSTMCell<T>::zeroState(unsigned batch) {
  State state(batch, _units);
  data::zeros(state.cs);
  data::zeros(state.h);
  return state;
}

template <class T>
std::pair<Variable<T>, typename LSTMCell<T>::State> LSTMCell<T>::operator()(
    const Tensor<T>& inputs, const LSTMCell<T>::State& state, size_t step,
    const std::vector<unsigned>& lengths) {
  if (inputs.shape().size() != 2) {
    throw InvalidArgumentError("invalid inputs");
  }
  if (inputs.shape(0) != state.cs.shape(0)) {
    throw InvalidArgumentError("state and inputs don't match");
  }
  if (lengths.size() != 0 && lengths.size() != inputs.shape(0)) {
    throw InvalidArgumentError("invalid lengths");
  }
  auto scopeG = Scope::scope(_name);

  if (!_initialized) {
    _kernel = &Constant<T>::getConstant("kernel");
    _bias = &Constant<T>::getConstant("bias");
    _initialized = true;
  }

  unsigned batch = inputs.shape(0);
  State nextState(batch, _units);
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  std::tie(nextState.h, nextState.cs) = interface.getCellInterface().processLSTMCell(
      state.cs, blas::matmul(inputs, View<T>(*_kernel, _kernel->shape(), 0, inputs.shape(1))),
      blas::matmul(state.h, View<T>(*_kernel, _kernel->shape(), inputs.shape(1))), *_bias, 1.0f);

  if (lengths.size() > 0) {
    for (unsigned i = 0; i < lengths.size(); ++i) {
      if (step < lengths[i]) {
        continue;
      }
      View<T> vcs(state.cs, state.cs.shape(), i, i + 1);
      base::assign_to(vcs, 0, i, nextState.cs);
      View<T> vh(state.h, state.h.shape(), i, i + 1);
      base::assign_to(vh, 0, i, nextState.h);
    }
  }

  Variable<T> output = data::copy(nextState.h);
  return std::make_pair(std::move(output), std::move(nextState));
}

#define DEFINE_FUNC(type) template class LSTMCell<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace rnn
}  // namespace tfcc
