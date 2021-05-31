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

#include "tfcc_pytorchgrucell.h"

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
PyTorchGRUCell<T>::PyTorchGRUCell(unsigned units) : PyTorchGRUCell(units, "gru_cell") {}

template <class T>
PyTorchGRUCell<T>::PyTorchGRUCell(unsigned units, const std::string& name)
    : _initialized(false),
      _units(units),
      _name(name),
      _inputKernel(nullptr),
      _stateKernel(nullptr),
      _gatesBias(nullptr),
      _candidateiBias(nullptr),
      _candidatehBias(nullptr) {}

template <class T>
typename PyTorchGRUCell<T>::State PyTorchGRUCell<T>::zeroState(unsigned batch) {
  State state({batch, _units});
  data::zeros(state);

  return state;
}

template <class T>
std::pair<Variable<T>, typename PyTorchGRUCell<T>::State> PyTorchGRUCell<T>::operator()(
    const Tensor<T>& input, const State& state, size_t step, const std::vector<unsigned>& lengths) {
  if (input.shape().size() != 2) {
    throw InvalidArgumentError("invalid inputs");
  }
  if (input.shape(0) != state.shape(0)) {
    throw InvalidArgumentError("state and inputs don't match");
  }
  if (lengths.size() != 0 && lengths.size() != input.shape(0)) {
    throw InvalidArgumentError("invalid lengths");
  }

  auto scopeG = Scope::scope(_name);

  if (!_initialized) {
    _gatesBias = &Constant<T>::getConstant("bias_gates");
    _candidateiBias = &Constant<T>::getConstant("bias_candidate_i");
    _candidatehBias = &Constant<T>::getConstant("bias_candidate_h");

    _inputKernel = Constant<T>::tryGetRuntimeConstant("input_kernel");
    _stateKernel = Constant<T>::tryGetRuntimeConstant("state_kernel");
    if (_inputKernel == nullptr || _stateKernel == nullptr) {
      Constant<T>& gatesKernel = Constant<T>::getConstant("weight_gates");
      Constant<T>& candidateiKernel = Constant<T>::getConstant("weight_candidate_i");
      Constant<T>& candidatehKernel = Constant<T>::getConstant("weight_candidate_h");

      View<T> g1(gatesKernel, gatesKernel.shape(), 0, candidateiKernel.shape(0));
      Variable<T> iKernel = base::concat({&g1, &candidateiKernel}, 1);
      Constant<T>::setRuntimeConstantIfNotExist(
          "input_kernel", iKernel.shape(), data::get(iKernel));
      _inputKernel = Constant<T>::tryGetRuntimeConstant("input_kernel");

      View<T> g2(gatesKernel, gatesKernel.shape(), candidateiKernel.shape(0));
      Variable<T> hKernel = base::concat({&g2, &candidatehKernel}, 1);
      Constant<T>::setRuntimeConstantIfNotExist(
          "state_kernel", hKernel.shape(), data::get(hKernel));
      _stateKernel = Constant<T>::tryGetRuntimeConstant("state_kernel");
    }

    _initialized = true;
  }

  Variable<T> inputValue = blas::matmul(input, *_inputKernel);
  Variable<T> stateValue = blas::matmul(state, *_stateKernel);

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  Variable<T> hy = interface.getCellInterface().processPyTorchGRUCell(
      state, inputValue, stateValue, *_gatesBias, *_candidateiBias, *_candidatehBias);

  if (lengths.size() > 0) {
    for (unsigned i = 0; i < lengths.size(); ++i) {
      if (step < lengths[i]) {
        continue;
      }
      View<T> vcs(state, state.shape(), i, i + 1);
      base::assign_to(vcs, 0, i, hy);
    }
  }

  Variable<T> output = data::copy(hy);
  return std::make_pair(std::move(output), std::move(hy));
}

#define DEFINE_FUNC(type) template class PyTorchGRUCell<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace rnn
}  // namespace tfcc
