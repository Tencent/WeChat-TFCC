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

#include "tfcc_grucell.h"

#include "exceptions/tfcc_invalidargumenterror.h"
#include "framework/tfcc_scope.h"
#include "framework/tfcc_types.h"
#include "framework/tfcc_view.h"
#include "interfaces/tfcc_cellinterface.h"
#include "layers/tfcc_denselayer.h"
#include "operations/tfcc_base.h"
#include "operations/tfcc_blas.h"
#include "operations/tfcc_data.h"
#include "operations/tfcc_math.h"
#include "operations/tfcc_operation.h"
#include "operations/tfcc_operator.h"

namespace tfcc {
namespace rnn {

template <class T>
GRUCell<T>::GRUCell(unsigned units) : GRUCell(units, "gru_cell") {}

template <class T>
GRUCell<T>::GRUCell(unsigned units, const std::string& name)
    : _initialized(false),
      _units(units),
      _name(name),
      _gatesKernel(nullptr),
      _gatesBias(nullptr),
      _candidateKernel(nullptr),
      _candidateBias(nullptr) {}

template <class T>
typename GRUCell<T>::State GRUCell<T>::zeroState(unsigned batch) {
  State state({batch, _units});
  data::zeros(state);

  return state;
}

template <class T>
std::pair<Variable<T>, typename GRUCell<T>::State> GRUCell<T>::operator()(
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
    {
      auto scopeG = Scope::scope("gates");
      _gatesKernel = &Constant<T>::getConstant("kernel");
      _gatesBias = &Constant<T>::getConstant("bias");
    }

    {
      auto scopeC = Scope::scope("candidate");
      _candidateKernel = &Constant<T>::getConstant("kernel");
      _candidateBias = &Constant<T>::getConstant("bias");
    }

    _initialized = true;
  }

  Variable<T> gi = base::concat(std::initializer_list<const Tensor<T>*>{&input, &state}, 1);
  Variable<T> value = blas::matmul(gi, *_gatesKernel);
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  Variable<T> ci =
      interface.getCellInterface().processGRUCellGates(state, input, value, *_gatesBias);

  State h = interface.getCellInterface().processGRUCellCandidate(
      state, value, *_gatesBias, blas::matmul(ci, *_candidateKernel), *_candidateBias);

  if (lengths.size() > 0) {
    for (unsigned i = 0; i < lengths.size(); ++i) {
      if (step < lengths[i]) {
        continue;
      }
      View<T> vcs(state, state.shape(), i, i + 1);
      base::assign_to(vcs, 0, i, h);
    }
  }

  Variable<T> output = data::copy(h);
  return std::make_pair(std::move(output), std::move(h));
}

#define DEFINE_FUNC(type) template class GRUCell<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace rnn
}  // namespace tfcc
