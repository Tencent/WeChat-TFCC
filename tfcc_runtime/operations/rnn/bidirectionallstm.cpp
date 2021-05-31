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

#include "bidirectionallstm.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/rnn.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace rnn {

template <class T>
static const char* _wrapper_bidirectional_lstm(
    const tfcc::Tensor<T>* a, const tfcc::Tensor<T>* inputKernel,
    const tfcc::Tensor<T>* stateKernel, const tfcc::Tensor<T>* bias, const tfcc::Tensor<T>* ih,
    const tfcc::Tensor<T>* ic, tfcc::Variable<T>* forwardOutput, tfcc::Variable<T>* backwardOutput,
    tfcc::Variable<T>* forwardH, tfcc::Variable<T>* backwardH, tfcc::Variable<T>* forwardC,
    tfcc::Variable<T>* backwardC) noexcept {
  try {
    tfcc::View<T> forwardInputKernel(*inputKernel, inputKernel->shape(), 0, 1);
    forwardInputKernel.reshape({inputKernel->shape(1), inputKernel->shape(2)});
    tfcc::View<T> backwardInputKernel(*inputKernel, inputKernel->shape(), 1, 2);
    backwardInputKernel.reshape({inputKernel->shape(1), inputKernel->shape(2)});
    tfcc::View<T> forwardStateKernel(*stateKernel, stateKernel->shape(), 0, 1);
    forwardStateKernel.reshape({stateKernel->shape(1), stateKernel->shape(2)});
    tfcc::View<T> backwardStateKernel(*stateKernel, stateKernel->shape(), 1, 2);
    backwardStateKernel.reshape({stateKernel->shape(1), stateKernel->shape(2)});
    tfcc::View<T> forwardBias(*bias, bias->shape(), 0, 1);
    forwardBias.reshape({bias->shape(1)});
    tfcc::View<T> backwardBias(*bias, bias->shape(), 1, 2);
    backwardBias.reshape({bias->shape(1)});
    tfcc::View<T> forwardIH(*ih, ih->shape(), 0, 1);
    forwardIH.reshape({ih->shape(1), ih->shape(2)});
    tfcc::View<T> backwardIH(*ih, ih->shape(), 1, 2);
    backwardIH.reshape({ih->shape(1), ih->shape(2)});
    tfcc::View<T> forwardIC(*ic, ic->shape(), 0, 1);
    forwardIC.reshape({ic->shape(1), ic->shape(2)});
    tfcc::View<T> backwardIC(*ic, ic->shape(), 1, 2);
    backwardIC.reshape({ic->shape(1), ic->shape(2)});

    std::tie(*forwardOutput, *forwardH, *forwardC) = tfcc::rnn::lstm_forward(
        *a, forwardInputKernel, forwardStateKernel, forwardBias, forwardIH, forwardIC);
    std::tie(*backwardOutput, *backwardH, *backwardC) = tfcc::rnn::lstm_backward(
        *a, backwardInputKernel, backwardStateKernel, backwardBias, backwardIH, backwardIC);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE>
std::string BidirectionalLSTM<DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::rnn::BidirectionalLSTM operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE>
std::set<unsigned> BidirectionalLSTM<DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::rnn::BidirectionalLSTM::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE>
std::unique_ptr<OperationResource> BidirectionalLSTM<DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 6 || node.outputs_size() != 6) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::rnn::BidirectionalLSTM>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  if (symbolManager.getSymbolInfo(node.outputs(0)).dtype != DTYPE) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource);
  tfcc::runtime::operations::rnn::BidirectionalLSTM operation;
  node.operation().UnpackTo(&operation);

  if (operation.hidden_size() == 0) {
    return nullptr;
  }

  auto& jit = getJIT(graph);

  const tfcc::Tensor<T>* inputSymbol = symbolManager.getTensor(node.inputs(0), T());
  const tfcc::Tensor<T>* inputKernelSymbol = symbolManager.getTensor(node.inputs(1), T());
  const tfcc::Tensor<T>* stateKernelSymbol = symbolManager.getTensor(node.inputs(2), T());
  const tfcc::Tensor<T>* biasSymbol = symbolManager.getTensor(node.inputs(3), T());
  const tfcc::Tensor<T>* initialHSymbol = symbolManager.getTensor(node.inputs(4), T());
  const tfcc::Tensor<T>* initialCSymbol = symbolManager.getTensor(node.inputs(5), T());
  tfcc::Variable<T>* forwardOutputSymbol = symbolManager.getVariable(node.outputs(0), T());
  tfcc::Variable<T>* backwardOutputSymbol = symbolManager.getVariable(node.outputs(1), T());
  tfcc::Variable<T>* forwardHSymbol = symbolManager.getVariable(node.outputs(2), T());
  tfcc::Variable<T>* backwardHSymbol = symbolManager.getVariable(node.outputs(3), T());
  tfcc::Variable<T>* forwardCSymbol = symbolManager.getVariable(node.outputs(4), T());
  tfcc::Variable<T>* backwardCSymbol = symbolManager.getVariable(node.outputs(5), T());

  callFunction(
      jit, _wrapper_bidirectional_lstm<T>, inputSymbol, inputKernelSymbol, stateKernelSymbol,
      biasSymbol, initialHSymbol, initialCSymbol, forwardOutputSymbol, backwardOutputSymbol,
      forwardHSymbol, backwardHSymbol, forwardCSymbol, backwardCSymbol);
  return resource;
}

std::vector<std::unique_ptr<Operation>> get_bidirectional_lstm_operations() {
#define DEFINE_FUNC(dtype) \
  operations.emplace_back(std::unique_ptr<Operation>(new BidirectionalLSTM<dtype>()));
  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace rnn
}  // namespace runtime
}  // namespace tfcc
