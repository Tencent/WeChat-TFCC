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

#include "if_.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/framework/model.h"
#include "tfcc_runtime/proto/operations/base.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace base {

static const char* _wrapper_graph_process(Graph* graph) {
  try {
    graph->process();
  } catch (const std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

std::string If::getOperationTypeUrl() const {
  tfcc::runtime::operations::base::If operation;
  return get_protobuf_type_url(operation);
}

std::set<unsigned> If::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::base::If::VERSION_1};
  return versions;
}

std::unique_ptr<OperationResource> If::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (!node.operation().Is<tfcc::runtime::operations::base::If>()) {
    return nullptr;
  }

  tfcc::runtime::operations::base::If operation;
  node.operation().UnpackTo(&operation);

  unsigned expectInputCount =
      1 + operation.then_graph_capture_count() + operation.else_graph_capture_count();
  if (node.inputs_size() != static_cast<int64_t>(expectInputCount)) {
    return nullptr;
  }

  auto& symbolManager = getSymbolManager(graph);
  Graph& thenGraph = getGraph(getModel(graph), operation.then_graph_name());
  Graph& elseGraph = getGraph(getModel(graph), operation.else_graph_name());
  std::vector<std::string> thenGraphInputs = thenGraph.getInputs();
  std::vector<std::string> elseGraphInputs = elseGraph.getInputs();
  std::vector<std::string> thenGraphOutputs = thenGraph.getOutputs();
  std::vector<std::string> elseGraphOutputs = elseGraph.getOutputs();

  if (thenGraphOutputs.size() != elseGraphOutputs.size()) {
    return nullptr;
  }
  if (node.outputs_size() != static_cast<int64_t>(thenGraphOutputs.size())) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource);

  auto& jit = getJIT(graph);
  tfcc::Xbyak::Label elseLabel;
  tfcc::Xbyak::Label endLabel;
  uint8_t* conditionSymbol = symbolManager.getValue(node.inputs(0), uint8_t());
  jit.mov(jit.rax, reinterpret_cast<uintptr_t>(conditionSymbol));
  jit.movzx(jit.rax, jit.ptr[jit.rax]);
  jit.test(jit.rax, jit.rax);
  jit.jz(elseLabel);
  // set then capture
  for (unsigned i = 0; i < operation.then_graph_capture_count(); ++i) {
    setGraphInput(jit, thenGraph, thenGraphInputs[i], graph, node.inputs(i + 1));
    jit.test(jit.rax, jit.rax);
    jit.jnz(endLabel, jit.T_NEAR);
  }

  // process then graph
  callFunction(jit, _wrapper_graph_process, &thenGraph);
  jit.test(jit.rax, jit.rax);
  jit.jnz(endLabel, jit.T_NEAR);

  // get outputs
  for (unsigned i = 0; i < thenGraphOutputs.size(); ++i) {
    getGraphOutput(jit, graph, node.outputs(i), thenGraph, thenGraphOutputs[i]);
    jit.test(jit.rax, jit.rax);
    jit.jnz(endLabel, jit.T_NEAR);
  }
  jit.jmp(endLabel, jit.T_NEAR);

  jit.L(elseLabel);
  // set else capture
  for (unsigned i = 0; i < operation.else_graph_capture_count(); ++i) {
    setGraphInput(
        jit, elseGraph, elseGraphInputs[i], graph,
        node.inputs(i + 1 + operation.then_graph_capture_count()));
    jit.test(jit.rax, jit.rax);
    jit.jnz(endLabel, jit.T_NEAR);
  }

  // process else graph
  callFunction(jit, _wrapper_graph_process, &elseGraph);
  jit.test(jit.rax, jit.rax);
  jit.jnz(endLabel, jit.T_NEAR);

  // get outputs
  for (unsigned i = 0; i < elseGraphOutputs.size(); ++i) {
    getGraphOutput(jit, graph, node.outputs(i), elseGraph, elseGraphOutputs[i]);
    jit.test(jit.rax, jit.rax);
    jit.jnz(endLabel, jit.T_NEAR);
  }
  jit.L(endLabel);
  return resource;
}

std::vector<std::unique_ptr<Operation>> get_if_operations() {
  std::vector<std::unique_ptr<Operation>> operations;
  operations.emplace_back(std::unique_ptr<Operation>(new If()));
  return operations;
}

}  // namespace base
}  // namespace runtime
}  // namespace tfcc
