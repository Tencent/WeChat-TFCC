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

#include "graph.h"

#include <type_traits>

#include "tfcc.h"

#include "tfcc_runtime/exceptions/runtimeerror.h"
#include "tfcc_runtime/framework/types.h"
#include "tfcc_runtime/operations/operation.h"

namespace tfcc {
namespace runtime {

thread_local bool Graph::_recordDetailErrorEnable = false;

Graph::Graph(Model& model, const tfcc::runtime::model::Graph& graphProto)
    : _model(model),
      _graphProto(graphProto),
      _jit(4096 * 1024, tfcc::Xbyak::AutoGrow),
      _func(nullptr) {
  static_assert(RESERVED_STACK_SIZE % 16 == 0, "Invalid reserved stack size");

  auto scopeG = tfcc::Scope::scope(_graphProto.name());
  _symbolManager = std::unique_ptr<SymbolManager>(
      new SymbolManager(_graphProto.symbols().begin(), _graphProto.symbols().end()));
}

Graph::~Graph() {}

std::string Graph::getGraphName() const { return _graphProto.name(); }

std::vector<std::string> Graph::getInputs() const {
  return std::vector<std::string>(_graphProto.inputs().begin(), _graphProto.inputs().end());
}

std::vector<std::string> Graph::getOutputs() const {
  return std::vector<std::string>(_graphProto.outputs().begin(), _graphProto.outputs().end());
}

void Graph::process() {
  const char* err = _func();
  if (err) {
    throw RuntimeError(err);
  }
}

template <class T>
void Graph::setInput(const std::string& name, const tfcc::Tensor<T>& tensor) {
  tfcc::View<T>* inp = _symbolManager->getView(name, T());
  *inp = tfcc::View<T>(tensor);
}

template <class T>
void Graph::setInput(const std::string& name, tfcc::Variable<T>&& variable) {
  _symbolManager->setView(name, std::move(variable));
}

template <class T>
typename std::enable_if<std::is_arithmetic<T>::value, void>::type Graph::setInput(
    const std::string& name, T value) {
  T* inp = _symbolManager->getValue(name, T());
  *inp = value;
}

template <class T>
const tfcc::Tensor<T>& Graph::getTensor(const std::string& name, T) {
  return *_symbolManager->getTensor(name, T());
}

template <class T>
T Graph::getValue(const std::string& name, T) {
  return *_symbolManager->getValue(name, T());
}

template <class T>
const std::vector<T>& Graph::getVector(const std::string& name, T) {
  return *_symbolManager->getVector(name, T());
}

void Graph::setRecordDetailErrorThreadLocal(bool enable) { _recordDetailErrorEnable = enable; }

bool Graph::isRecordDetailError() { return _recordDetailErrorEnable; }

void Graph::buildGraph() {
  std::vector<std::unique_ptr<tfcc::Xbyak::Label>> errorLabels;
  if (_func) {
    return;
  }
  // jit start
  _jit.sub(_jit.rsp, 8 + RESERVED_STACK_SIZE);
  _jit.xor_(_jit.rax, _jit.rax);
  for (auto& node : _graphProto.nodes()) {
    _resources.emplace_back(node_to_asm(node, *this));
    _jit.test(_jit.rax, _jit.rax);
    std::unique_ptr<tfcc::Xbyak::Label> errorLabel(new tfcc::Xbyak::Label);
    _jit.jnz(*errorLabel, _jit.T_NEAR);
    errorLabels.push_back(std::move(errorLabel));
  }
  _jit.add(_jit.rsp, 8 + RESERVED_STACK_SIZE);
  _jit.ret();
  for (size_t i = 0; i < errorLabels.size(); ++i) {
    _jit.L(*errorLabels[i]);
    jit::call_function_with_rax_as_1st_string_param(_jit, getErrorString, this, i);
    _jit.add(_jit.rsp, 8 + RESERVED_STACK_SIZE);
    _jit.ret();
  }
  _jit.readyRE();
  _func = _jit.getCode<const char* (*)()>();
}

SymbolManager& Graph::getSymbolManager() { return *_symbolManager; }

Xbyak::CodeGenerator& Graph::getJIT() { return _jit; }

Model& Graph::getModel() { return _model; }

template <class T>
static inline std::string get_symbol_debug_string(
    SymbolManager& manager, const std::string& name, T) {
  auto& info = manager.getSymbolInfo(name);
  try {
    if (is_value(info.stype)) {
      return std::to_string(*manager.getValue(name, T()));
    }
    if (is_tensor(info.stype)) {
      return tfcc::debug_string(*manager.getTensor(name, T()));
    }
    if (is_vector(info.stype)) {
      std::string result;
      for (T v : *manager.getVector(name, T())) {
        result += std::to_string(v) + ", ";
      }
      return result;
    }
  } catch (std::exception& e) {
    return std::string("get symbol debug error: ") + e.what();
  }
  return "unknow stype";
}

static inline std::string get_symbol_debug_string(SymbolManager& manager, const std::string& name) {
  auto& info = manager.getSymbolInfo(name);
  switch (info.dtype) {
#define GET_ERROR_SYMBOL_DETAIL_FUNC(dtype)                                        \
  case (dtype):                                                                    \
    return get_symbol_debug_string(manager, name, DataTypeInfo<dtype>::CPPType()); \
    break;
    TFCC_RUNTIME_FOR_ALL_DATA_TYPE(GET_ERROR_SYMBOL_DETAIL_FUNC);
    default:
      return "invalid symbol type";
  }
}

const char* Graph::getErrorString(const char* str, Graph* graph, size_t index) {
  if (!_recordDetailErrorEnable) {
    return str;
  }
  try {
    thread_local std::string reason;
    reason = str;
    reason += "\n";

    reason += "error at node " + std::to_string(index) + "\n";
    reason += "node proto:\n";
    reason += graph->_graphProto.nodes(index).DebugString();
    reason += "\n====================\n";
    reason += "node inputs:\n";
    for (const std::string& name : graph->_graphProto.nodes(index).inputs()) {
      reason += "expect: ";
      for (auto& symbol : graph->_graphProto.symbols()) {
        if (symbol.name() != name) {
          continue;
        }
        reason += symbol.DebugString();
        break;
      }
      reason += "\n";
      reason += "real: ";
      reason += get_symbol_debug_string(*graph->_symbolManager, name);
#undef GET_ERROR_SYMBOL_DETAIL_FUNC
      reason += "\n--------------------\n";
    }

    return reason.c_str();
  } catch (std::exception& e) {
    thread_local std::string reason;
    reason = str;
    reason += "get error detail get a unexpected error: ";
    reason += e.what();
    return reason.c_str();
  }
}

#define DEFINE_FUNC(type)                                                                   \
  template void Graph::setInput(const std::string& name, const tfcc::Tensor<type>& tensor); \
  template void Graph::setInput(const std::string& name, tfcc::Variable<type>&& tensor);    \
  template void Graph::setInput(const std::string& name, type tensor);                      \
  template const tfcc::Tensor<type>& Graph::getTensor(const std::string& name, type);       \
  template type Graph::getValue(const std::string& name, type);                             \
  template const std::vector<type>& Graph::getVector(const std::string& name, type);

TFCC_RUNTIME_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace runtime
}  // namespace tfcc
