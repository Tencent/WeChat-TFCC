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

#include "operation.h"

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tfcc_runtime/exceptions/unknownodeerror.h"
#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/framework/model.h"
#include "tfcc_runtime/operations/base.h"
#include "tfcc_runtime/operations/fusion.h"
#include "tfcc_runtime/operations/math.h"
#include "tfcc_runtime/operations/nn.h"
#include "tfcc_runtime/operations/random.h"
#include "tfcc_runtime/operations/relation.h"
#include "tfcc_runtime/operations/rnn.h"
#include "tfcc_runtime/operations/signal.h"

namespace tfcc {
namespace runtime {

OperationResource::~OperationResource() {}

Operation::Operation() {}

Operation::~Operation() {}

std::unique_ptr<OperationResource> Operation::debug(
    const tfcc::runtime::model::Node& node, Graph& graph) {
  struct DebugResource : public OperationResource {
    std::vector<std::unique_ptr<std::string>> prefixs;
  };

  std::unique_ptr<DebugResource> resource = std::unique_ptr<DebugResource>(new DebugResource);

  auto& jit = getJIT(graph);
  for (std::string name : node.outputs()) {
    const SymbolInfo& info = graph.getSymbolManager().getSymbolInfo(name);
    std::unique_ptr<std::string> prefix = std::unique_ptr<std::string>(new std::string);
    *prefix = "DEBUG: [" + graph.getGraphName() + "] [" + node.ShortDebugString() + "] " + name;
    switch (info.dtype) {
#define TENSOR_OUTPUT_SWICH_FUNC(dtype)                                             \
  case dtype:                                                                       \
    printSymbol(jit, graph, name, prefix->c_str(), DataTypeInfo<dtype>::CPPType()); \
    break;
      TFCC_RUNTIME_FOR_ALL_DATA_TYPE(TENSOR_OUTPUT_SWICH_FUNC);
#undef TENSOR_OUTPUT_SWICH_FUNC
      default:
        throw RuntimeError("Invalid dtype");
    }
    resource->prefixs.emplace_back(std::move(prefix));
  }
  return std::unique_ptr<OperationResource>(std::move(resource));
}

SymbolManager& Operation::getSymbolManager(Graph& graph) { return graph.getSymbolManager(); }

tfcc::Xbyak::CodeGenerator& Operation::getJIT(Graph& graph) { return graph.getJIT(); }

Model& Operation::getModel(Graph& graph) { return graph.getModel(); }

Graph& Operation::getGraph(Model& model, const std::string& name) { return model.getGraph(name); }

void Operation::setGraphInput(
    Xbyak::CodeGenerator& jit, Graph& dstGraph, std::string dstName, Graph& srcGraph,
    std::string srcName) {
  const SymbolInfo& srcInfo = srcGraph.getSymbolManager().getSymbolInfo(srcName);
  switch (srcInfo.dtype) {
#define TENSOR_OUTPUT_SWICH_FUNC(dtype)                                                            \
  case dtype:                                                                                      \
    setGraphInputInner(jit, dstGraph, dstName, srcGraph, srcName, DataTypeInfo<dtype>::CPPType()); \
    break;
    TFCC_RUNTIME_FOR_ALL_DATA_TYPE(TENSOR_OUTPUT_SWICH_FUNC);
#undef TENSOR_OUTPUT_SWICH_FUNC
    default:
      throw RuntimeError("Invalid dtype");
  }
}

void Operation::getGraphOutput(
    Xbyak::CodeGenerator& jit, Graph& dstGraph, std::string dstName, Graph& srcGraph,
    std::string srcName) {
  const SymbolInfo& srcInfo = srcGraph.getSymbolManager().getSymbolInfo(srcName);
  switch (srcInfo.dtype) {
#define TENSOR_OUTPUT_SWICH_FUNC(dtype)                                             \
  case dtype:                                                                       \
    getGraphOutputInner(                                                            \
        jit, dstGraph, dstName, srcGraph, srcName, DataTypeInfo<dtype>::CPPType()); \
    break;
    TFCC_RUNTIME_FOR_ALL_DATA_TYPE(TENSOR_OUTPUT_SWICH_FUNC);
#undef TENSOR_OUTPUT_SWICH_FUNC
    default:
      throw RuntimeError("Invalid dtype");
  }
}

template <class T>
static const char* _wrapper_print_tensor(const tfcc::Tensor<T>* value, const char* prefix) {
  std::cout << prefix << " TENSOR " << *value << std::endl;
  return nullptr;
}

template <class T>
static const char* _wrapper_print_vector(const std::vector<T>* value, const char* prefix) {
  std::cout << prefix << " VECTOR "
            << tfcc::data::set(*value, {static_cast<unsigned>(value->size())}) << std::endl;
  return nullptr;
}

template <class T>
static const char* _wrapper_print_value(const T* value, const char* prefix) {
  std::cout << prefix << " VALUE " << *value << std::endl;
  return nullptr;
}

template <class T>
void Operation::printSymbol(
    Xbyak::CodeGenerator& jit, Graph& graph, std::string name, const char* prefix, T) {
  const SymbolInfo& info = graph.getSymbolManager().getSymbolInfo(name);
  if (is_value(info.stype)) {
    T* x = graph.getSymbolManager().getValue(name, T());
    callFunction(jit, _wrapper_print_value<T>, x, prefix);
  } else if (is_vector(info.stype)) {
    std::vector<T>* x = graph.getSymbolManager().getVector(name, T());
    callFunction(jit, _wrapper_print_vector<T>, x, prefix);
  } else if (is_tensor(info.stype)) {
    tfcc::Tensor<T>* x = graph.getSymbolManager().getTensor(name, T());
    callFunction(jit, _wrapper_print_tensor<T>, x, prefix);
  } else {
    // stype error
    throw RuntimeError("Invalid stype");
  }
}

template <class T>
static const char* _wrapper_copy_value(T* dst, const T* src) {
  *dst = *src;
  return nullptr;
}

template <class T>
static const char* _wrapper_copy_vector(std::vector<T>* dst, const std::vector<T>* src) {
  *dst = *src;
  return nullptr;
}

template <class T>
static const char* _wrapper_ref_tensor(tfcc::View<T>* dst, const tfcc::Tensor<T>* src) {
  try {
    *dst = tfcc::View<T>(*src);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <class T>
static const char* _wrapper_copy_tensor(tfcc::Variable<T>* dst, const tfcc::Tensor<T>* src) {
  try {
    *dst = tfcc::data::copy(*src);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <class T>
static const char* _wrapper_move_variable(tfcc::Variable<T>* dst, tfcc::Variable<T>* src) {
  try {
    *dst = std::move(*src);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <class T>
void Operation::setGraphInputInner(
    Xbyak::CodeGenerator& jit, Graph& dstGraph, std::string dstName, Graph& srcGraph,
    std::string srcName, T) {
  const SymbolInfo& srcInfo = srcGraph.getSymbolManager().getSymbolInfo(srcName);
  if (is_value(srcInfo.stype)) {
    T* src = srcGraph.getSymbolManager().getValue(srcName, T());
    T* dst = dstGraph.getSymbolManager().getValue(dstName, T());
    callFunction(jit, _wrapper_copy_value<T>, dst, src);
  } else if (is_vector(srcInfo.stype)) {
    std::vector<T>* src = srcGraph.getSymbolManager().getVector(srcName, T());
    std::vector<T>* dst = dstGraph.getSymbolManager().getVector(dstName, T());
    callFunction(jit, _wrapper_copy_vector<T>, dst, src);
  } else if (is_tensor(srcInfo.stype)) {
    tfcc::Tensor<T>* src = srcGraph.getSymbolManager().getTensor(srcName, T());
    tfcc::View<T>* dst = dstGraph.getSymbolManager().getView(dstName, T());
    callFunction(jit, _wrapper_ref_tensor<T>, dst, src);
  } else {
    // stype error
    throw RuntimeError("Invalid stype");
  }
}

template <class T>
void Operation::getGraphOutputInner(
    Xbyak::CodeGenerator& jit, Graph& dstGraph, std::string dstName, Graph& srcGraph,
    std::string srcName, T) {
  const SymbolInfo& srcInfo = srcGraph.getSymbolManager().getSymbolInfo(srcName);
  if (is_value(srcInfo.stype)) {
    T* src = srcGraph.getSymbolManager().getValue(srcName, T());
    T* dst = dstGraph.getSymbolManager().getValue(dstName, T());
    callFunction(jit, _wrapper_copy_value<T>, dst, src);
  } else if (is_vector(srcInfo.stype)) {
    std::vector<T>* src = srcGraph.getSymbolManager().getVector(srcName, T());
    std::vector<T>* dst = dstGraph.getSymbolManager().getVector(dstName, T());
    callFunction(jit, _wrapper_copy_vector<T>, dst, src);
  } else if (srcInfo.stype == SymbolType::VARIABLE) {
    tfcc::Variable<T>* src = srcGraph.getSymbolManager().getVariable(srcName, T());
    tfcc::Variable<T>* dst = dstGraph.getSymbolManager().getVariable(dstName, T());
    callFunction(jit, _wrapper_move_variable<T>, dst, src);
  } else if (is_tensor(srcInfo.stype)) {
    tfcc::Tensor<T>* src = srcGraph.getSymbolManager().getTensor(srcName, T());
    tfcc::Variable<T>* dst = dstGraph.getSymbolManager().getVariable(dstName, T());
    callFunction(jit, _wrapper_copy_tensor<T>, dst, src);
  } else {
    // stype error
    throw RuntimeError("Invalid stype");
  }
}

class OperationCollector {
  std::unordered_map<std::string, std::vector<std::unique_ptr<Operation>>> _operations;

 public:
  OperationCollector();

  const std::vector<std::unique_ptr<Operation>>& getOperations(const std::string& url) const;

 private:
  void appendOperations(std::vector<std::unique_ptr<Operation>> operations);
};

OperationCollector::OperationCollector() {
  appendOperations(base::get_all_operations());
  appendOperations(math::get_all_operations());
  appendOperations(nn::get_all_operations());
  appendOperations(random::get_all_operations());
  appendOperations(relation::get_all_operations());
  appendOperations(fusion::get_all_operations());
  appendOperations(rnn::get_all_operations());
  appendOperations(signal::get_all_operations());
}

const std::vector<std::unique_ptr<Operation>>& OperationCollector::getOperations(
    const std::string& url) const {
  static std::vector<std::unique_ptr<Operation>> empty;
  auto it = _operations.find(url);
  if (it == _operations.end()) {
    return empty;
  }
  return it->second;
}

void OperationCollector::appendOperations(std::vector<std::unique_ptr<Operation>> operations) {
  for (std::unique_ptr<Operation>& op : operations) {
    std::string url = op->getOperationTypeUrl();
    _operations[url].emplace_back(std::move(op));
  }
}

std::unique_ptr<OperationResource> node_to_asm(
    const tfcc::runtime::model::Node& node, Graph& graph) {
  thread_local OperationCollector collector;
  for (auto& op : collector.getOperations(node.operation().type_url())) {
    std::set<unsigned> versionSet = op->getOperationVersions();
    if (versionSet.find(node.version()) == versionSet.end()) {
      continue;
    }
    auto resource = op->process(node, graph);
    if (resource) {
      return resource;
    }
  }
  throw UnknowNodeError(node);
}

}  // namespace runtime
}  // namespace tfcc
