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

#include "loop.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/framework/model.h"
#include "tfcc_runtime/proto/operations/base.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace base {

template <class T>
static const char* _wrapper_cast_max_loop_to_u64(const T* maxLoop, uint64_t* u64MaxLoop) {
  if (*maxLoop <= 0) {
    return "Invalid loop";
  }
  *u64MaxLoop = static_cast<uint64_t>(*maxLoop);
  return nullptr;
}

template <class T>
static void _wrapper_cast_iterator(T* dst, const uint64_t* it) {
  *dst = static_cast<T>(*it);
}

static const char* _wrapper_graph_process(Graph* graph) {
  try {
    graph->process();
  } catch (const std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <class T>
static const char* _wrapper_append_value(const T* data, std::vector<ScanValue<T>>* values) {
  values->emplace_back();
  values->rbegin()->value = *data;
  return nullptr;
}

template <class T>
static const char* _wrapper_append_vector(
    const std::vector<T>* data, std::vector<ScanValue<T>>* values) {
  values->emplace_back();
  values->rbegin()->vector = *data;
  return nullptr;
}

template <class T>
static const char* _wrapper_append_tensor(
    const tfcc::Tensor<T>* data, std::vector<ScanValue<T>>* values) {
  try {
    values->emplace_back();
    values->rbegin()->variable = tfcc::data::copy(*data);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <class T>
static const char* _wrapper_append_variable(
    tfcc::Variable<T>* data, std::vector<ScanValue<T>>* values) {
  try {
    values->emplace_back();
    values->rbegin()->variable = std::move(*data);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <class T>
static const char* _wrapper_concat_value(
    std::vector<T>* dst, const std::vector<ScanValue<T>>* src) {
  dst->clear();
  for (const ScanValue<T>& value : *src) {
    dst->push_back(value.value);
  }
  return nullptr;
}

template <class T>
static const char* _wrapper_concat_vector(
    tfcc::Variable<T>* dst, const std::vector<ScanValue<T>>* src) {
  std::vector<T> result;
  for (const ScanValue<T>& value : *src) {
    result.insert(result.end(), value.vector.begin(), value.vector.end());
  }
  try {
    *dst = tfcc::data::set(
        result,
        {static_cast<unsigned>(src->size()), static_cast<unsigned>(result.size() / src->size())});
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <class T>
static const char* _wrapper_concat_tensor(
    tfcc::Variable<T>* dst, const std::vector<ScanValue<T>>* src) {
  std::vector<const tfcc::Tensor<T>*> values;
  for (const ScanValue<T>& value : *src) {
    values.push_back(&value.variable);
  }
  try {
    *dst = tfcc::base::stack(values, 0);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

static const char* _wrapper_clear_collection(ScanCollection* collection) {
  collection->floatValues.clear();
  collection->doubleValues.clear();
  collection->uint8Values.clear();
  collection->int8Values.clear();
  collection->uint16Values.clear();
  collection->int16Values.clear();
  collection->uint32Values.clear();
  collection->int32Values.clear();
  collection->uint64Values.clear();
  collection->int64Values.clear();
  collection->boolValues.clear();
  return nullptr;
}

std::string Loop::getOperationTypeUrl() const {
  tfcc::runtime::operations::base::Loop operation;
  return get_protobuf_type_url(operation);
}

std::set<unsigned> Loop::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::base::Loop::VERSION_1};
  return versions;
}

std::unique_ptr<OperationResource> Loop::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (!node.operation().Is<tfcc::runtime::operations::base::Loop>()) {
    return nullptr;
  }

  tfcc::runtime::operations::base::Loop operation;
  node.operation().UnpackTo(&operation);

  if (node.inputs_size() <
      static_cast<int64_t>(1 + operation.carried_count() + operation.capture_count())) {
    return nullptr;
  }
  if (node.inputs_size() >
      static_cast<int64_t>(2 + operation.carried_count() + operation.capture_count())) {
    return nullptr;
  }
  if (node.outputs_size() !=
      static_cast<int64_t>(operation.carried_count() + operation.scan_count())) {
    return nullptr;
  }

  bool hasMaxLoop = false;
  if (node.inputs_size() ==
      static_cast<int64_t>(2 + operation.carried_count() + operation.capture_count())) {
    hasMaxLoop = true;
  }

  auto& symbolManager = getSymbolManager(graph);
  auto& jit = getJIT(graph);
  tfcc::Xbyak::Label endLabel;
  Graph& subGraph = getGraph(getModel(graph), operation.sub_graph_name());
  auto& subGraphSymbolManager = getSymbolManager(subGraph);
  std::vector<std::string> subGraphInputs = subGraph.getInputs();
  std::vector<std::string> subGraphOutputs = subGraph.getOutputs();

  // set all capture inputs
  for (unsigned i = 0; i < operation.capture_count(); ++i) {
    setGraphInput(
        jit, subGraph, subGraphInputs[i + 2 + operation.carried_count()], graph,
        node.inputs(i + 1 + operation.carried_count()));
    jit.test(jit.rax, jit.rax);
    jit.jnz(endLabel, jit.T_NEAR);
  }
  std::unique_ptr<LoopResource> resource(new LoopResource);
  resource->maxLoop = 0;
  resource->currentLoop = 0;
  resource->scans.resize(operation.scan_count());
  // clear resource
  jit.mov(jit.rax, reinterpret_cast<uintptr_t>(&resource->currentLoop));
  jit.mov(jit.qword[jit.rax], 0);
  for (unsigned i = 0; i < operation.scan_count(); ++i) {
    callFunction(jit, _wrapper_clear_collection, &resource->scans[i]);
    jit.test(jit.rax, jit.rax);
    jit.jnz(endLabel, jit.T_NEAR);
  }

  // check max loop large than zero
  if (hasMaxLoop) {
    std::string maxLoopName = node.inputs(node.inputs_size() - 1);
    const SymbolInfo& maxLoopInfo = symbolManager.getSymbolInfo(maxLoopName);
    switch (maxLoopInfo.dtype) {
#define VALUE_SWICH_FUNC(dtype)                                                          \
  case dtype: {                                                                          \
    DataTypeInfo<dtype>::CPPType* maxLoopSymbol =                                        \
        symbolManager.getValue(maxLoopName, DataTypeInfo<dtype>::CPPType());             \
    callFunction(                                                                        \
        jit, _wrapper_cast_max_loop_to_u64<DataTypeInfo<dtype>::CPPType>, maxLoopSymbol, \
        &resource->maxLoop);                                                             \
    break;                                                                               \
  }
      TFCC_RUNTIME_FOR_ALL_DATA_TYPE(VALUE_SWICH_FUNC);
#undef VALUE_SWICH_FUNC
      default:
        throw RuntimeError("Invalid dtype");
    }
    jit.test(jit.rax, jit.rax);
    jit.jnz(endLabel, jit.T_NEAR);
  }

  // check condition
  uint8_t* conditionSymbol = symbolManager.getValue(node.inputs(0), uint8_t());
  tfcc::Xbyak::Label checkConditionSuccessLabel;
  jit.mov(jit.rdx, reinterpret_cast<uintptr_t>(conditionSymbol));
  jit.movzx(jit.rdx, jit.byte[jit.rdx]);
  jit.test(jit.rdx, jit.rdx);
  jit.jnz(checkConditionSuccessLabel, jit.T_NEAR);
  const char* invalidConditionErrMsg = "Invalid condition";
  jit.mov(jit.rax, reinterpret_cast<uintptr_t>(invalidConditionErrMsg));
  jit.jmp(endLabel, jit.T_NEAR);
  jit.L(checkConditionSuccessLabel);

  // first loop start
  // set carried symbol
  for (unsigned i = 0; i < operation.carried_count(); ++i) {
    setGraphInput(jit, subGraph, subGraphInputs[i + 2], graph, node.inputs(i + 1));
    jit.test(jit.rax, jit.rax);
    jit.jnz(endLabel, jit.T_NEAR);
  }

  // set iterator and condition
  setIteratorSymbol(jit, subGraphSymbolManager, subGraphInputs[0], &resource->currentLoop);
  uint8_t* subGraphConditionSymbol = subGraphSymbolManager.getValue(subGraphInputs[1], uint8_t());
  jit.mov(jit.rax, reinterpret_cast<uintptr_t>(conditionSymbol));
  jit.mov(jit.rdx, reinterpret_cast<uintptr_t>(subGraphConditionSymbol));
  jit.movzx(jit.eax, jit.byte[jit.rax]);
  jit.mov(jit.byte[jit.rdx], jit.al);

  // process
  callFunction(jit, _wrapper_graph_process, &subGraph);
  jit.test(jit.rax, jit.rax);
  jit.jnz(endLabel, jit.T_NEAR);

  // set result carried symbol
  for (unsigned i = 0; i < operation.carried_count(); ++i) {
    getGraphOutput(jit, graph, node.outputs(i), subGraph, subGraphOutputs[i + 1]);
    jit.test(jit.rax, jit.rax);
    jit.jnz(endLabel, jit.T_NEAR);
  }

  // set result scan
  for (unsigned i = 0; i < operation.scan_count(); ++i) {
    appendScanValue(
        jit, subGraphSymbolManager, subGraphOutputs[i + 1 + operation.carried_count()],
        resource->scans[i]);
    jit.test(jit.rax, jit.rax);
    jit.jnz(endLabel, jit.T_NEAR);
  }

  tfcc::Xbyak::Label loopStartLabel;
  tfcc::Xbyak::Label loopEndLabel;
  // first loop end
  // other loop
  // check condition
  uint8_t* subGraphOutputConditionSymbol =
      subGraphSymbolManager.getValue(subGraphOutputs[0], uint8_t());
  jit.L(loopStartLabel);
  jit.mov(jit.rdx, reinterpret_cast<uintptr_t>(subGraphOutputConditionSymbol));
  jit.movzx(jit.rdx, jit.byte[jit.rdx]);
  jit.test(jit.rdx, jit.rdx);
  jit.jz(loopEndLabel, jit.T_NEAR);

  // add current loop
  jit.mov(jit.rax, reinterpret_cast<uintptr_t>(&resource->currentLoop));
  jit.mov(jit.rsi, jit.ptr[jit.rax]);
  jit.add(jit.rsi, 1);
  jit.mov(jit.ptr[jit.rax], jit.rsi);
  if (hasMaxLoop) {
    jit.mov(jit.rdi, reinterpret_cast<uintptr_t>(&resource->maxLoop));
    jit.mov(jit.rdi, jit.ptr[jit.rdi]);
    jit.cmp(jit.rsi, jit.rdi);
    jit.jae(loopEndLabel, jit.T_NEAR);
  }

  // set carried symbol
  for (unsigned i = 0; i < operation.carried_count(); ++i) {
    setGraphInput(jit, subGraph, subGraphInputs[i + 2], graph, node.outputs(i));
    jit.test(jit.rax, jit.rax);
    jit.jnz(endLabel, jit.T_NEAR);
  }

  // set iterator and condition
  setIteratorSymbol(jit, subGraphSymbolManager, subGraphInputs[0], &resource->currentLoop);
  jit.mov(jit.rax, reinterpret_cast<uintptr_t>(subGraphConditionSymbol));
  jit.mov(jit.byte[jit.rax], jit.dl);

  // process
  callFunction(jit, _wrapper_graph_process, &subGraph);
  jit.test(jit.rax, jit.rax);
  jit.jnz(endLabel, jit.T_NEAR);

  // set result carried symbol
  for (unsigned i = 0; i < operation.carried_count(); ++i) {
    getGraphOutput(jit, graph, node.outputs(i), subGraph, subGraphOutputs[i + 1]);
    jit.test(jit.rax, jit.rax);
    jit.jnz(endLabel, jit.T_NEAR);
  }

  // set result scan
  for (unsigned i = 0; i < operation.scan_count(); ++i) {
    appendScanValue(
        jit, subGraphSymbolManager, subGraphOutputs[i + 1 + operation.carried_count()],
        resource->scans[i]);
    jit.test(jit.rax, jit.rax);
    jit.jnz(endLabel, jit.T_NEAR);
  }

  // endloop
  jit.jmp(loopStartLabel, jit.T_NEAR);
  jit.L(loopEndLabel);

  // concat scans
  for (unsigned i = 0; i < operation.scan_count(); ++i) {
    SymbolType scanStype =
        subGraphSymbolManager.getSymbolInfo(subGraphOutputs[i + 1 + operation.carried_count()])
            .stype;
    getScanResult(
        jit, scanStype, symbolManager, node.outputs(i + operation.carried_count()),
        resource->scans[i]);
    jit.test(jit.rax, jit.rax);
    jit.jnz(endLabel, jit.T_NEAR);
  }
  jit.L(endLabel);

  return std::unique_ptr<OperationResource>(std::move(resource));
}

void Loop::setIteratorSymbol(
    tfcc::Xbyak::CodeGenerator& jit, SymbolManager& manager, std::string name,
    const uint64_t* it) const {
  const SymbolInfo& info = manager.getSymbolInfo(name);
  switch (info.dtype) {
#define VALUE_SWICH_FUNC(dtype)                                                                    \
  case dtype: {                                                                                    \
    DataTypeInfo<dtype>::CPPType* symbol = manager.getValue(name, DataTypeInfo<dtype>::CPPType()); \
    callFunction(jit, _wrapper_cast_iterator<DataTypeInfo<dtype>::CPPType>, symbol, it);           \
    break;                                                                                         \
  }
    TFCC_RUNTIME_FOR_ALL_DATA_TYPE(VALUE_SWICH_FUNC);
#undef VALUE_SWICH_FUNC
    default:
      throw RuntimeError("Invalid dtype");
  }
}

void Loop::appendScanValue(
    tfcc::Xbyak::CodeGenerator& jit, SymbolManager& manager, std::string name,
    ScanCollection& collection) const {
  const SymbolInfo& info = manager.getSymbolInfo(name);
  if (info.dtype == tfcc::runtime::common::FLOAT) {
    appendScanValueInner(jit, manager, name, collection.floatValues);
  } else if (info.dtype == tfcc::runtime::common::DOUBLE) {
    appendScanValueInner(jit, manager, name, collection.doubleValues);
  } else if (info.dtype == tfcc::runtime::common::UINT8) {
    appendScanValueInner(jit, manager, name, collection.uint8Values);
  } else if (info.dtype == tfcc::runtime::common::INT8) {
    appendScanValueInner(jit, manager, name, collection.int8Values);
  } else if (info.dtype == tfcc::runtime::common::UINT16) {
    appendScanValueInner(jit, manager, name, collection.uint16Values);
  } else if (info.dtype == tfcc::runtime::common::INT16) {
    appendScanValueInner(jit, manager, name, collection.int16Values);
  } else if (info.dtype == tfcc::runtime::common::UINT32) {
    appendScanValueInner(jit, manager, name, collection.uint32Values);
  } else if (info.dtype == tfcc::runtime::common::INT32) {
    appendScanValueInner(jit, manager, name, collection.int32Values);
  } else if (info.dtype == tfcc::runtime::common::UINT64) {
    appendScanValueInner(jit, manager, name, collection.uint64Values);
  } else if (info.dtype == tfcc::runtime::common::INT64) {
    appendScanValueInner(jit, manager, name, collection.int64Values);
  } else if (info.dtype == tfcc::runtime::common::BOOL) {
    appendScanValueInner(jit, manager, name, collection.boolValues);
  } else {
    throw RuntimeError("Invalid data type " + std::to_string(info.dtype));
  }
}

template <class T>
void Loop::appendScanValueInner(
    tfcc::Xbyak::CodeGenerator& jit, SymbolManager& manager, std::string name,
    std::vector<ScanValue<T>>& values) const {
  const SymbolInfo& info = manager.getSymbolInfo(name);
  if (is_value(info.stype)) {
    T* src = manager.getValue(name, T());
    callFunction(jit, _wrapper_append_value<T>, src, &values);
  } else if (is_vector(info.stype)) {
    std::vector<T>* src = manager.getVector(name, T());
    callFunction(jit, _wrapper_append_vector<T>, src, &values);
  } else if (info.stype == SymbolType::VARIABLE) {
    tfcc::Variable<T>* src = manager.getVariable(name, T());
    callFunction(jit, _wrapper_append_variable<T>, src, &values);
  } else if (is_tensor(info.stype)) {
    tfcc::Tensor<T>* src = manager.getTensor(name, T());
    callFunction(jit, _wrapper_append_tensor<T>, src, &values);
  } else {
    // stype error
    throw RuntimeError("Invalid stype");
  }
}

void Loop::getScanResult(
    tfcc::Xbyak::CodeGenerator& jit, SymbolType scanStype, SymbolManager& manager, std::string name,
    ScanCollection& collection) const {
  const SymbolInfo& info = manager.getSymbolInfo(name);
  if (info.dtype == tfcc::runtime::common::FLOAT) {
    getScanResultInner(jit, scanStype, manager, name, collection.floatValues);
  } else if (info.dtype == tfcc::runtime::common::DOUBLE) {
    getScanResultInner(jit, scanStype, manager, name, collection.doubleValues);
  } else if (info.dtype == tfcc::runtime::common::UINT8) {
    getScanResultInner(jit, scanStype, manager, name, collection.uint8Values);
  } else if (info.dtype == tfcc::runtime::common::INT8) {
    getScanResultInner(jit, scanStype, manager, name, collection.int8Values);
  } else if (info.dtype == tfcc::runtime::common::UINT16) {
    getScanResultInner(jit, scanStype, manager, name, collection.uint16Values);
  } else if (info.dtype == tfcc::runtime::common::INT16) {
    getScanResultInner(jit, scanStype, manager, name, collection.int16Values);
  } else if (info.dtype == tfcc::runtime::common::UINT32) {
    getScanResultInner(jit, scanStype, manager, name, collection.uint32Values);
  } else if (info.dtype == tfcc::runtime::common::INT32) {
    getScanResultInner(jit, scanStype, manager, name, collection.int32Values);
  } else if (info.dtype == tfcc::runtime::common::UINT64) {
    getScanResultInner(jit, scanStype, manager, name, collection.uint64Values);
  } else if (info.dtype == tfcc::runtime::common::INT64) {
    getScanResultInner(jit, scanStype, manager, name, collection.int64Values);
  } else if (info.dtype == tfcc::runtime::common::BOOL) {
    getScanResultInner(jit, scanStype, manager, name, collection.boolValues);
  } else {
    throw RuntimeError("Invalid data type " + std::to_string(info.dtype));
  }
}

template <class T>
void Loop::getScanResultInner(
    tfcc::Xbyak::CodeGenerator& jit, SymbolType scanStype, SymbolManager& manager, std::string name,
    std::vector<ScanValue<T>>& values) const {
  if (is_value(scanStype)) {
    std::vector<T>* dst = manager.getVector(name, T());
    callFunction(jit, _wrapper_concat_value<T>, dst, &values);
  } else if (is_vector(scanStype)) {
    tfcc::Variable<T>* dst = manager.getVariable(name, T());
    callFunction(jit, _wrapper_concat_vector<T>, dst, &values);
  } else if (is_tensor(scanStype)) {
    tfcc::Variable<T>* dst = manager.getVariable(name, T());
    callFunction(jit, _wrapper_concat_tensor<T>, dst, &values);
  } else {
    // stype error
    throw RuntimeError("Invalid stype");
  }
}

std::vector<std::unique_ptr<Operation>> get_loop_operations() {
  std::vector<std::unique_ptr<Operation>> operations;
  operations.emplace_back(std::unique_ptr<Operation>(new Loop()));
  return operations;
}

}  // namespace base
}  // namespace runtime
}  // namespace tfcc
