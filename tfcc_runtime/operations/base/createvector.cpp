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

#include "createvector.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/base.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace base {

template <class T>
void _wrapper_clear_vector(std::vector<T>* result) noexcept {
  result->clear();
}

template <class T>
void _wrapper_append_value_v1(T a, std::vector<T>* result) noexcept {
  result->push_back(a);
}

template <class T1, class T2>
void _wrapper_append_value_v2(T1* a, std::vector<T2>* result) noexcept {
  result->push_back(static_cast<T2>(*a));
}

template <class T1, class T2>
void _wrapper_append_vector(std::vector<T1>* vec, std::vector<T2>* result) noexcept {
  for (T1 v : *vec) {
    result->push_back(static_cast<T2>(v));
  }
}

template <tfcc::runtime::common::DataType DTYPE>
std::string CreateVector<DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::base::CreateVector operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE>
std::set<unsigned> CreateVector<DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::base::CreateVector::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE>
std::unique_ptr<OperationResource> CreateVector<DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::base::CreateVector>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  if (symbolManager.getSymbolInfo(node.outputs(0)).dtype != DTYPE) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource);
  tfcc::runtime::operations::base::CreateVector operation;
  node.operation().UnpackTo(&operation);
  if (operation.data_type() != DTYPE) {
    return nullptr;
  }

  auto& jit = getJIT(graph);

  std::vector<T>* outputSymbol = symbolManager.getVector(node.outputs(0), T());
  callFunction(jit, _wrapper_clear_vector<T>, outputSymbol);
  for (auto& value : operation.values()) {
    if (value.source_case() ==
        tfcc::runtime::operations::base::CreateVector_Value::SourceCase::kPos) {
      if (static_cast<int>(value.pos()) >= node.inputs_size()) {
        return nullptr;
      }
      processPosition(node.inputs(value.pos()), node.outputs(0), graph);
      continue;
    }

    T v;
    if (value.source_case() ==
        tfcc::runtime::operations::base::CreateVector_Value::SourceCase::kInt64Value) {
      v = static_cast<T>(value.int64_value());
    } else if (
        value.source_case() ==
        tfcc::runtime::operations::base::CreateVector_Value::SourceCase::kUint64Value) {
      v = static_cast<T>(value.uint64_value());
    } else if (
        value.source_case() ==
        tfcc::runtime::operations::base::CreateVector_Value::SourceCase::kFloatValue) {
      v = static_cast<T>(value.float_value());
    } else if (
        value.source_case() ==
        tfcc::runtime::operations::base::CreateVector_Value::SourceCase::kDoubleValue) {
      v = static_cast<T>(value.double_value());
    } else {
      return nullptr;
    }
    callFunction(jit, _wrapper_append_value_v1<T>, v, outputSymbol);
  }
  jit.xor_(jit.rax, jit.rax);

  return resource;
}

template <tfcc::runtime::common::DataType DTYPE>
void CreateVector<DTYPE>::processPosition(
    const std::string& input, const std::string& output, Graph& graph) const {
  auto& symbolManager = getSymbolManager(graph);
  std::vector<T>* outputSymbol = symbolManager.getVector(output, T());
  const SymbolInfo& inputSymbolInfo = symbolManager.getSymbolInfo(input);

  auto& jit = getJIT(graph);
  if (inputSymbolInfo.dtype == tfcc::runtime::common::FLOAT) {
    using T2 = typename DataTypeInfo<tfcc::runtime::common::FLOAT>::CPPType;
    if (is_vector(inputSymbolInfo.stype)) {
      std::vector<T2>* inputSymbol = symbolManager.getVector(input, T2());
      callFunction(jit, _wrapper_append_vector<T2, T>, inputSymbol, outputSymbol);
    } else if (is_value(inputSymbolInfo.stype)) {
      T2* inputSymbol = symbolManager.getValue(input, T2());
      callFunction(jit, _wrapper_append_value_v2<T2, T>, inputSymbol, outputSymbol);
    } else {
      // unknow stype
      throw RuntimeError(
          "Invalid symbol type" + std::to_string(static_cast<int>(inputSymbolInfo.stype)));
    }
  } else if (inputSymbolInfo.dtype == tfcc::runtime::common::DOUBLE) {
    using T2 = typename DataTypeInfo<tfcc::runtime::common::DOUBLE>::CPPType;
    if (is_vector(inputSymbolInfo.stype)) {
      std::vector<T2>* inputSymbol = symbolManager.getVector(input, T2());
      callFunction(jit, _wrapper_append_vector<T2, T>, inputSymbol, outputSymbol);
    } else if (is_value(inputSymbolInfo.stype)) {
      T2* inputSymbol = symbolManager.getValue(input, T2());
      callFunction(jit, _wrapper_append_value_v2<T2, T>, inputSymbol, outputSymbol);
    } else {
      // unknow stype
      throw RuntimeError(
          "Invalid symbol type" + std::to_string(static_cast<int>(inputSymbolInfo.stype)));
    }
  } else if (inputSymbolInfo.dtype == tfcc::runtime::common::UINT8) {
    using T2 = typename DataTypeInfo<tfcc::runtime::common::UINT8>::CPPType;
    if (is_vector(inputSymbolInfo.stype)) {
      std::vector<T2>* inputSymbol = symbolManager.getVector(input, T2());
      callFunction(jit, _wrapper_append_vector<T2, T>, inputSymbol, outputSymbol);
    } else if (is_value(inputSymbolInfo.stype)) {
      T2* inputSymbol = symbolManager.getValue(input, T2());
      callFunction(jit, _wrapper_append_value_v2<T2, T>, inputSymbol, outputSymbol);
    } else {
      // unknow stype
      throw RuntimeError(
          "Invalid symbol type" + std::to_string(static_cast<int>(inputSymbolInfo.stype)));
    }
  } else if (inputSymbolInfo.dtype == tfcc::runtime::common::INT8) {
    using T2 = typename DataTypeInfo<tfcc::runtime::common::INT8>::CPPType;
    if (is_vector(inputSymbolInfo.stype)) {
      std::vector<T2>* inputSymbol = symbolManager.getVector(input, T2());
      callFunction(jit, _wrapper_append_vector<T2, T>, inputSymbol, outputSymbol);
    } else if (is_value(inputSymbolInfo.stype)) {
      T2* inputSymbol = symbolManager.getValue(input, T2());
      callFunction(jit, _wrapper_append_value_v2<T2, T>, inputSymbol, outputSymbol);
    } else {
      // unknow stype
      throw RuntimeError(
          "Invalid symbol type" + std::to_string(static_cast<int>(inputSymbolInfo.stype)));
    }
  } else if (inputSymbolInfo.dtype == tfcc::runtime::common::UINT16) {
    using T2 = typename DataTypeInfo<tfcc::runtime::common::UINT16>::CPPType;
    if (is_vector(inputSymbolInfo.stype)) {
      std::vector<T2>* inputSymbol = symbolManager.getVector(input, T2());
      callFunction(jit, _wrapper_append_vector<T2, T>, inputSymbol, outputSymbol);
    } else if (is_value(inputSymbolInfo.stype)) {
      T2* inputSymbol = symbolManager.getValue(input, T2());
      callFunction(jit, _wrapper_append_value_v2<T2, T>, inputSymbol, outputSymbol);
    } else {
      // unknow stype
      throw RuntimeError(
          "Invalid symbol type" + std::to_string(static_cast<int>(inputSymbolInfo.stype)));
    }
  } else if (inputSymbolInfo.dtype == tfcc::runtime::common::INT16) {
    using T2 = typename DataTypeInfo<tfcc::runtime::common::INT16>::CPPType;
    if (is_vector(inputSymbolInfo.stype)) {
      std::vector<T2>* inputSymbol = symbolManager.getVector(input, T2());
      callFunction(jit, _wrapper_append_vector<T2, T>, inputSymbol, outputSymbol);
    } else if (is_value(inputSymbolInfo.stype)) {
      T2* inputSymbol = symbolManager.getValue(input, T2());
      callFunction(jit, _wrapper_append_value_v2<T2, T>, inputSymbol, outputSymbol);
    } else {
      // unknow stype
      throw RuntimeError(
          "Invalid symbol type" + std::to_string(static_cast<int>(inputSymbolInfo.stype)));
    }
  } else if (inputSymbolInfo.dtype == tfcc::runtime::common::UINT32) {
    using T2 = typename DataTypeInfo<tfcc::runtime::common::UINT32>::CPPType;
    if (is_vector(inputSymbolInfo.stype)) {
      std::vector<T2>* inputSymbol = symbolManager.getVector(input, T2());
      callFunction(jit, _wrapper_append_vector<T2, T>, inputSymbol, outputSymbol);
    } else if (is_value(inputSymbolInfo.stype)) {
      T2* inputSymbol = symbolManager.getValue(input, T2());
      callFunction(jit, _wrapper_append_value_v2<T2, T>, inputSymbol, outputSymbol);
    } else {
      // unknow stype
      throw RuntimeError(
          "Invalid symbol type" + std::to_string(static_cast<int>(inputSymbolInfo.stype)));
    }
  } else if (inputSymbolInfo.dtype == tfcc::runtime::common::INT32) {
    using T2 = typename DataTypeInfo<tfcc::runtime::common::INT32>::CPPType;
    if (is_vector(inputSymbolInfo.stype)) {
      std::vector<T2>* inputSymbol = symbolManager.getVector(input, T2());
      callFunction(jit, _wrapper_append_vector<T2, T>, inputSymbol, outputSymbol);
    } else if (is_value(inputSymbolInfo.stype)) {
      T2* inputSymbol = symbolManager.getValue(input, T2());
      callFunction(jit, _wrapper_append_value_v2<T2, T>, inputSymbol, outputSymbol);
    } else {
      // unknow stype
      throw RuntimeError(
          "Invalid symbol type" + std::to_string(static_cast<int>(inputSymbolInfo.stype)));
    }
  } else if (inputSymbolInfo.dtype == tfcc::runtime::common::UINT64) {
    using T2 = typename DataTypeInfo<tfcc::runtime::common::UINT64>::CPPType;
    if (is_vector(inputSymbolInfo.stype)) {
      std::vector<T2>* inputSymbol = symbolManager.getVector(input, T2());
      callFunction(jit, _wrapper_append_vector<T2, T>, inputSymbol, outputSymbol);
    } else if (is_value(inputSymbolInfo.stype)) {
      T2* inputSymbol = symbolManager.getValue(input, T2());
      callFunction(jit, _wrapper_append_value_v2<T2, T>, inputSymbol, outputSymbol);
    } else {
      // unknow stype
      throw RuntimeError(
          "Invalid symbol type" + std::to_string(static_cast<int>(inputSymbolInfo.stype)));
    }
  } else if (inputSymbolInfo.dtype == tfcc::runtime::common::INT64) {
    using T2 = typename DataTypeInfo<tfcc::runtime::common::INT64>::CPPType;
    if (is_vector(inputSymbolInfo.stype)) {
      std::vector<T2>* inputSymbol = symbolManager.getVector(input, T2());
      callFunction(jit, _wrapper_append_vector<T2, T>, inputSymbol, outputSymbol);
    } else if (is_value(inputSymbolInfo.stype)) {
      T2* inputSymbol = symbolManager.getValue(input, T2());
      callFunction(jit, _wrapper_append_value_v2<T2, T>, inputSymbol, outputSymbol);
    } else {
      // unknow stype
      throw RuntimeError(
          "Invalid symbol type" + std::to_string(static_cast<int>(inputSymbolInfo.stype)));
    }
  } else if (inputSymbolInfo.dtype == tfcc::runtime::common::BOOL) {
    using T2 = typename DataTypeInfo<tfcc::runtime::common::BOOL>::CPPType;
    if (is_vector(inputSymbolInfo.stype)) {
      std::vector<T2>* inputSymbol = symbolManager.getVector(input, T2());
      callFunction(jit, _wrapper_append_vector<T2, T>, inputSymbol, outputSymbol);
    } else if (is_value(inputSymbolInfo.stype)) {
      T2* inputSymbol = symbolManager.getValue(input, T2());
      callFunction(jit, _wrapper_append_value_v2<T2, T>, inputSymbol, outputSymbol);
    } else {
      // unknow stype
      throw RuntimeError(
          "Invalid symbol type" + std::to_string(static_cast<int>(inputSymbolInfo.stype)));
    }
  } else {
    throw RuntimeError("Invalid data type " + std::to_string(inputSymbolInfo.dtype));
  }
}

std::vector<std::unique_ptr<Operation>> get_create_vector_operations() {
#define DEFINE_FUNC(dtype) \
  operations.emplace_back(std::unique_ptr<Operation>(new CreateVector<dtype>()));
  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace base
}  // namespace runtime
}  // namespace tfcc
