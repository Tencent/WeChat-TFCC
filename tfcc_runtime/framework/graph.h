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

#pragma once

#include <memory>
#include <type_traits>
#include <vector>

#include "tfcc.h"

#include "3rd/xbyak/xbyak.h"
#include "tfcc_runtime/framework/symbolmanager.h"
#include "tfcc_runtime/framework/types.h"
#include "tfcc_runtime/proto/model.pb.h"

namespace tfcc {
namespace runtime {

namespace jit {
void set_argument_inner(Xbyak::CodeGenerator& jit, size_t& pos, float value);
void set_argument_inner(Xbyak::CodeGenerator& jit, size_t& pos, double value);
void set_argument_inner(Xbyak::CodeGenerator& jit, size_t& pos, uint64_t value);
}  // namespace jit

class Model;
class Operation;
class OperationResource;

class Graph {
  Model& _model;
  const tfcc::runtime::model::Graph& _graphProto;
  tfcc::Xbyak::CodeGenerator _jit;
  std::unique_ptr<SymbolManager> _symbolManager;
  std::vector<std::unique_ptr<OperationResource>> _resources;
  const char* (*_func)();

  friend class Operation;
  friend class Model;
  friend void jit::set_argument_inner(Xbyak::CodeGenerator& jit, size_t& pos, float value);
  friend void jit::set_argument_inner(Xbyak::CodeGenerator& jit, size_t& pos, double value);
  friend void jit::set_argument_inner(Xbyak::CodeGenerator& jit, size_t& pos, uint64_t value);
  static constexpr size_t RESERVED_STACK_SIZE = 0x80;

 private:
  static thread_local bool _recordDetailErrorEnable;

 public:
  Graph(Model& model, const tfcc::runtime::model::Graph& graphProto);
  ~Graph();

  /**
   * Get graph name.
   * @return The name of the graph.
   */
  std::string getGraphName() const;

  /**
   * Get names of graph inputs.
   * @return A vector.
   */
  std::vector<std::string> getInputs() const;

  /**
   * Get names of graph outputs.
   * @return A vector.
   */
  std::vector<std::string> getOutputs() const;

  void process();

  /**
   * Set graph input.
   * @param name The input name.
   * @param tensor A tensor.
   * @warning You must keep the tensor alive before call process.
   */
  template <class T>
  void setInput(const std::string& name, const tfcc::Tensor<T>& tensor);

  /**
   * Set graph input.
   * @param name The input name.
   * @param variable A variable.
   */
  template <class T>
  void setInput(const std::string& name, tfcc::Variable<T>&& variable);

  /**
   * Set graph input.
   * @param name The input name.
   * @param value A value.
   */
  template <class T>
  typename std::enable_if<std::is_arithmetic<T>::value, void>::type setInput(
      const std::string& name, T value);

  /**
   * Get graph tensor output.
   * @param name The output name.
   */
  template <class T>
  const tfcc::Tensor<T>& getTensor(const std::string& name, T);

  /**
   * Get graph value output.
   * @param name The output name.
   */
  template <class T>
  T getValue(const std::string& name, T);

  /**
   * Get graph vector output.
   * @param name The output name.
   */
  template <class T>
  const std::vector<T>& getVector(const std::string& name, T);

 public:
  static void setRecordDetailErrorThreadLocal(bool enable);
  static bool isRecordDetailError();

 private:
  void buildGraph();

  SymbolManager& getSymbolManager();
  Xbyak::CodeGenerator& getJIT();
  Model& getModel();

 private:
  static const char* getErrorString(const char* str, Graph* graph, size_t index);
};

}  // namespace runtime
}  // namespace tfcc
