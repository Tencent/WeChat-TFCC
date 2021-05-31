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
#include <set>
#include <string>

#include "tfcc_runtime/proto/model.pb.h"
#include "tfcc_runtime/utils/jitutils.h"

namespace tfcc {
namespace runtime {

class Model;
class Graph;
class SymbolManager;

class OperationResource {
 public:
  virtual ~OperationResource();
};

class Operation {
 public:
  Operation();
  virtual ~Operation();

  virtual std::string getOperationTypeUrl() const = 0;
  virtual std::set<unsigned> getOperationVersions() const = 0;
  virtual std::unique_ptr<OperationResource> process(
      const tfcc::runtime::model::Node& node, Graph& graph) const = 0;
  virtual std::unique_ptr<OperationResource> debug(
      const tfcc::runtime::model::Node& node, Graph& graph);

 protected:
  static SymbolManager& getSymbolManager(Graph& graph);
  static tfcc::Xbyak::CodeGenerator& getJIT(Graph& graph);
  static Model& getModel(Graph& graph);
  static Graph& getGraph(Model& model, const std::string& name);

  template <class Func, class... Args>
  static void callFunction(Xbyak::CodeGenerator& jit, Func func, Args... arguments) {
    jit::call_function(jit, func, arguments...);
  }

  static void setGraphInput(
      Xbyak::CodeGenerator& jit, Graph& dstGraph, std::string dstName, Graph& srcGraph,
      std::string srcName);
  static void getGraphOutput(
      Xbyak::CodeGenerator& jit, Graph& dstGraph, std::string dstName, Graph& srcGraph,
      std::string srcName);
  template <class T>
  static void printSymbol(
      Xbyak::CodeGenerator& jit, Graph& graph, std::string name, const char* prefix, T);

 private:
  template <class T>
  static void setGraphInputInner(
      Xbyak::CodeGenerator& jit, Graph& dstGraph, std::string dstName, Graph& srcGraph,
      std::string srcName, T);
  template <class T>
  static void getGraphOutputInner(
      Xbyak::CodeGenerator& jit, Graph& dstGraph, std::string dstName, Graph& srcGraph,
      std::string srcName, T);

  template <size_t COUNT, size_t POS>
  friend class _OperationAutoExplicitInstantiationWrapper;
};

std::unique_ptr<OperationResource> node_to_asm(
    const tfcc::runtime::model::Node& node, Graph& graph);

}  // namespace runtime
}  // namespace tfcc
