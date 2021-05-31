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
#include <ostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/data.pb.h"
#include "tfcc_runtime/proto/model.pb.h"
#include "tfcc_runtime/utils/modeldataguard.h"

namespace tfcc {

class NPZDataLoader;
class Session;

namespace runtime {

class Model {
  struct ThreadLocalData {
    std::vector<std::unique_ptr<Graph>> graphs;
  };

 private:
  static tfcc::SpinLock _globalMutex;
  static std::unordered_set<Model*> _allModels;

  tfcc::SpinLock _mutex;
  std::unordered_map<const tfcc::Session*, ThreadLocalData> _dataMap;
  tfcc::runtime::model::Model _modelProto;
  tfcc::runtime::model::Model _fusionOpModelProto;
  std::string _userData;
  std::unique_ptr<ModelDataGuard> _dataGuard;

 public:
  explicit Model(const std::string& data);
  ~Model();

  /**
   * Run the model and generate the output
   * @param inputs, An input protobuf
   * @param outputs, An output protobuf reference. The result of the inference will be stored
   * inside.
   */
  void run(const tfcc::runtime::data::Inputs& inputs, tfcc::runtime::data::Outputs& outputs);

  /**
   * Test data and print summary to std::cout
   * @param npz A npz data with all inputs and outputs.
   * @return True if pass the test.
   */
  bool test(const std::string& npz);

  /**
   * Test data and print summary to stream
   * @param npz A npz data with all inputs and outputs.
   * @param stream The stream to print summary.
   * @return True if pass the test.
   */
  bool test(const std::string& npz, std::ostream& stream);

  /**
   * Get entrance graph. It can be used for inference.
   * @return A graph.
   */
  Graph& getEntranceGraph();

  const std::string& getUserData() const;

 private:
  friend class Operation;

  ThreadLocalData& initializeGraphs(tfcc::Session* session);
  void releaseGraphs(const tfcc::Session* session);

  template <class T>
  void addInputToGraph(Graph& graph, const tfcc::runtime::data::DataItem& input, T);
  template <class T>
  void getOutputFromGraph(
      Graph& graph, const std::string& name, tfcc::runtime::data::DataItem& output, T);

  Graph& getGraph(const std::string& name);

  static void releaseCallback(const tfcc::Session* session);
};

}  // namespace runtime
}  // namespace tfcc
