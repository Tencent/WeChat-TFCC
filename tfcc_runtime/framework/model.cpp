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

#include "model.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <mutex>
#include <unordered_set>

#include "framework/tfcc_mklinstruction.h"
#include "tfcc.h"
#include "tfcc_mkl.h"

#include "tfcc_runtime/exceptions/runtimeerror.h"
#include "tfcc_runtime/framework/symbolmanager.h"
#include "tfcc_runtime/framework/types.h"

namespace tfcc {
namespace runtime {

template <class T>
static inline bool _is_similar(T a, T b) {
  return a == b;
}

static inline bool _is_similar(float a, float b) {
  if (isnan(a) && isnan(b)) {
    return true;
  }
  return a == b;
}

static inline bool _is_similar(double a, double b) {
  if (isnan(a) && isnan(b)) {
    return true;
  }
  return a == b;
}

tfcc::SpinLock Model::_globalMutex;
std::unordered_set<Model*> Model::_allModels;

Model::Model(const std::string& data) {
  {
    std::lock_guard<tfcc::SpinLock> lck(_globalMutex);
    _allModels.insert(this);
  }

  auto zipMap = tfcc::unzip(data);
  if (zipMap.find("model.pb") == zipMap.end()) {
    throw RuntimeError("Invalid model data");
  }
  if (zipMap.find("model.npz") == zipMap.end()) {
    throw RuntimeError("Invalid model data");
  }
  if (zipMap.find("userdata") != zipMap.end()) {
    _userData = std::move(zipMap["userdata"]);
  }

  const std::string& protoData = zipMap["model.pb"];
  const std::string& npzData = zipMap["model.npz"];
  _modelProto.ParseFromString(protoData);

  if (zipMap.find("model.cpu.pb") != zipMap.end()) {
    _fusionOpModelProto.ParseFromString(zipMap["model.cpu.pb"]);
  }

  std::string entranceName = _modelProto.entrance_graph_name();
  auto comp = [entranceName](
                  const tfcc::runtime::model::Graph& g1, const tfcc::runtime::model::Graph& g2) {
    if (g1.name() == entranceName && g2.name() == entranceName) {
      return g1.name() < g2.name();
    }
    if (g1.name() == entranceName && g2.name() != entranceName) {
      return true;
    }
    if (g1.name() != entranceName && g2.name() == entranceName) {
      return false;
    }
    return g1.name() < g2.name();
  };
  std::sort(_modelProto.mutable_graphs()->begin(), _modelProto.mutable_graphs()->end(), comp);
  if (_modelProto.graphs_size() < 1 || _modelProto.graphs(0).name() != entranceName) {
    throw RuntimeError("Invalid model proto");
  }

  if (zipMap.find("model.cpu.pb") != zipMap.end()) {
    std::sort(
        _fusionOpModelProto.mutable_graphs()->begin(), _fusionOpModelProto.mutable_graphs()->end(),
        comp);
    if (_fusionOpModelProto.graphs_size() < 1 ||
        _fusionOpModelProto.graphs(0).name() != entranceName) {
      throw RuntimeError("Invalid model proto");
    }
  }
  _dataGuard = std::unique_ptr<ModelDataGuard>(new ModelDataGuard(npzData));
}

Model::~Model() {
  std::lock_guard<tfcc::SpinLock> lck(_globalMutex);
  _allModels.erase(this);
}

template <class T>
inline void Model::addInputToGraph(Graph& graph, const tfcc::runtime::data::DataItem& input, T) {
  tfcc::Shape shape(std::vector<unsigned>(input.shape().begin(), input.shape().end()));
  if (input.data().size() != shape.area() * sizeof(T)) {
    throw RuntimeError("Invalid data size");
  }
  const T* p = reinterpret_cast<const T*>(input.data().data());
  std::vector<T> tmpVec(p, p + shape.area());
  tfcc::Variable<T> variable = tfcc::data::set(std::move(tmpVec), shape);
  graph.setInput(input.name(), std::move(variable));
}

template <class T>
inline void Model::getOutputFromGraph(
    Graph& graph, const std::string& name, tfcc::runtime::data::DataItem& output, T) {
  const SymbolInfo& info = graph.getSymbolManager().getSymbolInfo(name);
  output.set_name(name);
  output.set_dtype(info.dtype);
  const tfcc::Tensor<T>& tensor = graph.getTensor(name, T());
  for (size_t i = 0; i < tensor.shape().size(); ++i) {
    output.add_shape(tensor.shape(i));
  }
  std::vector<T> v = tfcc::data::get(tensor);
  output.mutable_data()->assign(
      reinterpret_cast<const char*>(v.data()), reinterpret_cast<const char*>(v.data() + v.size()));
}

void Model::run(const data::Inputs& inputs, data::Outputs& outputs) {
  Graph& graph = getEntranceGraph();

  for (int i = 0; i < inputs.items_size(); ++i) {
    auto& item = inputs.items(i);
    switch (item.dtype()) {
#define INPUT_PB_VARIABLE_TO_GRAPH_SWITCH_FUNC(dtype)                          \
  {                                                                            \
    case dtype:                                                                \
      addInputToGraph(graph, inputs.items(i), DataTypeInfo<dtype>::CPPType()); \
      break;                                                                   \
  }
      TFCC_RUNTIME_FOR_ALL_DATA_TYPE(INPUT_PB_VARIABLE_TO_GRAPH_SWITCH_FUNC);
      default:
        throw RuntimeError("Invalid input dtype");
    }
#undef INPUT_PB_VARIABLE_TO_GRAPH_SWITCH_FUNC
  }

  graph.process();

  for (const std::string& out : _modelProto.graphs(0).outputs()) {
    const SymbolInfo& info = graph.getSymbolManager().getSymbolInfo(out);
    auto outData = outputs.add_items();

    switch (info.dtype) {
#define PARSE_GRAPH_OUTPUT_2_PB_SWITCH_FUNC(dtype)                            \
  case (dtype):                                                               \
    getOutputFromGraph(graph, out, *outData, DataTypeInfo<dtype>::CPPType()); \
    break;
      TFCC_RUNTIME_FOR_ALL_DATA_TYPE(PARSE_GRAPH_OUTPUT_2_PB_SWITCH_FUNC);
      default:
        throw RuntimeError("Invalid output type");
    }
#undef PARSE_GRAPH_OUTPUT_2_PB_SWITCH_FUNC
  }
}

bool Model::test(const std::string& npz) { return test(npz, std::cout); }

bool Model::test(const std::string& npz, std::ostream& stream) {
  ModelDataGuard testDataGuard(npz);
  Graph& graph = getEntranceGraph();

  auto scopeG = tfcc::Scope::scope(testDataGuard.getMID());

  tfcc::Coster coster;

  for (const std::string& inp : _modelProto.graphs(0).inputs()) {
    const SymbolInfo& info = graph.getSymbolManager().getSymbolInfo(inp);
    if (info.stype == SymbolType::VIEW) {
      switch (info.dtype) {
#define VIEW_INPUT_SWICH_FUNC(dtype)                                                     \
  case dtype:                                                                            \
    graph.setInput(inp, tfcc::Constant<DataTypeInfo<dtype>::CPPType>::getConstant(inp)); \
    break;
        TFCC_RUNTIME_FOR_ALL_DATA_TYPE(VIEW_INPUT_SWICH_FUNC);
        default:
          throw RuntimeError("Invalid input dtype");
      }
#undef VIEW_INPUT_SWICH_FUNC
    } else if (is_value(info.stype)) {
      switch (info.dtype) {
#define VALUE_INPUT_SWICH_FUNC(dtype)                                                      \
  case dtype:                                                                              \
    graph.setInput(inp, tfcc::Configure<DataTypeInfo<dtype>::CPPType>::getConfigure(inp)); \
    break;
        TFCC_RUNTIME_FOR_ALL_DATA_TYPE(VALUE_INPUT_SWICH_FUNC);
        default:
          throw RuntimeError("Invalid input dtype");
      }
#undef VALUE_INPUT_SWICH_FUNC
    } else {
      // stype error
      throw RuntimeError("Invalid input stype");
    }
  }

  stream << "set inputs cost: " << coster.lap().milliseconds() << " ms" << std::endl;
  coster.reset();

  graph.process();
  tfcc::Session::getThreadDefault()->sync();
  stream << "process cost: " << coster.lap().milliseconds() << " ms" << std::endl;

#ifdef __linux__
  constexpr char YELLOW_COLOR[] = "\033[33m";
  constexpr char RESET_ALL[] = "\033[0m";
#else
  constexpr char YELLOW_COLOR[] = "";
  constexpr char RESET_ALL[] = "";
#endif

  bool succ = true;
  for (const std::string& out : _modelProto.graphs(0).outputs()) {
    const SymbolInfo& info = graph.getSymbolManager().getSymbolInfo(out);
    if (is_tensor(info.stype)) {
      switch (info.dtype) {
#define TENSOR_OUTPUT_SWICH_FUNC(dtype)                                                         \
  case dtype:                                                                                   \
    if (testDataGuard.getLoader().hasData(out)) {                                               \
      bool similar = tfcc::is_similar(                                                          \
          graph.getTensor(out, DataTypeInfo<dtype>::CPPType()),                                 \
          tfcc::Constant<DataTypeInfo<dtype>::CPPType>::getConstant(out));                      \
      succ = succ && similar;                                                                   \
    } else {                                                                                    \
      stream << YELLOW_COLOR << "WARNING output [" << out << "] not in test data" << RESET_ALL  \
             << std::endl;                                                                      \
      succ = false;                                                                             \
    }                                                                                           \
    stream << out << ": " << graph.getTensor(out, DataTypeInfo<dtype>::CPPType()) << std::endl; \
    break;
        TFCC_RUNTIME_FOR_ALL_DATA_TYPE(TENSOR_OUTPUT_SWICH_FUNC);
        default:
          throw RuntimeError("Invalid input dtype");
      }
#undef TENSOR_OUTPUT_SWICH_FUNC
    } else if (is_value(info.stype)) {
      switch (info.dtype) {
#define VALUE_OUTPUT_SWICH_FUNC(dtype)                                                         \
  case dtype:                                                                                  \
    if (testDataGuard.getLoader().hasData(out)) {                                              \
      bool similar = _is_similar(                                                              \
          graph.getValue(out, DataTypeInfo<dtype>::CPPType()),                                 \
          tfcc::Configure<DataTypeInfo<dtype>::CPPType>::getConfigure(out));                   \
      succ = succ && similar;                                                                  \
    } else {                                                                                   \
      stream << YELLOW_COLOR << "WARNING output [" << out << "] not in test data" << RESET_ALL \
             << std::endl;                                                                     \
      succ = false;                                                                            \
    }                                                                                          \
    stream << out << ": " << graph.getValue(out, DataTypeInfo<dtype>::CPPType()) << std::endl; \
    break;
        TFCC_RUNTIME_FOR_ALL_DATA_TYPE(VALUE_OUTPUT_SWICH_FUNC);
        default:
          throw RuntimeError("Invalid input dtype");
      }
#undef VALUE_OUTPUT_SWICH_FUNC
    } else {
      // stype error
      throw RuntimeError("Invalid input stype");
    }
  }

  return succ;
}

Model::ThreadLocalData& Model::initializeGraphs(tfcc::Session* session) {
  ThreadLocalData* data = nullptr;
  {
    std::lock_guard<tfcc::SpinLock> lck(_mutex);
    auto it = _dataMap.find(session);
    if (it != _dataMap.end()) {
      return it->second;
    }
    data = &_dataMap[session];
  }

  thread_local std::unordered_set<const tfcc::Session*> sessionSet;
  if (sessionSet.find(session) == sessionSet.end()) {
    session->registerReleaseCallback(releaseCallback);
    sessionSet.insert(session);
  }

  auto scopeG = tfcc::Scope::scope(_dataGuard->getMID());

  // If MKL environment and AVX256 are available, run fusionop model
  auto device = dynamic_cast<tfcc::MKLDevice*>(tfcc::Device::getThreadDefault());
  if ((device != nullptr) && (device->getCPUInstructionFlags() & tfcc::MKLInstruction::AVX256) &&
      (_fusionOpModelProto.graphs().size() > 0)) {
    for (auto& graphProto : _fusionOpModelProto.graphs()) {
      data->graphs.push_back(std::unique_ptr<Graph>(new Graph(*this, graphProto)));
    }
  } else {
    for (auto& graphProto : _modelProto.graphs()) {
      data->graphs.push_back(std::unique_ptr<Graph>(new Graph(*this, graphProto)));
    }
  }

  for (auto& graph : data->graphs) {
    graph->buildGraph();
  }
  return *data;
}

void Model::releaseGraphs(const tfcc::Session* session) {
  ThreadLocalData data;
  {
    std::lock_guard<tfcc::SpinLock> lck(_mutex);
    data = std::move(_dataMap[session]);
    _dataMap.erase(session);
  }
}

Graph& Model::getEntranceGraph() {
  tfcc::Session* session = tfcc::Session::getThreadDefault();
  auto& data = initializeGraphs(session);
  return *data.graphs[0];
}

const std::string& Model::getUserData() const { return _userData; }

Graph& Model::getGraph(const std::string& name) {
  tfcc::Session* session = tfcc::Session::getThreadDefault();
  auto& data = initializeGraphs(session);
  for (auto& graph : data.graphs) {
    if (graph->getGraphName() == name) {
      return *graph;
    }
  }
  throw RuntimeError("Invalid graph name");
}

void Model::releaseCallback(const tfcc::Session* session) {
  std::lock_guard<tfcc::SpinLock> lck(_globalMutex);
  for (Model* model : _allModels) {
    model->releaseGraphs(session);
  }
}

}  // namespace runtime
}  // namespace tfcc
