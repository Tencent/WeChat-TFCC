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

#include "tile.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/base.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace base {

template <class T, class IDX>
static const char* _wrapper_tile(
    const tfcc::Tensor<T>* a, const std::vector<IDX>* repeated,
    tfcc::Variable<T>* result) noexcept {
  tfcc::Variable<T> b;
  try {
    if (repeated->size() != a->shape().size()) {
      return "Tile: repeated error";
    }

    std::vector<unsigned> shape = a->shape().toVector();
    std::vector<unsigned> targetShape = shape;
    std::vector<unsigned> tempAShape;
    std::vector<unsigned> tempRepeats;
    for (size_t i = 0; i < repeated->size(); ++i) {
      IDX r = (*repeated)[i];
      if (r <= 0) {
        return "Tile: repeated error";
      }
      targetShape[i] *= r;
      tempAShape.push_back(1);
      tempAShape.push_back(shape[i]);
      tempRepeats.push_back(r);
      tempRepeats.push_back(shape[i]);
    }
    tfcc::View<T> tempA(*a, tempAShape);
    tfcc::Variable<T> repeatTensor(tempRepeats);
    tfcc::data::ones(repeatTensor);
    *result = tempA * repeatTensor;
    result->reshape(targetShape);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType IDX_DTYPE>
std::string Tile<DTYPE, IDX_DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::base::Tile operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType IDX_DTYPE>
std::set<unsigned> Tile<DTYPE, IDX_DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::base::Tile::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType IDX_DTYPE>
std::unique_ptr<OperationResource> Tile<DTYPE, IDX_DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 2 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::base::Tile>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  if (symbolManager.getSymbolInfo(node.outputs(0)).dtype != DTYPE ||
      symbolManager.getSymbolInfo(node.inputs(0)).dtype != DTYPE) {
    return nullptr;
  }
  if (symbolManager.getSymbolInfo(node.inputs(1)).dtype != IDX_DTYPE) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource);
  tfcc::runtime::operations::base::Tile operation;
  node.operation().UnpackTo(&operation);

  auto& jit = getJIT(graph);

  const tfcc::Tensor<T>* inputSymbol = symbolManager.getTensor(node.inputs(0), T());
  const std::vector<IDX>* repeatedSymbol = symbolManager.getVector(node.inputs(1), IDX());
  tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());

  callFunction(jit, _wrapper_tile<T, IDX>, inputSymbol, repeatedSymbol, outputSymbol);

  return resource;
}

std::vector<std::unique_ptr<Operation>> get_tile_operations() {
#define DEFINE_FUNC(dtype)                                                           \
  operations.emplace_back(                                                           \
      std::unique_ptr<Operation>(new Tile<dtype, tfcc::runtime::common::INT32>()));  \
  operations.emplace_back(                                                           \
      std::unique_ptr<Operation>(new Tile<dtype, tfcc::runtime::common::UINT32>())); \
  operations.emplace_back(                                                           \
      std::unique_ptr<Operation>(new Tile<dtype, tfcc::runtime::common::INT64>()));  \
  operations.emplace_back(                                                           \
      std::unique_ptr<Operation>(new Tile<dtype, tfcc::runtime::common::UINT64>()));

  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace base
}  // namespace runtime
}  // namespace tfcc
