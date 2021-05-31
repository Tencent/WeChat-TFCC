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

#include "reshape.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/base.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace base {

template <class T, class SHAPE>
static const char* _wrapper_reshape(
    const tfcc::Tensor<T>* a, const std::vector<SHAPE>* shape, tfcc::View<T>* result) noexcept {
  try {
    int pos = -1;
    unsigned area = 1;
    std::vector<unsigned> realShape(shape->size());
    for (size_t i = 0; i < shape->size(); ++i) {
      if ((*shape)[i] == 0) {
        if (i >= a->shape().size()) {
          return "Reshape: invalid shape";
        }
        realShape[i] = a->shape(i);
        area *= a->shape(i);
      } else if ((*shape)[i] < 0) {
        if (pos >= 0) {
          return "Reshape: invalid shape";
        }
        pos = i;
      } else {
        realShape[i] = (*shape)[i];
        area *= (*shape)[i];
      }
    }
    if (pos >= 0) {
      realShape[pos] = a->size() / area;
    }
    *result = tfcc::View<T>(*a, realShape);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType SHAPE_DTYPE>
std::string Reshape<DTYPE, SHAPE_DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::base::Reshape operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType SHAPE_DTYPE>
std::set<unsigned> Reshape<DTYPE, SHAPE_DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::base::Reshape::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType SHAPE_DTYPE>
std::unique_ptr<OperationResource> Reshape<DTYPE, SHAPE_DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 2 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::base::Reshape>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  if (symbolManager.getSymbolInfo(node.outputs(0)).dtype != DTYPE ||
      symbolManager.getSymbolInfo(node.inputs(0)).dtype != DTYPE) {
    return nullptr;
  }
  if (symbolManager.getSymbolInfo(node.inputs(1)).dtype != SHAPE_DTYPE) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource);
  tfcc::runtime::operations::base::Reshape operation;
  node.operation().UnpackTo(&operation);

  auto& jit = getJIT(graph);

  const tfcc::Tensor<T>* inputSymbol = symbolManager.getTensor(node.inputs(0), T());
  const std::vector<SHAPE>* shapeSymbol = symbolManager.getVector(node.inputs(1), SHAPE());
  tfcc::View<T>* outputSymbol = symbolManager.getView(node.outputs(0), T());

  callFunction(jit, _wrapper_reshape<T, SHAPE>, inputSymbol, shapeSymbol, outputSymbol);

  return resource;
}

std::vector<std::unique_ptr<Operation>> get_reshape_operations() {
#define DEFINE_FUNC(dtype)                                                              \
  operations.emplace_back(                                                              \
      std::unique_ptr<Operation>(new Reshape<dtype, tfcc::runtime::common::INT32>()));  \
  operations.emplace_back(                                                              \
      std::unique_ptr<Operation>(new Reshape<dtype, tfcc::runtime::common::UINT32>())); \
  operations.emplace_back(                                                              \
      std::unique_ptr<Operation>(new Reshape<dtype, tfcc::runtime::common::INT64>()));  \
  operations.emplace_back(                                                              \
      std::unique_ptr<Operation>(new Reshape<dtype, tfcc::runtime::common::UINT64>()));

  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace base
}  // namespace runtime
}  // namespace tfcc
