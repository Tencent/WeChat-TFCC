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

#include "createtensor.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/base.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace base {

template <class T, class SHAPE>
static const char* _wrapper_create_tensor(
    T a, const std::vector<SHAPE>* shape, tfcc::Variable<T>* result) noexcept {
  tfcc::Variable<T> b;
  try {
    std::vector<unsigned> realShape;
    for (SHAPE s : *shape) {
      realShape.push_back(s);
    }
    tfcc::Variable<T> value(realShape);
    tfcc::data::zeros(value);
    *result = value + a;
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType SHAPE_DTYPE>
std::string CreateTensor<DTYPE, SHAPE_DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::base::CreateTensor operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType SHAPE_DTYPE>
std::set<unsigned> CreateTensor<DTYPE, SHAPE_DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::base::CreateTensor::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType SHAPE_DTYPE>
std::unique_ptr<OperationResource> CreateTensor<DTYPE, SHAPE_DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 1 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::base::CreateTensor>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  if (symbolManager.getSymbolInfo(node.outputs(0)).dtype != DTYPE) {
    return nullptr;
  }
  if (symbolManager.getSymbolInfo(node.inputs(0)).dtype != SHAPE_DTYPE) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource);
  tfcc::runtime::operations::base::CreateTensor operation;
  node.operation().UnpackTo(&operation);
  if (operation.data_type() != DTYPE) {
    return nullptr;
  }
  T value = static_cast<T>(0);
  if (!get_value(operation.value(), value)) {
    return nullptr;
  }

  auto& jit = getJIT(graph);

  const std::vector<SHAPE>* shapeSymbol = symbolManager.getVector(node.inputs(0), SHAPE());
  tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());

  callFunction(jit, _wrapper_create_tensor<T, SHAPE>, value, shapeSymbol, outputSymbol);

  return resource;
}

std::vector<std::unique_ptr<Operation>> get_create_tensor_operations() {
#define DEFINE_FUNC(dtype)                                                                   \
  operations.emplace_back(                                                                   \
      std::unique_ptr<Operation>(new CreateTensor<dtype, tfcc::runtime::common::INT32>()));  \
  operations.emplace_back(                                                                   \
      std::unique_ptr<Operation>(new CreateTensor<dtype, tfcc::runtime::common::UINT32>())); \
  operations.emplace_back(                                                                   \
      std::unique_ptr<Operation>(new CreateTensor<dtype, tfcc::runtime::common::INT64>()));  \
  operations.emplace_back(                                                                   \
      std::unique_ptr<Operation>(new CreateTensor<dtype, tfcc::runtime::common::UINT64>()));

  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace base
}  // namespace runtime
}  // namespace tfcc
