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

#include "range.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/base.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace base {

template <class T>
static const char* _wrapper_range(
    const T* start, const T* limit, const T* delta, std::vector<T>* result) noexcept {
  result->clear();
  for (T x = *start; x < *limit; x += *delta) {
    result->push_back(x);
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE>
std::string Range<DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::base::Range operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE>
std::set<unsigned> Range<DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::base::Range::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE>
std::unique_ptr<OperationResource> Range<DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 3 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::base::Range>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  if (symbolManager.getSymbolInfo(node.outputs(0)).dtype != DTYPE) {
    return nullptr;
  }
  for (int i = 0; i < 3; ++i) {
    if (symbolManager.getSymbolInfo(node.inputs(i)).dtype != DTYPE) {
      return nullptr;
    }
  }

  std::unique_ptr<OperationResource> resource(new OperationResource);
  tfcc::runtime::operations::base::Range operation;
  node.operation().UnpackTo(&operation);

  auto& jit = getJIT(graph);

  const T* startSymbol = symbolManager.getValue(node.inputs(0), T());
  const T* limitSymbol = symbolManager.getValue(node.inputs(1), T());
  const T* deltaSymbol = symbolManager.getValue(node.inputs(2), T());
  std::vector<T>* outputSymbol = symbolManager.getVector(node.outputs(0), T());

  callFunction(jit, _wrapper_range<T>, startSymbol, limitSymbol, deltaSymbol, outputSymbol);

  return resource;
}

std::vector<std::unique_ptr<Operation>> get_range_operations() {
#define DEFINE_FUNC(dtype) operations.emplace_back(std::unique_ptr<Operation>(new Range<dtype>()));

  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace base
}  // namespace runtime
}  // namespace tfcc
