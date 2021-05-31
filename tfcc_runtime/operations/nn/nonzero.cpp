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

#include "nonzero.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/nn.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace nn {

template <class T>
static tfcc::Variable<int64_t> _non_zero_dimn(const tfcc::Tensor<T>& input) {
  std::vector<unsigned> skip = input.shape().toVector();
  for (size_t i = 1; i < skip.size(); ++i) {
    skip[skip.size() - i - 1] *= skip[skip.size() - i];
  }
  skip.push_back(1);
  std::vector<T> data = tfcc::data::get(input);
  std::vector<int64_t> result;
  for (unsigned i = 0; i < input.size(); ++i) {
    if (data[i] > 0 || data[i] < 0) {
      unsigned pos = i;
      for (size_t j = 0; j < input.shape().size(); ++j) {
        result.push_back(pos / skip[j + 1]);
        pos = pos % skip[j + 1];
      }
    }
  }

  return tfcc::data::set(result, {static_cast<unsigned>(result.size()), 1u});
}

template <class T>
static tfcc::Variable<int64_t> _non_zero(const Tensor<T>& input) {
  if (input.shape().size() != 1) {
    return _non_zero_dimn(input);
  }
  std::vector<T> data = tfcc::data::get(input);
  std::vector<int64_t> result;
  for (unsigned i = 0; i < input.size(); ++i) {
    if (data[i] > 0 || data[i] < 0) {
      result.push_back(i);
    }
  }

  return tfcc::data::set(result, {static_cast<unsigned>(result.size()), 1u});
}

template <class T>
static const char* _wrapper_non_zero(
    const tfcc::Tensor<T>* a, tfcc::Variable<int64_t>* result) noexcept {
  try {
    *result = _non_zero(*a);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE>
std::string NonZero<DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::nn::NonZero operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE>
std::set<unsigned> NonZero<DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::nn::NonZero::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE>
std::unique_ptr<OperationResource> NonZero<DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 1 || node.outputs_size() != 1) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::nn::NonZero>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  if (symbolManager.getSymbolInfo(node.outputs(0)).dtype != tfcc::runtime::common::INT64) {
    return nullptr;
  }
  if (symbolManager.getSymbolInfo(node.inputs(0)).dtype != DTYPE) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource);
  tfcc::runtime::operations::nn::NonZero operation;
  node.operation().UnpackTo(&operation);

  auto& jit = getJIT(graph);
  const tfcc::Tensor<T>* inputSymbol = symbolManager.getTensor(node.inputs(0), T());
  tfcc::Variable<int64_t>* outputSymbol = symbolManager.getVariable(node.outputs(0), int64_t());
  callFunction(jit, _wrapper_non_zero<T>, inputSymbol, outputSymbol);
  return resource;
}

std::vector<std::unique_ptr<Operation>> get_non_zero_operations() {
#define DEFINE_FUNC(dtype) \
  operations.emplace_back(std::unique_ptr<Operation>(new NonZero<dtype>()));
  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace nn
}  // namespace runtime
}  // namespace tfcc
