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

#include "topk.h"

#include <algorithm>

#include "tfcc_runtime/framework/graph.h"
#include "tfcc_runtime/proto/operations/math.pb.h"
#include "tfcc_runtime/utils/commutils.h"

namespace tfcc {
namespace runtime {
namespace math {

template <class T>
static inline std::tuple<tfcc::Variable<T>, tfcc::Variable<int64_t>> _top_k(
    const Tensor<T>& a, unsigned k) {
  struct VP {
    T v;
    size_t pos;

    bool operator<(const VP& v2) { return v > v2.v; }
  };

  std::vector<T> vec = tfcc::data::get(a);
  unsigned chunk = a.shape(a.shape().size() - 1);
  unsigned batch = a.size() / chunk;
  k = std::min(k, chunk);

  std::vector<unsigned> shape = a.shape().toVector();
  shape[shape.size() - 1] = k;

  std::vector<T> values;
  std::vector<int64_t> indices;

  for (unsigned i = 0; i < batch; ++i) {
    size_t currentCount = 0;
    std::vector<VP> vs(k + 1);
    for (unsigned j = 0; j < chunk; ++j) {
      if (currentCount + 1 >= vs.size() && vec[i * chunk + j] <= vs[0].v) {
        continue;
      }
      vs[currentCount] = {vec[i * chunk + j], j};
      ++currentCount;
      std::push_heap(vs.begin(), vs.begin() + currentCount);
      if (currentCount == vs.size()) {
        std::pop_heap(vs.begin(), vs.begin() + currentCount);
        --currentCount;
      }
    }
    std::sort_heap(vs.begin(), vs.begin() + currentCount);
    for (unsigned j = 0; j < k; ++j) {
      values.push_back(vs[j].v);
      indices.push_back(vs[j].pos);
    }
  }

  return std::make_tuple(tfcc::data::set(values, shape), tfcc::data::set(indices, shape));
}

template <class T, class LEN>
static const char* _wrapper_top_k(
    const tfcc::Tensor<T>* a, const LEN* k, tfcc::Variable<T>* result,
    tfcc::Variable<int64_t>* indices) noexcept {
  if (*k <= 0) {
    return "Invalid k";
  }
  unsigned num = static_cast<unsigned>(*k);
  try {
    std::tie(*result, *indices) = _top_k(*a, num);
  } catch (std::exception& e) {
    thread_local std::string reason = e.what();
    return reason.c_str();
  }
  return nullptr;
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType LEN_DTYPE>
std::string TopK<DTYPE, LEN_DTYPE>::getOperationTypeUrl() const {
  tfcc::runtime::operations::math::TopK operation;
  return get_protobuf_type_url(operation);
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType LEN_DTYPE>
std::set<unsigned> TopK<DTYPE, LEN_DTYPE>::getOperationVersions() const {
  std::set<unsigned> versions = {tfcc::runtime::operations::math::TopK::VERSION_1};
  return versions;
}

template <tfcc::runtime::common::DataType DTYPE, tfcc::runtime::common::DataType LEN_DTYPE>
std::unique_ptr<OperationResource> TopK<DTYPE, LEN_DTYPE>::process(
    const tfcc::runtime::model::Node& node, Graph& graph) const {
  if (node.inputs_size() != 2 || node.outputs_size() != 2) {
    return nullptr;
  }
  if (!node.operation().Is<tfcc::runtime::operations::math::TopK>()) {
    return nullptr;
  }
  auto& symbolManager = getSymbolManager(graph);
  if (symbolManager.getSymbolInfo(node.outputs(0)).dtype != DTYPE) {
    return nullptr;
  }
  if (symbolManager.getSymbolInfo(node.inputs(0)).dtype != DTYPE ||
      symbolManager.getSymbolInfo(node.inputs(1)).dtype != LEN_DTYPE) {
    return nullptr;
  }

  std::unique_ptr<OperationResource> resource(new OperationResource);
  tfcc::runtime::operations::math::TopK operation;
  node.operation().UnpackTo(&operation);

  auto& jit = getJIT(graph);

  const tfcc::Tensor<T>* inputSymbol = symbolManager.getTensor(node.inputs(0), T());
  const LEN* kSymbol = symbolManager.getValue(node.inputs(1), LEN());
  tfcc::Variable<T>* outputSymbol = symbolManager.getVariable(node.outputs(0), T());
  tfcc::Variable<int64_t>* indicesSymbol = symbolManager.getVariable(node.outputs(1), int64_t());

  callFunction(jit, _wrapper_top_k<T, LEN>, inputSymbol, kSymbol, outputSymbol, indicesSymbol);

  return resource;
}

std::vector<std::unique_ptr<Operation>> get_top_k_operations() {
#define DEFINE_FUNC(dtype)                                                           \
  operations.emplace_back(                                                           \
      std::unique_ptr<Operation>(new TopK<dtype, tfcc::runtime::common::UINT8>()));  \
  operations.emplace_back(                                                           \
      std::unique_ptr<Operation>(new TopK<dtype, tfcc::runtime::common::INT8>()));   \
  operations.emplace_back(                                                           \
      std::unique_ptr<Operation>(new TopK<dtype, tfcc::runtime::common::UINT16>())); \
  operations.emplace_back(                                                           \
      std::unique_ptr<Operation>(new TopK<dtype, tfcc::runtime::common::INT16>()));  \
  operations.emplace_back(                                                           \
      std::unique_ptr<Operation>(new TopK<dtype, tfcc::runtime::common::UINT32>())); \
  operations.emplace_back(                                                           \
      std::unique_ptr<Operation>(new TopK<dtype, tfcc::runtime::common::INT32>()));  \
  operations.emplace_back(                                                           \
      std::unique_ptr<Operation>(new TopK<dtype, tfcc::runtime::common::UINT64>())); \
  operations.emplace_back(                                                           \
      std::unique_ptr<Operation>(new TopK<dtype, tfcc::runtime::common::INT64>()));

  std::vector<std::unique_ptr<Operation>> operations;
  TFCC_RUNTIME_FOR_ALL_DATA_TYPE(DEFINE_FUNC);
  return operations;
}

}  // namespace math
}  // namespace runtime
}  // namespace tfcc
