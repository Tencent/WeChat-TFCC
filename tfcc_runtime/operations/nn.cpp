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

#include "nn.h"

#include "tfcc_runtime/operations/nn/averagepool1d.h"
#include "tfcc_runtime/operations/nn/averagepool2d.h"
#include "tfcc_runtime/operations/nn/batchnormalization.h"
#include "tfcc_runtime/operations/nn/conv2d.h"
#include "tfcc_runtime/operations/nn/gather.h"
#include "tfcc_runtime/operations/nn/globalaveragepool2d.h"
#include "tfcc_runtime/operations/nn/layernormalization.h"
#include "tfcc_runtime/operations/nn/maxpool1d.h"
#include "tfcc_runtime/operations/nn/maxpool2d.h"
#include "tfcc_runtime/operations/nn/nonzero.h"
#include "tfcc_runtime/operations/nn/onehot.h"
#include "tfcc_runtime/operations/nn/scatterndwithdata.h"
#include "tfcc_runtime/operations/nn/where.h"
#include "tfcc_runtime/operations/operation.h"

namespace tfcc {
namespace runtime {
namespace nn {

static void append_vector(
    std::vector<std::unique_ptr<Operation>>& dst, std::vector<std::unique_ptr<Operation>> src) {
  for (auto& op : src) {
    dst.emplace_back(std::move(op));
  }
}

std::vector<std::unique_ptr<Operation>> get_all_operations() {
  std::vector<std::unique_ptr<Operation>> operations;
  append_vector(operations, get_average_pool1d_operations());
  append_vector(operations, get_average_pool2d_operations());
  append_vector(operations, get_gather_operations());
  append_vector(operations, get_batch_normalization_operations());
  append_vector(operations, get_conv2d_operations());
  append_vector(operations, get_global_average_pool2d_operations());
  append_vector(operations, get_layer_normalization_operations());
  append_vector(operations, get_max_pool1d_operations());
  append_vector(operations, get_max_pool2d_operations());
  append_vector(operations, get_non_zero_operations());
  append_vector(operations, get_one_hot_operations());
  append_vector(operations, get_scatter_nd_with_data_operations());
  append_vector(operations, get_where_operations());
  return operations;
}

}  // namespace nn
}  // namespace runtime
}  // namespace tfcc
