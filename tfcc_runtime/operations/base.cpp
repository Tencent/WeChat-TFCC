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

#include "base.h"

#include "tfcc_runtime/operations/base/at.h"
#include "tfcc_runtime/operations/base/at1.h"
#include "tfcc_runtime/operations/base/cast.h"
#include "tfcc_runtime/operations/base/concat.h"
#include "tfcc_runtime/operations/base/createtensor.h"
#include "tfcc_runtime/operations/base/createvector.h"
#include "tfcc_runtime/operations/base/expand.h"
#include "tfcc_runtime/operations/base/flatten.h"
#include "tfcc_runtime/operations/base/getdimension.h"
#include "tfcc_runtime/operations/base/getshape.h"
#include "tfcc_runtime/operations/base/identity.h"
#include "tfcc_runtime/operations/base/if_.h"
#include "tfcc_runtime/operations/base/loop.h"
#include "tfcc_runtime/operations/base/pad.h"
#include "tfcc_runtime/operations/base/range.h"
#include "tfcc_runtime/operations/base/reshape.h"
#include "tfcc_runtime/operations/base/slicev1.h"
#include "tfcc_runtime/operations/base/slicev2.h"
#include "tfcc_runtime/operations/base/split.h"
#include "tfcc_runtime/operations/base/squeeze.h"
#include "tfcc_runtime/operations/base/stack.h"
#include "tfcc_runtime/operations/base/tile.h"
#include "tfcc_runtime/operations/base/totensor.h"
#include "tfcc_runtime/operations/base/tovalue.h"
#include "tfcc_runtime/operations/base/tovector.h"
#include "tfcc_runtime/operations/base/transpose.h"
#include "tfcc_runtime/operations/base/tril.h"
#include "tfcc_runtime/operations/base/triu.h"
#include "tfcc_runtime/operations/base/unsqueeze.h"
#include "tfcc_runtime/operations/operation.h"

namespace tfcc {
namespace runtime {
namespace base {

static void append_vector(
    std::vector<std::unique_ptr<Operation>>& dst, std::vector<std::unique_ptr<Operation>> src) {
  for (auto& op : src) {
    dst.emplace_back(std::move(op));
  }
}

std::vector<std::unique_ptr<Operation>> get_all_operations() {
  std::vector<std::unique_ptr<Operation>> operations;
  append_vector(operations, get_at_operations());
  append_vector(operations, get_at1_operations());
  append_vector(operations, get_cast_operations());
  append_vector(operations, get_concat_operations());
  append_vector(operations, get_create_tensor_operations());
  append_vector(operations, get_create_vector_operations());
  append_vector(operations, get_expand_operations());
  append_vector(operations, get_flatten_operations());
  append_vector(operations, get_get_dimension_operations());
  append_vector(operations, get_get_shape_operations());
  append_vector(operations, get_identity_operations());
  append_vector(operations, get_if_operations());
  append_vector(operations, get_loop_operations());
  append_vector(operations, get_pad_operations());
  append_vector(operations, get_range_operations());
  append_vector(operations, get_reshape_operations());
  append_vector(operations, get_slice_v1_operations());
  append_vector(operations, get_slice_v2_operations());
  append_vector(operations, get_split_operations());
  append_vector(operations, get_squeeze_operations());
  append_vector(operations, get_stack_operations());
  append_vector(operations, get_tile_operations());
  append_vector(operations, get_to_tensor_operations());
  append_vector(operations, get_to_vector_operations());
  append_vector(operations, get_to_value_operations());
  append_vector(operations, get_transpose_operations());
  append_vector(operations, get_tril_operations());
  append_vector(operations, get_triu_operations());
  append_vector(operations, get_unsqueeze_operations());
  return operations;
}

}  // namespace base
}  // namespace runtime
}  // namespace tfcc
