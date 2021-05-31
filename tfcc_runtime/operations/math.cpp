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

#include "math.h"

#include "tfcc_runtime/operations/math/abs.h"
#include "tfcc_runtime/operations/math/add.h"
#include "tfcc_runtime/operations/math/argmax.h"
#include "tfcc_runtime/operations/math/clip.h"
#include "tfcc_runtime/operations/math/div.h"
#include "tfcc_runtime/operations/math/erf.h"
#include "tfcc_runtime/operations/math/gelu.h"
#include "tfcc_runtime/operations/math/leakyrelu.h"
#include "tfcc_runtime/operations/math/log.h"
#include "tfcc_runtime/operations/math/matmul.h"
#include "tfcc_runtime/operations/math/matmulwithbias.h"
#include "tfcc_runtime/operations/math/max.h"
#include "tfcc_runtime/operations/math/min.h"
#include "tfcc_runtime/operations/math/mul.h"
#include "tfcc_runtime/operations/math/neg.h"
#include "tfcc_runtime/operations/math/pow.h"
#include "tfcc_runtime/operations/math/reciprocal.h"
#include "tfcc_runtime/operations/math/reducemax.h"
#include "tfcc_runtime/operations/math/reducemean.h"
#include "tfcc_runtime/operations/math/reducemin.h"
#include "tfcc_runtime/operations/math/reduceprod.h"
#include "tfcc_runtime/operations/math/reducesum.h"
#include "tfcc_runtime/operations/math/relu.h"
#include "tfcc_runtime/operations/math/rsqrt.h"
#include "tfcc_runtime/operations/math/sigmoid.h"
#include "tfcc_runtime/operations/math/sign.h"
#include "tfcc_runtime/operations/math/softmax.h"
#include "tfcc_runtime/operations/math/softplus.h"
#include "tfcc_runtime/operations/math/sqrt.h"
#include "tfcc_runtime/operations/math/sub.h"
#include "tfcc_runtime/operations/math/tanh.h"
#include "tfcc_runtime/operations/math/topk.h"
#include "tfcc_runtime/operations/operation.h"

namespace tfcc {
namespace runtime {
namespace math {

static void append_vector(
    std::vector<std::unique_ptr<Operation>>& dst, std::vector<std::unique_ptr<Operation>> src) {
  for (auto& op : src) {
    dst.emplace_back(std::move(op));
  }
}

std::vector<std::unique_ptr<Operation>> get_all_operations() {
  std::vector<std::unique_ptr<Operation>> operations;
  append_vector(operations, get_abs_operations());
  append_vector(operations, get_argmax_operations());
  append_vector(operations, get_add_operations());
  append_vector(operations, get_sub_operations());
  append_vector(operations, get_mul_operations());
  append_vector(operations, get_div_operations());
  append_vector(operations, get_clip_operations());
  append_vector(operations, get_erf_operations());
  append_vector(operations, get_gelu_operations());
  append_vector(operations, get_leaky_relu_operations());
  append_vector(operations, get_log_operations());
  append_vector(operations, get_max_operations());
  append_vector(operations, get_min_operations());
  append_vector(operations, get_neg_operations());
  append_vector(operations, get_pow_operations());
  append_vector(operations, get_reciprocal_operations());
  append_vector(operations, get_relu_operations());
  append_vector(operations, get_sigmoid_operations());
  append_vector(operations, get_sign_operations());
  append_vector(operations, get_softplus_operations());
  append_vector(operations, get_softmax_operations());
  append_vector(operations, get_sqrt_operations());
  append_vector(operations, get_tanh_operations());
  append_vector(operations, get_matmul_operations());
  append_vector(operations, get_matmul_with_bias_operations());
  append_vector(operations, get_reduce_mean_operations());
  append_vector(operations, get_reduce_sum_operations());
  append_vector(operations, get_reduce_prod_operations());
  append_vector(operations, get_reduce_max_operations());
  append_vector(operations, get_reduce_min_operations());
  append_vector(operations, get_rsqrt_operations());
  append_vector(operations, get_top_k_operations());
  return operations;
}

}  // namespace math
}  // namespace runtime
}  // namespace tfcc
