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

#include "tfcc_helper_math.h"

#include "tfcc_helper_inner.h"

namespace tfcc {
namespace helper {
namespace math {

static inline std::vector<unsigned> _get_reduce_target_shape(
    const tfcc::Shape& shape, const std::vector<size_t>& axes) {
  std::vector<unsigned> targetShape = shape.toVector();
  for (size_t axis : axes) targetShape[axis] = 1;
  return targetShape;
}

static inline std::vector<size_t> _get_reduce_perm(
    const tfcc::Shape& shape, std::vector<size_t> axes) {
  std::sort(axes.begin(), axes.end());
  std::vector<bool> axesMask(shape.size());
  for (size_t axis : axes) axesMask[axis] = true;
  std::vector<size_t> perm;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (!axesMask[i]) perm.push_back(i);
  }
  for (size_t axis : axes) perm.push_back(axis);
  return perm;
}

template <class T>
tfcc::Variable<T> reduce_mean(const Tensor<T>& input, std::vector<size_t> axes) {
  std::vector<unsigned> targetShape = _get_reduce_target_shape(input.shape(), axes);
  std::vector<size_t> perm = _get_reduce_perm(input.shape(), axes);
  tfcc::Variable<T> result = tfcc::base::transpose(input, perm);
  result = tfcc::math::reduce_mean(result, perm.size() - axes.size());
  result.reshape(targetShape);
  return result;
}

template <class T>
tfcc::Variable<T> reduce_sum(const Tensor<T>& input, std::vector<size_t> axes) {
  std::vector<unsigned> targetShape = _get_reduce_target_shape(input.shape(), axes);
  std::vector<size_t> perm = _get_reduce_perm(input.shape(), axes);
  tfcc::Variable<T> result = tfcc::base::transpose(input, perm);
  result = tfcc::math::reduce_sum(result, perm.size() - axes.size());
  result.reshape(targetShape);
  return result;
}

template <class T>
tfcc::Variable<T> reduce_prod(const Tensor<T>& input, std::vector<size_t> axes) {
  std::vector<unsigned> targetShape = _get_reduce_target_shape(input.shape(), axes);
  std::vector<size_t> perm = _get_reduce_perm(input.shape(), axes);
  tfcc::Variable<T> result = tfcc::base::transpose(input, perm);
  result = tfcc::math::reduce_prod(result, perm.size() - axes.size());
  result.reshape(targetShape);
  return result;
}

template <class T>
tfcc::Variable<T> reduce_max(const Tensor<T>& input, std::vector<size_t> axes) {
  std::vector<unsigned> targetShape = _get_reduce_target_shape(input.shape(), axes);
  std::vector<size_t> perm = _get_reduce_perm(input.shape(), axes);
  tfcc::Variable<T> result = tfcc::base::transpose(input, perm);
  result = tfcc::math::reduce_max(result, perm.size() - axes.size());
  result.reshape(targetShape);
  return result;
}

template <class T>
tfcc::Variable<T> reduce_min(const Tensor<T>& input, std::vector<size_t> axes) {
  std::vector<unsigned> targetShape = _get_reduce_target_shape(input.shape(), axes);
  std::vector<size_t> perm = _get_reduce_perm(input.shape(), axes);
  tfcc::Variable<T> result = tfcc::base::transpose(input, perm);
  result = tfcc::math::reduce_min(result, perm.size() - axes.size());
  result.reshape(targetShape);
  return result;
}

template <class T>
std::tuple<tfcc::Variable<T>, tfcc::Variable<int64_t>> top_k(const Tensor<T>& a, unsigned k) {
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
      if (currentCount + 1 >= vs.size() && vec[i * chunk + j] <= vs[0].v) continue;
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

#define DEFINE_FUNC(type)                                                                         \
  template tfcc::Variable<type> reduce_mean(const Tensor<type>& input, std::vector<size_t> axes); \
  template tfcc::Variable<type> reduce_sum(const Tensor<type>& input, std::vector<size_t> axes);  \
  template tfcc::Variable<type> reduce_prod(const Tensor<type>& input, std::vector<size_t> axes); \
  template tfcc::Variable<type> reduce_max(const Tensor<type>& input, std::vector<size_t> axes);  \
  template tfcc::Variable<type> reduce_min(const Tensor<type>& input, std::vector<size_t> axes);  \
  template std::tuple<tfcc::Variable<type>, tfcc::Variable<int64_t>> top_k(                       \
      const Tensor<type>& a, unsigned k);

TFCC_HELPER_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace math
}  // namespace helper
}  // namespace tfcc
