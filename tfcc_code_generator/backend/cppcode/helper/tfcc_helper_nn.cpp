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

#include "tfcc_helper_nn.h"

#include "tfcc_mkl.h"

#include "tfcc_helper_inner.h"

namespace tfcc {
namespace helper {
namespace nn {

template <class T>
tfcc::Variable<int64_t> non_zero_dimn(const Tensor<T>& input) {
  std::vector<unsigned> skip = input.shape().toVector();
  for (size_t i = 1; i < skip.size(); ++i) skip[skip.size() - i - 1] *= skip[skip.size() - i];
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
tfcc::Variable<int64_t> non_zero(const Tensor<T>& input) {
  if (input.shape().size() != 1) return non_zero_dimn(input);
  std::vector<T> data = tfcc::data::get(input);
  std::vector<int64_t> result;
  for (unsigned i = 0; i < input.size(); ++i) {
    if (data[i] > 0 || data[i] < 0) result.push_back(i);
  }

  return tfcc::data::set(result, {static_cast<unsigned>(result.size()), 1u});
}

template <class T>
tfcc::Variable<T> batch_normalization(
    const tfcc::Tensor<T>& a, const tfcc::Tensor<T>& scale, const tfcc::Tensor<T>& offset,
    const tfcc::Tensor<T>& mean, const tfcc::Tensor<T>& var, T epsilon) {
  std::vector<unsigned> shape(a.shape().size(), 1);

  Variable<T> inv = math::rsqrt(var + epsilon);
  inv = inv * scale;
  shape[1] = inv.size();
  inv.reshape(shape);

  shape[1] = offset.size();
  tfcc::View<T> offsetView(offset, shape);

  shape[1] = mean.size();
  tfcc::View<T> meanView(mean, shape);

  return a * inv + offsetView - meanView * inv;
}

template <class T>
tfcc::Variable<T> one_hot(const Tensor<int64_t>& indices, unsigned depth, T offValue, T onValue) {
  std::vector<unsigned> ids;
  for (auto idx : tfcc::data::get(indices)) ids.push_back(static_cast<unsigned>(idx));
  return tfcc::nn::one_hot(ids, indices.shape(), depth, offValue, onValue);
}

template <class T>
tfcc::Variable<T> one_hot(const Tensor<int32_t>& indices, unsigned depth, T offValue, T onValue) {
  std::vector<unsigned> ids;
  for (auto idx : tfcc::data::get(indices)) ids.push_back(static_cast<unsigned>(idx));
  return tfcc::nn::one_hot(ids, indices.shape(), depth, offValue, onValue);
}

#define DEFINE_FUNC(type)                                                           \
  template tfcc::Variable<int64_t> non_zero(const Tensor<type>& input);             \
  template tfcc::Variable<type> batch_normalization(                                \
      const tfcc::Tensor<type>& a, const tfcc::Tensor<type>& scale,                 \
      const tfcc::Tensor<type>& offset, const tfcc::Tensor<type>& mean,             \
      const tfcc::Tensor<type>& var, type epsilon);                                 \
  template tfcc::Variable<type> one_hot(                                            \
      const Tensor<int64_t>& indices, unsigned depth, type offValue, type onValue); \
  template tfcc::Variable<type> one_hot(                                            \
      const Tensor<int32_t>& indices, unsigned depth, type offValue, type onValue);

TFCC_HELPER_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace nn
}  // namespace helper
}  // namespace tfcc
