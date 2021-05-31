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

#include "tfcc_helper_base.h"

#include <cassert>
#include <stdexcept>

#include "tfcc_helper_inner.h"

namespace tfcc {
namespace helper {
namespace base {

template <class T, class LEN>
tfcc::View<T> reshape(const tfcc::Tensor<T>& a, const std::vector<LEN>& shape) {
  std::vector<unsigned> realShape(shape.size());
  unsigned* p = nullptr;
  unsigned area = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] > 0) {
      realShape[i] = static_cast<unsigned>(shape[i]);
      area *= shape[i];
    } else if (shape[i] == 0) {
      realShape[i] = a.shape()[i];
      area *= a.shape()[i];
    } else {
      assert(p == nullptr);
      p = &realShape[i];
    }
  }
  if (p != nullptr) *p = a.size() / area;
  tfcc::View<T> result(a, realShape);
  return result;
}

template <class T, class LEN>
tfcc::Variable<T> expand(const tfcc::Tensor<T>& a, const std::vector<LEN>& shape) {
  if (shape.size() < a.shape().size()) throw std::runtime_error("invalid shape");
  std::vector<unsigned> realShape;
  for (size_t i = 0; i < shape.size() - a.shape().size(); ++i) realShape.push_back(1);
  for (size_t i = 0; i < a.shape().size(); ++i) realShape.push_back(a.shape(i));
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] >= 0) realShape[i] = static_cast<unsigned>(shape[i]);
  }
  tfcc::Variable<T> result(realShape);
  tfcc::data::zeros(result);
  return result + a;
}

template <class T, class LEN>
tfcc::Variable<T> create_tensor(T a, const std::vector<LEN>& shape) {
  std::vector<unsigned> realShape;
  for (LEN s : shape) realShape.push_back(s);
  tfcc::Variable<T> result(realShape);
  tfcc::data::zeros(result);
  return result + a;
}

template <class T>
tfcc::Variable<T> slice(const tfcc::Tensor<T>& a, size_t axis, int64_t start, int64_t end) {
  if (axis >= a.shape().size()) throw std::runtime_error("invalid axis");
  while (start < 0) start += static_cast<int64_t>(a.shape(axis));
  while (end < 0) end += static_cast<int64_t>(a.shape(axis));
  return tfcc::base::slice(a, axis, static_cast<unsigned>(start), static_cast<unsigned>(end));
}

template <class T, class IDX>
tfcc::Variable<T> tile(const tfcc::Tensor<T>& a, const std::vector<IDX>& repeated) {
  if (repeated.size() != a.shape().size()) throw std::runtime_error("repeated error");

  std::vector<unsigned> shape = a.shape().toVector();
  std::vector<unsigned> targetShape = shape;
  std::vector<unsigned> tempAShape;
  std::vector<unsigned> tempRepeats;
  for (size_t i = 0; i < repeated.size(); ++i) {
    IDX r = repeated[i];
    if (r <= 0) throw std::runtime_error("repeated error");
    targetShape[i] *= r;
    tempAShape.push_back(1);
    tempAShape.push_back(shape[i]);
    tempRepeats.push_back(r);
    tempRepeats.push_back(shape[i]);
  }
  tfcc::View<T> tempA(a, tempAShape);
  tfcc::Variable<T> repeatTensor(tempRepeats);
  tfcc::data::ones(repeatTensor);
  tfcc::Variable<T> result = tempA * repeatTensor;
  result.reshape(targetShape);
  return result;
}

template <class T>
std::vector<T> range(T start, T limit, T delta) {
  std::vector<T> result;
  for (T x = start; x < limit; x += delta) result.push_back(x);
  return result;
}

template <class T>
tfcc::Variable<T> eye(unsigned m, unsigned n, int64_t k) {
  std::vector<T> data(m * n);
  for (unsigned i = 0; i < m; ++i) {
    for (unsigned j = 0; j < n; ++j) {
      T value = static_cast<int64_t>(j) - static_cast<int64_t>(i) == k ? static_cast<T>(1)
                                                                       : static_cast<T>(0);
      data[i * n + j] = value;
    }
  }
  return tfcc::data::set(data, {m, n});
}

#define DEFINE_FUNC(type)                                                                  \
  template tfcc::View<type> reshape(                                                       \
      const tfcc::Tensor<type>& a, const std::vector<uint32_t>& shape);                    \
  template tfcc::View<type> reshape(                                                       \
      const tfcc::Tensor<type>& a, const std::vector<int32_t>& shape);                     \
  template tfcc::View<type> reshape(                                                       \
      const tfcc::Tensor<type>& a, const std::vector<uint64_t>& shape);                    \
  template tfcc::View<type> reshape(                                                       \
      const tfcc::Tensor<type>& a, const std::vector<int64_t>& shape);                     \
  template tfcc::Variable<type> expand(                                                    \
      const tfcc::Tensor<type>& a, const std::vector<uint32_t>& shape);                    \
  template tfcc::Variable<type> expand(                                                    \
      const tfcc::Tensor<type>& a, const std::vector<int32_t>& shape);                     \
  template tfcc::Variable<type> expand(                                                    \
      const tfcc::Tensor<type>& a, const std::vector<uint64_t>& shape);                    \
  template tfcc::Variable<type> expand(                                                    \
      const tfcc::Tensor<type>& a, const std::vector<int64_t>& shape);                     \
  template tfcc::Variable<type> create_tensor(type a, const std::vector<uint32_t>& shape); \
  template tfcc::Variable<type> create_tensor(type a, const std::vector<int32_t>& shape);  \
  template tfcc::Variable<type> create_tensor(type a, const std::vector<uint64_t>& shape); \
  template tfcc::Variable<type> create_tensor(type a, const std::vector<int64_t>& shape);  \
  template tfcc::Variable<type> slice(                                                     \
      const tfcc::Tensor<type>& a, size_t axis, int64_t start, int64_t end);               \
  template tfcc::Variable<type> tile(                                                      \
      const tfcc::Tensor<type>& a, const std::vector<int32_t>& repeated);                  \
  template tfcc::Variable<type> tile(                                                      \
      const tfcc::Tensor<type>& a, const std::vector<uint32_t>& repeated);                 \
  template tfcc::Variable<type> tile(                                                      \
      const tfcc::Tensor<type>& a, const std::vector<int64_t>& repeated);                  \
  template tfcc::Variable<type> tile(                                                      \
      const tfcc::Tensor<type>& a, const std::vector<uint64_t>& repeated);                 \
  template std::vector<type> range(type start, type limit, type delta);                    \
  template tfcc::Variable<type> eye(unsigned m, unsigned n, int64_t k);

TFCC_HELPER_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace base
}  // namespace helper
}  // namespace tfcc
