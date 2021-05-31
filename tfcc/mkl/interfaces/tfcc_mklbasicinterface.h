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

#pragma once

#include "interfaces/tfcc_basicinterface.h"

namespace tfcc {

template <class T>
class MKLBasicInterface : public BasicInterface<T> {
 public:
  Variable<T> slice(const Tensor<T>& a, size_t axis, unsigned start, unsigned end) override;
  void assignTo(const Tensor<T>& a, size_t axis, unsigned start, Variable<T>& b) override;
  Variable<T> transpose(const Tensor<T>& a, const std::vector<size_t>& perm) override;
  Variable<T> clip(const Tensor<T>& a, T minValue, T maxValue) override;
  Variable<T> concat(const std::vector<const Tensor<T>*>& values, size_t axis) override;
  Variable<T> where(
      const Tensor<uint8_t>& condition, const Tensor<T>& x, const Tensor<T>& y) override;
  Variable<T> where(const Tensor<uint8_t>& condition, T x, const Tensor<T>& y) override;
  Variable<T> where(const Tensor<uint8_t>& condition, const Tensor<T>& x, T y) override;
  Variable<T> abs(const Tensor<T>& a) override;
  Variable<T> tril(const Tensor<T>& a, int64_t k) override;
  Variable<T> triu(const Tensor<T>& a, int64_t k) override;
  std::tuple<Variable<T>, Variable<uint32_t>> topK(const Tensor<T>& a, unsigned k) override;
  Variable<int64_t> argmax(const Tensor<T>& a, size_t axis) override;
};

};  // namespace tfcc
