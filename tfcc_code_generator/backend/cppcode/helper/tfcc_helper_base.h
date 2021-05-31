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

#include <cstdint>
#include <vector>
#include "tfcc.h"

namespace tfcc {
namespace helper {
namespace base {

template <class T, class LEN>
tfcc::View<T> reshape(const tfcc::Tensor<T>& a, const std::vector<LEN>& shape);

template <class T, class LEN>
tfcc::Variable<T> expand(const tfcc::Tensor<T>& a, const std::vector<LEN>& shape);

template <class T, class LEN>
tfcc::Variable<T> create_tensor(T a, const std::vector<LEN>& shape);

template <class T>
tfcc::Variable<T> slice(const tfcc::Tensor<T>& a, size_t axis, int64_t start, int64_t end);

template <class T, class IDX>
tfcc::Variable<T> tile(const tfcc::Tensor<T>& a, const std::vector<IDX>& repeated);

template <class T>
std::vector<T> range(T start, T limit, T delta);

template <class T>
tfcc::Variable<T> eye(unsigned m, unsigned n, int64_t k);

}  // namespace base
}  // namespace helper
}  // namespace tfcc
