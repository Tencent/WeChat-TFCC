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

#include <vector>
#include "tfcc.h"

namespace tfcc {
namespace helper {
namespace math {

template <class T>
tfcc::Variable<T> reduce_mean(const Tensor<T>& input, std::vector<size_t> axes);

template <class T>
tfcc::Variable<T> reduce_sum(const Tensor<T>& input, std::vector<size_t> axes);

template <class T>
tfcc::Variable<T> reduce_prod(const Tensor<T>& input, std::vector<size_t> axes);

template <class T>
tfcc::Variable<T> reduce_max(const Tensor<T>& input, std::vector<size_t> axes);

template <class T>
tfcc::Variable<T> reduce_min(const Tensor<T>& input, std::vector<size_t> axes);

template <class T>
std::tuple<tfcc::Variable<T>, tfcc::Variable<int64_t>> top_k(const Tensor<T>& a, unsigned k);

}  // namespace math
}  // namespace helper
}  // namespace tfcc
