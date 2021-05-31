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
namespace nn {

template <class T>
tfcc::Variable<int64_t> non_zero(const Tensor<T>& input);

template <class T>
tfcc::Variable<T> batch_normalization(
    const tfcc::Tensor<T>& a, const tfcc::Tensor<T>& scale, const tfcc::Tensor<T>& offset,
    const tfcc::Tensor<T>& mean, const tfcc::Tensor<T>& var, T epsilon);

template <class T>
tfcc::Variable<T> one_hot(const Tensor<int64_t>& indices, unsigned depth, T offValue, T onValue);

template <class T>
tfcc::Variable<T> one_hot(const Tensor<int32_t>& indices, unsigned depth, T offValue, T onValue);

}  // namespace nn
}  // namespace helper
}  // namespace tfcc
