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

#include <tuple>
#include <vector>

#include "tfcc.h"

namespace tfcc {
namespace runtime {

inline std::tuple<std::vector<size_t>, tfcc::Shape, bool> get_reduce_perm_and_target_shape(
    const tfcc::Shape& shape, const std::vector<size_t>& axes) {
  std::vector<unsigned> targetShape = shape.toVector();
  std::vector<bool> axesMask(shape.size());
  for (size_t axis : axes) {
    targetShape[axis] = 1;
    axesMask[axis] = true;
  }
  std::vector<size_t> perm;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (!axesMask[i]) {
      perm.push_back(i);
    }
  }
  for (size_t axis : axes) {
    perm.push_back(axis);
  }
  bool needTranspose = false;
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] != i) {
      needTranspose = true;
      break;
    }
  }
  return std::make_tuple(std::move(perm), tfcc::Shape(std::move(targetShape)), needTranspose);
}

}  // namespace runtime
}  // namespace tfcc
