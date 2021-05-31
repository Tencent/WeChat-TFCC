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

#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"

namespace tfcc {

template <class T>
class GatherInterface {
 public:
  GatherInterface() {}
  GatherInterface(const GatherInterface&) = delete;
  virtual ~GatherInterface() {}

  GatherInterface& operator=(const GatherInterface&) = delete;

  /**
   * Gather slices from params axis 0 according to indices.
   * @param params The Tensor from which to gather values.
   * @param indices The index Tensor.
   * @return A Tensor.
   */
  virtual Variable<T> gather(const Tensor<T>& params, const Tensor<uint32_t>& indices);

  /**
   * @overload
   */
  virtual Variable<T> gather(const Tensor<T>& params, const Tensor<int32_t>& indices);

  /**
   * @overload
   */
  virtual Variable<T> gather(const Tensor<T>& params, const Tensor<uint64_t>& indices);

  /**
   * @overload
   */
  virtual Variable<T> gather(const Tensor<T>& params, const Tensor<int64_t>& indices);
};

}  // namespace tfcc
