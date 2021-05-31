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

#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"

namespace tfcc {

template <class T>
class BatchArithmeticInterface {
 public:
  BatchArithmeticInterface() {}
  BatchArithmeticInterface(const BatchArithmeticInterface&) = delete;
  virtual ~BatchArithmeticInterface() {}

  BatchArithmeticInterface& operator=(const BatchArithmeticInterface&) = delete;

  /**
   * return values[0] + values[1] + ....
   * Attention: All values must have same shape.
   */
  virtual Variable<T> add(const std::vector<const Tensor<T>*>& values);

  /**
   * return a * b
   * Attention: see add
   */
  virtual Variable<T> mul(const std::vector<const Tensor<T>*>& values);
};

}  // namespace tfcc
