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

#include "interfaces/tfcc_reduceinterface.h"

namespace tfcc {

template <class T>
class MKLReduceInterface : public ReduceInterface<T> {
 public:
  Variable<T> reduceSum(const Tensor<T>& a, size_t keep) override;
  Variable<T> reduceMean(const Tensor<T>& a, size_t keep) override;
  Variable<T> reduceProd(const Tensor<T>& a, size_t keep) override;
  Variable<T> reduceMax(const Tensor<T>& a, size_t keep) override;
  Variable<T> reduceMin(const Tensor<T>& a, size_t keep) override;
  Variable<T> reduceAny(const Tensor<T>& a, size_t keep) override;
};

}  // namespace tfcc
