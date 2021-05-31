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

#include "framework/tfcc_types.h"
#include "interfaces/tfcc_arithmeticinterface.h"

namespace tfcc {

template <class T>
class MKLArithmeticInterface : public ArithmeticInterface<T> {
 public:
  Variable<T> add(const Tensor<T>& a, const Tensor<T>& b) override;
  Variable<T> sub(const Tensor<T>& a, const Tensor<T>& b) override;
  Variable<T> mul(const Tensor<T>& a, const Tensor<T>& b) override;
  Variable<T> div(const Tensor<T>& a, const Tensor<T>& b) override;
};

template <class T>
class MKLArithmeticInterface<Complex<T>> : public ArithmeticInterface<Complex<T>> {
 public:
  Variable<Complex<T>> mul(const Tensor<Complex<T>>& a, const Tensor<Complex<T>>& b) override;
};

}  // namespace tfcc
