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

#include "interfaces/tfcc_normalizationinterface.h"

namespace tfcc {

template <class T>
class MKLNormalizationInterface : public NormalizationInterface<T> {
 public:
  Variable<T> layerNormalize(
      const Tensor<T>& a, const Tensor<T>& gamma, const Tensor<T>& beta, T epsilon,
      size_t beginNormAxis) override;
  Variable<T> localResponseNormalization(
      const Tensor<T>& a, T alpha, T beta, T bias, unsigned size) override;
  Variable<T> batchNormalization(
      const Tensor<T>& a, size_t axis, const Tensor<T>& scale, const Tensor<T>& offset,
      const Tensor<T>& mean, const Tensor<T>& var, T epsilon) override;
};

}  // namespace tfcc
