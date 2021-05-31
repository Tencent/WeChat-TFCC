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

#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"

namespace tfcc {

template <class T>
class NormalizationInterface {
 public:
  NormalizationInterface() {}
  NormalizationInterface(const NormalizationInterface&) = delete;
  virtual ~NormalizationInterface() {}

  NormalizationInterface& operator=(const NormalizationInterface&) = delete;

  /**
   * @param a A Tensor.
   * @param gamma A Tensor with shape `a.shape[beginNormAxis:]`.
   * @param beta A Tensor with shape `a.shape[beginNormAxis:]`.
   * @param epsilon A value.
   * @param beginNormAxis Axis of params begin in a.
   * @return A Tensor with shape as same as a.
   */
  virtual Variable<T> layerNormalize(
      const Tensor<T>& a, const Tensor<T>& gamma, const Tensor<T>& beta, T epsilon,
      size_t beginNormAxis);

  /**
   * Local Response Normalization proposed in the AlexNet paper.
   * It normalizes over local input regions.
   * The local region is defined across the channels.
   * For an element X[n, c, d] in a tensor of shape (N x C x D),
   * its region is {X[n, i, d] | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size
   * - 1) / 2))}. square_sum[n, c, d] = sum(X[n, i, d] ^ 2), where max(0, c - floor((size - 1) / 2))
   * <= i <= min(C - 1, c + ceil((size - 1) / 2)). Y[n, c, d] = X[n, c, d] / (bias + alpha / size *
   * square_sum[n, c, d] ) ^ beta
   */
  virtual Variable<T> localResponseNormalization(
      const Tensor<T>& a, T alpha, T beta, T bias, unsigned size);

  virtual Variable<T> batchNormalization(
      const Tensor<T>& a, size_t axis, const Tensor<T>& scale, const Tensor<T>& offset,
      const Tensor<T>& mean, const Tensor<T>& var, T epsilon);
};

}  // namespace tfcc
