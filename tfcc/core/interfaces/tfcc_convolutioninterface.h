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
class ConvolutionInterface {
 public:
  ConvolutionInterface() {}
  ConvolutionInterface(const ConvolutionInterface&) = delete;
  virtual ~ConvolutionInterface() {}

  ConvolutionInterface& operator=(const ConvolutionInterface&) = delete;

  /**
   * kernel: [out_channels, in_channels, height, width]
   */
  virtual Variable<T> conv2d(
      const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned paddingHeight,
      unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth);

  /**
   * kernel: [out_channels, in_channels, height, width]
   */
  virtual Variable<T> conv2d(
      const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned paddingHeight,
      unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth, unsigned dilateHeight,
      unsigned dilateWidth);

  /**
   * kernel: [out_channels, in_channels, height, width]
   */
  virtual Variable<T> conv2d(
      const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned paddingHeight,
      unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth, unsigned dilateHeight,
      unsigned dilateWidth, unsigned group);

  /**
   * kernel: [in_channels, out_channels, height, width]
   */
  virtual Variable<T> conv2dBackwardData(
      const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned paddingHeight,
      unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth);

  virtual Variable<T> maxPool2d(
      const Tensor<T>& input, bool nhwc, unsigned kernelHeight, unsigned kernelWidth,
      unsigned paddingHeight, unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth);

  virtual Variable<T> avgPool2d(
      const Tensor<T>& input, bool nhwc, unsigned kernelHeight, unsigned kernelWidth,
      unsigned paddingHeight, unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth);
};

}  // namespace tfcc
