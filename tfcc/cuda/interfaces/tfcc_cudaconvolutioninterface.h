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

#include "framework/tfcc_cudadeviceproperty.h"
#include "interfaces/tfcc_convolutioninterface.h"

namespace tfcc {

template <class T>
class CUDAConvolutionInterface : public ConvolutionInterface<T> {
  CUDADeviceProperty _property;

 public:
  explicit CUDAConvolutionInterface(const CUDADeviceProperty& property);
  ~CUDAConvolutionInterface();

  Variable<T> conv2d(
      const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned paddingHeight,
      unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth) override;

  Variable<T> conv2d(
      const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned paddingHeight,
      unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth, unsigned dilateHeight,
      unsigned dilateWidth) override;

  Variable<T> conv2dBackwardData(
      const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned paddingHeight,
      unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth) override;

  Variable<T> maxPool2d(
      const Tensor<T>& input, bool nhwc, unsigned kernelHeight, unsigned kernelWidth,
      unsigned paddingHeight, unsigned paddingWidth, unsigned strideHeight,
      unsigned strideWidth) override;

  Variable<T> avgPool2d(
      const Tensor<T>& input, bool nhwc, unsigned kernelHeight, unsigned kernelWidth,
      unsigned paddingHeight, unsigned paddingWidth, unsigned strideHeight,
      unsigned strideWidth) override;

 private:
  Variable<T> conv2dBackwardDataNCHW(
      const Tensor<T>& input, const Tensor<T>& kernel, unsigned paddingHeight,
      unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth);

  Variable<T> nhwc2nchw(const Tensor<T>& a);
  Variable<T> nchw2nhwc(const Tensor<T>& a);
};

}  // namespace tfcc
