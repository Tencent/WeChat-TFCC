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

#include <string>

#include "framework/tfcc_constant.h"
#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"

namespace tfcc {
namespace layer {

enum class Conv2DKernelFormat {
  HWIO,  // [height, width, inChannels, outChannels] (DEFAULT) tensorflow use this format
  OIHW,  // [outChannels, inChannels, height, width]
};

template <class T>
class Conv2D {
  bool _initialized;
  Constant<T>* _kernel;
  Constant<T>* _bias;
  unsigned _filters;
  unsigned _kernelHeight;
  unsigned _kernelWidth;
  unsigned _strideHeight;
  unsigned _strideWidth;
  bool _paddingSame;
  bool _channelsLast;
  bool _useBias;
  Conv2DKernelFormat _kernelFormat;
  std::string _name;

 public:
  /**
   * @see Conv2D(unsigned, unsigned, unsigned, unsigned, unsigned, bool, bool, bool,
   * Conv2DKernelFormat, std::string)
   */
  Conv2D(
      unsigned filters, unsigned kernelHeight, unsigned kernelWidth, unsigned strideHeight,
      unsigned strideWidth, bool paddingSame, bool channelsLast, bool useBias);

  /*
   * @see Conv2D(unsigned, unsigned, unsigned, unsigned, unsigned, bool, bool, bool,
   * Conv2DKernelFormat, std::string)
   */
  Conv2D(
      unsigned filters, unsigned kernelHeight, unsigned kernelWidth, unsigned strideHeight,
      unsigned strideWidth, bool paddingSame, bool channelsLast, bool useBias, std::string name);

  /**
   * @param filters The dimensionality of the output space. If zero, filters will be initialized by
   * auto.
   * @param kernelHeight Specifying the height of the 2D convolution window. If zero, kernelHeight
   * will be initialized by auto.
   * @param kernelWidth Specifying the length of the 2D convolution window. If zero, kernelWidth
   * will be initialized by auto.
   * @param strideHeight Specifying the height of the convolution stride length.
   * @param strideWidth Specifying the width of the convolution stride length.
   * @param paddingSame Whether the layer uses padding same or padding valid.
   * @param channelsLast Whether the inputs uses [batch, width, channels] format or [batch,
   * channels, width] format
   * @param useBias Whether the layer uses a bias.
   * @param kernelFormat Specifying the format of kernel. HWIO is the default format.
   * @param name Name of the layer. The default value is 'conv2d'.
   */
  Conv2D(
      unsigned filters, unsigned kernelHeight, unsigned kernelWidth, unsigned strideHeight,
      unsigned strideWidth, bool paddingSame, bool channelsLast, bool useBias,
      Conv2DKernelFormat kernelFormat, std::string name);

  Variable<T> operator()(const Tensor<T>& inputs);

 private:
  std::pair<Shape, std::vector<T>> biasInitialize(Shape shape, std::vector<T> data);
};

template <class T>
Variable<T> conv2d(
    const Tensor<T>& inputs, unsigned filters, unsigned kernelHeight, unsigned kernelWidth,
    unsigned strideHeight, unsigned strideWidth, bool paddingSame, bool channelsLast, bool useBias);

template <class T>
Variable<T> conv2d(
    const Tensor<T>& inputs, unsigned filters, unsigned kernelHeight, unsigned kernelWidth,
    unsigned strideHeight, unsigned strideWidth, bool paddingSame, bool channelsLast, bool useBias,
    std::string name);

template <class T>
Variable<T> conv2d(
    const Tensor<T>& inputs, unsigned filters, unsigned kernelHeight, unsigned kernelWidth,
    unsigned strideHeight, unsigned strideWidth, bool paddingSame, bool channelsLast, bool useBias,
    Conv2DKernelFormat kernelFormat, std::string name);

}  // namespace layer
}  // namespace tfcc
