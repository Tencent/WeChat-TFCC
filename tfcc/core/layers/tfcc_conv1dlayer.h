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

enum class Conv1DKernelFormat {
  WIO,  // [width, inChannels, outChannels] (DEFAULT) tensorflow use this format
  OIW,  // [outChannels, inChannels, width]
};

template <class T>
class Conv1D {
  bool _initialized;
  Constant<T>* _kernel;
  Constant<T>* _bias;
  unsigned _filters;
  unsigned _kernelSize;
  unsigned _stride;
  bool _paddingSame;
  bool _channelsLast;
  bool _useBias;
  Conv1DKernelFormat _kernelFormat;
  std::string _name;

 public:
  /**
   * @see Conv1D(unsigned filters, unsigned kernelSize, unsigned stride, bool paddingSame, bool
   * channelsLast, bool useBias, Conv1DKernelFormat kernelFormat, std::string name)
   */
  Conv1D(
      unsigned filters, unsigned kernelSize, unsigned stride, bool paddingSame, bool channelsLast,
      bool useBias);

  /**
   * @see Conv1D(unsigned filters, unsigned kernelSize, unsigned stride, bool paddingSame, bool
   * channelsLast, bool useBias, Conv1DKernelFormat kernelFormat, std::string name)
   */
  Conv1D(
      unsigned filters, unsigned kernelSize, unsigned stride, bool paddingSame, bool channelsLast,
      bool useBias, std::string name);

  /**
   * @param filters The dimensionality of the output space. If zero, filters will be initialized by
   * auto.
   * @param kernelSize Specifying the length of the 1D convolution window. If zero, kernelSize will
   * be initialized by auto.
   * @param stride Specifying the stride length of the convolution.
   * @param paddingSame Whether the layer uses padding same or padding valid.
   * @param channelsLast Whether the inputs uses [batch, width, channels] format or [batch,
   * channels, width] format
   * @param useBias Whether the layer uses a bias.
   * @param kernelFormat Specifying the format of kernel. WIO is the default format.
   * @param name Name of the layer. The default value is 'conv1d'.
   */
  Conv1D(
      unsigned filters, unsigned kernelSize, unsigned stride, bool paddingSame, bool channelsLast,
      bool useBias, Conv1DKernelFormat kernelFormat, std::string name);

  Variable<T> operator()(const Tensor<T>& inputs);

 private:
  std::pair<Shape, std::vector<T>> kernelInitialize(Shape shape, std::vector<T> data);
  std::pair<Shape, std::vector<T>> biasInitialize(Shape shape, std::vector<T> data);
};

template <class T>
Variable<T> conv1d(
    const Tensor<T>& inputs, unsigned filters, unsigned kernelSize, unsigned stride,
    bool paddingSame, bool channelsLast, bool useBias);

template <class T>
Variable<T> conv1d(
    const Tensor<T>& inputs, unsigned filters, unsigned kernelSize, unsigned stride,
    bool paddingSame, bool channelsLast, bool useBias, std::string name);

template <class T>
Variable<T> conv1d(
    const Tensor<T>& inputs, unsigned filters, unsigned kernelSize, unsigned stride,
    bool paddingSame, bool channelsLast, bool useBias, Conv1DKernelFormat kernelFormat,
    std::string name);

}  // namespace layer
}  // namespace tfcc
