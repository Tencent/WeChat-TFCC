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

#include "tfcc_convolutioninterface.h"

#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_types.h"

namespace tfcc {

template <class T>
Variable<T> ConvolutionInterface<T>::conv2d(
    const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned paddingHeight,
    unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ConvolutionInterface<T>::conv2d(
    const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned paddingHeight,
    unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth, unsigned dilateHeight,
    unsigned dilateWidth) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ConvolutionInterface<T>::conv2d(
    const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned paddingHeight,
    unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth, unsigned dilateHeight,
    unsigned dilateWidth, unsigned group) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ConvolutionInterface<T>::conv2dBackwardData(
    const Tensor<T>& input, bool nhwc, const Tensor<T>& kernel, unsigned paddingHeight,
    unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ConvolutionInterface<T>::maxPool2d(
    const Tensor<T>& input, bool nhwc, unsigned kernelHeight, unsigned kernelWidth,
    unsigned paddingHeight, unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth) {
  throw NotImplementedError();
}

template <class T>
Variable<T> ConvolutionInterface<T>::avgPool2d(
    const Tensor<T>& input, bool nhwc, unsigned kernelHeight, unsigned kernelWidth,
    unsigned paddingHeight, unsigned paddingWidth, unsigned strideHeight, unsigned strideWidth) {
  throw NotImplementedError();
}

#define DEFINE_FUNC(type) template class ConvolutionInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);
TFCC_FOR_COMPLEX_TYPES(DEFINE_FUNC);

}  // namespace tfcc
