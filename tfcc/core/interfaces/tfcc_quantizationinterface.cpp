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

#include "tfcc_quantizationinterface.h"

#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_types.h"

namespace tfcc {

template <class T>
std::tuple<Variable<T>, Variable<float>, Variable<float>> QuantizationInterface<T>::quantize(
    const Tensor<float>& a) {
  throw NotImplementedError();
}

template <class T>
Variable<T> QuantizationInterface<T>::quantize(
    const Tensor<float>& a, const Tensor<float>& minValue, const Tensor<float>& maxValue) {
  throw NotImplementedError();
}

template <class T>
Variable<T> QuantizationInterface<T>::requantize(
    const Tensor<int8_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
    const Tensor<float>& minOutput, const Tensor<float>& maxOutput) {
  throw NotImplementedError();
}

template <class T>
Variable<T> QuantizationInterface<T>::requantize(
    const Tensor<uint8_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
    const Tensor<float>& minOutput, const Tensor<float>& maxOutput) {
  throw NotImplementedError();
}

template <class T>
Variable<T> QuantizationInterface<T>::requantize(
    const Tensor<int16_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
    const Tensor<float>& minOutput, const Tensor<float>& maxOutput) {
  throw NotImplementedError();
}

template <class T>
Variable<T> QuantizationInterface<T>::requantize(
    const Tensor<uint16_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
    const Tensor<float>& minOutput, const Tensor<float>& maxOutput) {
  throw NotImplementedError();
}

template <class T>
Variable<T> QuantizationInterface<T>::requantize(
    const Tensor<int32_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
    const Tensor<float>& minOutput, const Tensor<float>& maxOutput) {
  throw NotImplementedError();
}

template <class T>
Variable<T> QuantizationInterface<T>::requantize(
    const Tensor<uint32_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
    const Tensor<float>& minOutput, const Tensor<float>& maxOutput) {
  throw NotImplementedError();
}

template <class T>
Variable<float> QuantizationInterface<T>::dequantize(
    const Tensor<T>& a, const Tensor<float>& minValue, const Tensor<float>& maxValue) {
  throw NotImplementedError();
}

#define DEFINE_FUNC(type) template class QuantizationInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);
TFCC_FOR_COMPLEX_TYPES(DEFINE_FUNC);

}  // namespace tfcc
