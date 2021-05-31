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

#include "tfcc_quantization.h"

#include "exceptions/tfcc_invalidargumenterror.h"
#include "framework/tfcc_types.h"
#include "framework/tfcc_view.h"
#include "interfaces/tfcc_quantizationinterface.h"
#include "operations/tfcc_operation.h"

namespace tfcc {
namespace quantization {

template <class T>
std::tuple<Variable<T>, Variable<float>, Variable<float>> quantize(const Tensor<float>& a) {
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getQuantizationInterface().quantize(a);
}

template <class T>
Variable<T> quantize(
    const Tensor<float>& a, const Tensor<float>& minValue, const Tensor<float>& maxValue) {
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getQuantizationInterface().quantize(a, minValue, maxValue);
}

template <class T>
Variable<T> requantize(
    const Tensor<int8_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
    const Tensor<float>& minOutput, const Tensor<float>& maxOutput) {
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getQuantizationInterface().requantize(
      a, minInput, maxInput, minOutput, maxOutput);
}

template <class T>
Variable<T> requantize(
    const Tensor<uint8_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
    const Tensor<float>& minOutput, const Tensor<float>& maxOutput) {
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getQuantizationInterface().requantize(
      a, minInput, maxInput, minOutput, maxOutput);
}

template <class T>
Variable<T> requantize(
    const Tensor<int16_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
    const Tensor<float>& minOutput, const Tensor<float>& maxOutput) {
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getQuantizationInterface().requantize(
      a, minInput, maxInput, minOutput, maxOutput);
}

template <class T>
Variable<T> requantize(
    const Tensor<uint16_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
    const Tensor<float>& minOutput, const Tensor<float>& maxOutput) {
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getQuantizationInterface().requantize(
      a, minInput, maxInput, minOutput, maxOutput);
}

template <class T>
Variable<T> requantize(
    const Tensor<int32_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
    const Tensor<float>& minOutput, const Tensor<float>& maxOutput) {
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getQuantizationInterface().requantize(
      a, minInput, maxInput, minOutput, maxOutput);
}

template <class T>
Variable<T> requantize(
    const Tensor<uint32_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
    const Tensor<float>& minOutput, const Tensor<float>& maxOutput) {
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getQuantizationInterface().requantize(
      a, minInput, maxInput, minOutput, maxOutput);
}

template <class T>
Variable<float> dequantize(
    const Tensor<T>& a, const Tensor<float>& minValue, const Tensor<float>& maxValue) {
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getQuantizationInterface().dequantize(a, minValue, maxValue);
}

#define DEFINE_FUNC(type)                                                                      \
  template std::tuple<Variable<type>, Variable<float>, Variable<float>> quantize(              \
      const Tensor<float>& a);                                                                 \
  template Variable<type> quantize(                                                            \
      const Tensor<float>& a, const Tensor<float>& minValue, const Tensor<float>& maxValue);   \
  template Variable<type> requantize(                                                          \
      const Tensor<int8_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,   \
      const Tensor<float>& minOutput, const Tensor<float>& maxOutput);                         \
  template Variable<type> requantize(                                                          \
      const Tensor<uint8_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,  \
      const Tensor<float>& minOutput, const Tensor<float>& maxOutput);                         \
  template Variable<type> requantize(                                                          \
      const Tensor<int16_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,  \
      const Tensor<float>& minOutput, const Tensor<float>& maxOutput);                         \
  template Variable<type> requantize(                                                          \
      const Tensor<uint16_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput, \
      const Tensor<float>& minOutput, const Tensor<float>& maxOutput);                         \
  template Variable<type> requantize(                                                          \
      const Tensor<int32_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,  \
      const Tensor<float>& minOutput, const Tensor<float>& maxOutput);                         \
  template Variable<type> requantize(                                                          \
      const Tensor<uint32_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput, \
      const Tensor<float>& minOutput, const Tensor<float>& maxOutput);                         \
  template Variable<float> dequantize(                                                         \
      const Tensor<type>& a, const Tensor<float>& minValue, const Tensor<float>& maxValue);

TFCC_FOR_QUANTIZATION_TYPES(DEFINE_FUNC);

}  // namespace quantization
}  // namespace tfcc
