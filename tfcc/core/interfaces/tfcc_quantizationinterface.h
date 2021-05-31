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

#include <cstdint>
#include <tuple>

#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"

namespace tfcc {

template <class T>
class QuantizationInterface {
 public:
  QuantizationInterface() {}
  QuantizationInterface(const QuantizationInterface&) = delete;
  virtual ~QuantizationInterface() {}

  QuantizationInterface& operator=(const QuantizationInterface&) = delete;

  /**
   * Quantize a tensor.
   * @param a A tensor.
   * @return A tuple of quantized variable, min value variable and max value variable.
   */
  virtual std::tuple<Variable<T>, Variable<float>, Variable<float>> quantize(
      const Tensor<float>& a);

  /**
   * Quantize a tensor.
   * @param a A tensor.
   * @param minValue A tensor with shape [1,].
   * @param maxValue A tensor with shape [1,].
   * @return Quantized variable.
   */
  virtual Variable<T> quantize(
      const Tensor<float>& a, const Tensor<float>& minValue, const Tensor<float>& maxValue);

  /**
   * Requantize a tensor.
   * @param a A tensor.
   * @param minInput A tensor with shape [1,].
   * @param maxInput A tensor with shape [1,].
   * @param minOutput A tensor with shape [1,].
   * @param maxOutput A tensor with shape [1,].
   * @return Quantized variable.
   */
  virtual Variable<T> requantize(
      const Tensor<int8_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
      const Tensor<float>& minOutput, const Tensor<float>& maxOutput);

  /**
   * Requantize a tensor.
   * @param a A tensor.
   * @param minInput A tensor with shape [1,].
   * @param maxInput A tensor with shape [1,].
   * @param minOutput A tensor with shape [1,].
   * @param maxOutput A tensor with shape [1,].
   * @return Quantized variable.
   */
  virtual Variable<T> requantize(
      const Tensor<uint8_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
      const Tensor<float>& minOutput, const Tensor<float>& maxOutput);

  /**
   * Requantize a tensor.
   * @param a A tensor.
   * @param minInput A tensor with shape [1,].
   * @param maxInput A tensor with shape [1,].
   * @param minOutput A tensor with shape [1,].
   * @param maxOutput A tensor with shape [1,].
   * @return Quantized variable.
   */
  virtual Variable<T> requantize(
      const Tensor<int16_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
      const Tensor<float>& minOutput, const Tensor<float>& maxOutput);

  /**
   * Requantize a tensor.
   * @param a A tensor.
   * @param minInput A tensor with shape [1,].
   * @param maxInput A tensor with shape [1,].
   * @param minOutput A tensor with shape [1,].
   * @param maxOutput A tensor with shape [1,].
   * @return Quantized variable.
   */
  virtual Variable<T> requantize(
      const Tensor<uint16_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
      const Tensor<float>& minOutput, const Tensor<float>& maxOutput);

  /**
   * Requantize a tensor.
   * @param a A tensor.
   * @param minInput A tensor with shape [1,].
   * @param maxInput A tensor with shape [1,].
   * @param minOutput A tensor with shape [1,].
   * @param maxOutput A tensor with shape [1,].
   * @return Quantized variable.
   */
  virtual Variable<T> requantize(
      const Tensor<int32_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
      const Tensor<float>& minOutput, const Tensor<float>& maxOutput);

  /**
   * Requantize a tensor.
   * @param a A tensor.
   * @param minInput A tensor with shape [1,].
   * @param maxInput A tensor with shape [1,].
   * @param minOutput A tensor with shape [1,].
   * @param maxOutput A tensor with shape [1,].
   * @return Quantized variable.
   */
  virtual Variable<T> requantize(
      const Tensor<uint32_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
      const Tensor<float>& minOutput, const Tensor<float>& maxOutput);

  /**
   * Dequantize a tensor.
   * @param a A tensor.
   * @param minValue A tensor with shape [1,].
   * @param maxValue A tensor with shape [1,].
   * @return Dequantized variable.
   */
  virtual Variable<float> dequantize(
      const Tensor<T>& a, const Tensor<float>& minValue, const Tensor<float>& maxValue);
};

}  // namespace tfcc
