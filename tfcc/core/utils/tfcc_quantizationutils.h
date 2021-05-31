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

#include <cmath>
#include <limits>
#include <tuple>

namespace tfcc {

template <class T>
inline float get_float_for_one_quantization_level(float rangeMin, float rangeMax) {
  constexpr int64_t highest = static_cast<int64_t>(std::numeric_limits<T>::max());
  constexpr int64_t lowest = static_cast<int64_t>(std::numeric_limits<T>::lowest());
  return (rangeMax - rangeMin) / (highest - lowest);
}

template <class T1, class T2, class TO>
inline std::tuple<float, float> get_quantization_range_for_multiplication(
    float minA, float maxA, float minB, float maxB) {
  float aLevel = get_float_for_one_quantization_level<T1>(minA, maxA);
  float bLevel = get_float_for_one_quantization_level<T2>(minB, maxB);

  constexpr int64_t cHighest = static_cast<int64_t>(std::numeric_limits<TO>::max());
  constexpr int64_t cLowest = static_cast<int64_t>(std::numeric_limits<TO>::lowest());

  float cLevel = aLevel * bLevel;
  float minC = cLevel * cLowest;
  float maxC = cLevel * cHighest;
  return std::make_tuple(minC, maxC);
}

template <class T>
inline std::tuple<double, int64_t> get_quantized_scale_info(float minValue, float maxValue) {
  constexpr int bits = sizeof(T) * 8;
  constexpr int64_t stepCnt = static_cast<int64_t>(1) << bits;

  double rangeAdjust = stepCnt / (stepCnt - 1.0);
  double range = (maxValue - minValue) * rangeAdjust;
  double rangeScale = stepCnt / range;
  int64_t offset = static_cast<int64_t>(round(minValue * rangeScale));

  return std::make_tuple(rangeScale, offset);
}

template <class T>
inline std::tuple<double, double> get_dequantized_scale_info(float minValue, float maxValue) {
  constexpr int bits = sizeof(T) * 8;
  constexpr int64_t stepCnt = static_cast<int64_t>(1) << bits;

  double rangeAdjust = stepCnt / (stepCnt - 1.0);
  double range = (maxValue - minValue) * rangeAdjust;
  double rangeScale = range / stepCnt;
  // double minRounded = round(minValue / rangeScale) * rangeScale;
  double minRounded = minValue;

  return std::make_tuple(rangeScale, minRounded);
}

template <class T>
inline T quantize_one(float a, double scale, int64_t offset) {
  int64_t quantized = static_cast<int64_t>(round(a * scale)) - offset;
  quantized += static_cast<int64_t>(std::numeric_limits<T>::lowest());
  quantized = std::max(quantized, static_cast<int64_t>(std::numeric_limits<T>::lowest()));
  quantized = std::min(quantized, static_cast<int64_t>(std::numeric_limits<T>::max()));
  return static_cast<T>(quantized);
}

}  // namespace tfcc
