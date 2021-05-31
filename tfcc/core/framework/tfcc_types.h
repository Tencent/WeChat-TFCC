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
#include <type_traits>

#define TFCC_FOR_ALL_TYPES(macro) \
  macro(float);                   \
  macro(double);                  \
  macro(int8_t);                  \
  macro(uint8_t);                 \
  macro(int16_t);                 \
  macro(uint16_t);                \
  macro(int32_t);                 \
  macro(uint32_t);                \
  macro(int64_t);                 \
  macro(uint64_t)

#define TFCC_FOR_FLOATING_POINT_TYPES(macro) \
  macro(float);                              \
  macro(double)

#define TFCC_FOR_QUANTIZATION_TYPES(macro) \
  macro(int8_t);                           \
  macro(uint8_t);                          \
  macro(int16_t);                          \
  macro(uint16_t);                         \
  macro(int32_t);                          \
  macro(uint32_t)

#define TFCC_FOR_COMPLEX_TYPES(macro) \
  macro(Complex<float>);              \
  macro(Complex<double>)

namespace tfcc {

template <class T>
struct Complex {
  T real;
  T imag;
};

template <class T>
struct TypeInfo {};

template <>
struct TypeInfo<float> {
  static constexpr const char* name = "float";
  static constexpr std::false_type quantizationType{};
  typedef double HighPrecisionType;
};

template <>
struct TypeInfo<double> {
  static constexpr const char* name = "double";
  static constexpr std::false_type quantizationType{};
  typedef double HighPrecisionType;
};

template <>
struct TypeInfo<int8_t> {
  static constexpr const char* name = "int8_t";
  static constexpr std::true_type quantizationType{};
  typedef int64_t HighPrecisionType;
};

template <>
struct TypeInfo<uint8_t> {
  static constexpr const char* name = "uint8";
  static constexpr std::true_type quantizationType{};
  typedef uint64_t HighPrecisionType;
};

template <>
struct TypeInfo<int16_t> {
  static constexpr const char* name = "int16";
  static constexpr std::true_type quantizationType{};
  typedef int64_t HighPrecisionType;
};

template <>
struct TypeInfo<uint16_t> {
  static constexpr const char* name = "uint16";
  static constexpr std::true_type quantizationType{};
  typedef uint64_t HighPrecisionType;
};

template <>
struct TypeInfo<int32_t> {
  static constexpr const char* name = "int32";
  static constexpr std::true_type quantizationType{};
  typedef int64_t HighPrecisionType;
};

template <>
struct TypeInfo<uint32_t> {
  static constexpr const char* name = "uint32";
  static constexpr std::true_type quantizationType{};
  typedef uint64_t HighPrecisionType;
};

template <>
struct TypeInfo<int64_t> {
  static constexpr const char* name = "int64";
  static constexpr std::false_type quantizationType{};
  typedef int64_t HighPrecisionType;
};

template <>
struct TypeInfo<uint64_t> {
  static constexpr const char* name = "uint64";
  static constexpr std::false_type quantizationType{};
  typedef uint64_t HighPrecisionType;
};

template <>
struct TypeInfo<Complex<float>> {
  static constexpr const char* name = "complex64";
  static constexpr std::false_type quantizationType{};
  typedef Complex<double> HighPrecisionType;
  using BaseType = float;
};

template <>
struct TypeInfo<Complex<double>> {
  static constexpr const char* name = "complex128";
  static constexpr std::false_type quantizationType{};
  typedef Complex<double> HighPrecisionType;
  using BaseType = double;
};

}  // namespace tfcc
