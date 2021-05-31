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

#include "tfcc_operator.h"

#include "framework/tfcc_types.h"

#include "tfcc_math.h"
#include "tfcc_relation.h"

namespace tfcc {

template <class T>
Variable<T> operator+(const Tensor<T>& a, const Tensor<T>& b) {
  return math::add(a, b);
}

template <class T>
Variable<T> operator-(const Tensor<T>& a, const Tensor<T>& b) {
  return math::sub(a, b);
}

template <class T>
Variable<T> operator*(const Tensor<T>& a, const Tensor<T>& b) {
  return math::mul(a, b);
}

template <class T>
Variable<T> operator/(const Tensor<T>& a, const Tensor<T>& b) {
  return math::div(a, b);
}

template <class T>
Variable<uint8_t> operator==(const Tensor<T>& a, const Tensor<T>& b) {
  return relation::equal(a, b);
}

template <class T>
Variable<uint8_t> operator!=(const Tensor<T>& a, const Tensor<T>& b) {
  return relation::unequal(a, b);
}

template <class T>
Variable<uint8_t> operator>(const Tensor<T>& a, const Tensor<T>& b) {
  return relation::greater(a, b);
}

template <class T>
Variable<uint8_t> operator>=(const Tensor<T>& a, const Tensor<T>& b) {
  return relation::greater_equal(a, b);
}

template <class T>
Variable<uint8_t> operator<(const Tensor<T>& a, const Tensor<T>& b) {
  return relation::less(a, b);
}

template <class T>
Variable<uint8_t> operator<=(const Tensor<T>& a, const Tensor<T>& b) {
  return relation::less_equal(a, b);
}

template <class T>
Variable<T> operator+(const Tensor<T>& a, T b) {
  return math::transform(a, static_cast<T>(1), b);
}

template <class T>
Variable<T> operator-(const Tensor<T>& a, T b) {
  return math::transform(a, static_cast<T>(1), static_cast<T>(-b));
}

template <class T>
Variable<T> operator*(const Tensor<T>& a, T b) {
  return math::transform(a, b, static_cast<T>(0));
}

template <class T>
Variable<T> operator/(const Tensor<T>& a, T b) {
  return math::transform2(a, b, static_cast<T>(0));
}

template <class T>
Variable<uint8_t> operator==(const Tensor<T>& a, T b) {
  return relation::equal(a, b);
}

template <class T>
Variable<uint8_t> operator!=(const Tensor<T>& a, T b) {
  return relation::unequal(a, b);
}

template <class T>
Variable<uint8_t> operator>(const Tensor<T>& a, T b) {
  return relation::greater(a, b);
}

template <class T>
Variable<uint8_t> operator>=(const Tensor<T>& a, T b) {
  return relation::greater_equal(a, b);
}

template <class T>
Variable<uint8_t> operator<(const Tensor<T>& a, T b) {
  return relation::less(a, b);
}

template <class T>
Variable<uint8_t> operator<=(const Tensor<T>& a, T b) {
  return relation::less_equal(a, b);
}

template <class T>
Variable<T> operator+(T a, const Tensor<T>& b) {
  return math::transform(b, static_cast<T>(1), a);
}

template <class T>
Variable<T> operator-(T a, const Tensor<T>& b) {
  return math::transform4(b, static_cast<T>(1), a);
}

template <class T>
Variable<T> operator*(T a, const Tensor<T>& b) {
  return math::transform(b, a, static_cast<T>(0));
}

template <class T>
Variable<T> operator/(T a, const Tensor<T>& b) {
  return math::transform3(b, a, static_cast<T>(0));
}

template <class T>
Variable<uint8_t> operator==(T a, const Tensor<T>& b) {
  return relation::equal(a, b);
}

template <class T>
Variable<uint8_t> operator!=(T a, const Tensor<T>& b) {
  return relation::unequal(a, b);
}

template <class T>
Variable<uint8_t> operator>(T a, const Tensor<T>& b) {
  return relation::greater(a, b);
}

template <class T>
Variable<uint8_t> operator>=(T a, const Tensor<T>& b) {
  return relation::greater_equal(a, b);
}

template <class T>
Variable<uint8_t> operator<(T a, const Tensor<T>& b) {
  return relation::less(a, b);
}

template <class T>
Variable<uint8_t> operator<=(T a, const Tensor<T>& b) {
  return relation::less_equal(a, b);
}

#define DEFINE_FUNC(type)                                                              \
  template Variable<type> operator+(const Tensor<type>& a, const Tensor<type>& b);     \
  template Variable<type> operator-(const Tensor<type>& a, const Tensor<type>& b);     \
  template Variable<type> operator*(const Tensor<type>& a, const Tensor<type>& b);     \
  template Variable<type> operator/(const Tensor<type>& a, const Tensor<type>& b);     \
  template Variable<uint8_t> operator==(const Tensor<type>& a, const Tensor<type>& b); \
  template Variable<uint8_t> operator!=(const Tensor<type>& a, const Tensor<type>& b); \
  template Variable<uint8_t> operator>(const Tensor<type>& a, const Tensor<type>& b);  \
  template Variable<uint8_t> operator>=(const Tensor<type>& a, const Tensor<type>& b); \
  template Variable<uint8_t> operator<(const Tensor<type>& a, const Tensor<type>& b);  \
  template Variable<uint8_t> operator<=(const Tensor<type>& a, const Tensor<type>& b); \
  template Variable<type> operator+(const Tensor<type>& a, type b);                    \
  template Variable<type> operator-(const Tensor<type>& a, type b);                    \
  template Variable<type> operator*(const Tensor<type>& a, type b);                    \
  template Variable<type> operator/(const Tensor<type>& a, type b);                    \
  template Variable<uint8_t> operator==(const Tensor<type>& a, type b);                \
  template Variable<uint8_t> operator!=(const Tensor<type>& a, type b);                \
  template Variable<uint8_t> operator>(const Tensor<type>& a, type b);                 \
  template Variable<uint8_t> operator>=(const Tensor<type>& a, type b);                \
  template Variable<uint8_t> operator<(const Tensor<type>& a, type b);                 \
  template Variable<uint8_t> operator<=(const Tensor<type>& a, type b);                \
  template Variable<type> operator+(type a, const Tensor<type>& b);                    \
  template Variable<type> operator-(type a, const Tensor<type>& b);                    \
  template Variable<type> operator*(type a, const Tensor<type>& b);                    \
  template Variable<type> operator/(type a, const Tensor<type>& b);                    \
  template Variable<uint8_t> operator==(type a, const Tensor<type>& b);                \
  template Variable<uint8_t> operator!=(type a, const Tensor<type>& b);                \
  template Variable<uint8_t> operator>(type a, const Tensor<type>& b);                 \
  template Variable<uint8_t> operator>=(type a, const Tensor<type>& b);                \
  template Variable<uint8_t> operator<(type a, const Tensor<type>& b);                 \
  template Variable<uint8_t> operator<=(type a, const Tensor<type>& b);

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

#define COMPLEX_DEFINE_FUNC(type) \
  template Variable<type> operator*(const Tensor<type>& a, const Tensor<type>& b);

TFCC_FOR_COMPLEX_TYPES(COMPLEX_DEFINE_FUNC);

}  // namespace tfcc
