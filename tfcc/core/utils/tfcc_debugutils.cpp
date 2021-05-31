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

#include "tfcc_debugutils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

#include "framework/tfcc_types.h"
#include "operations/tfcc_data.h"

namespace tfcc {

template <class T>
typename std::enable_if<std::is_integral<T>::value, std::string>::type data_to_string(T v) {
  return std::to_string(v);
}

template <class T>
typename std::enable_if<std::is_floating_point<T>::value, std::string>::type data_to_string(T v) {
  return std::to_string(v);
}

template <class T>
std::string debug_string(const Tensor<T>& a, size_t maxDumpLen) {
  typedef typename TypeInfo<T>::HighPrecisionType HType;

  std::string str = TypeInfo<T>::name;
  str += " (";
  for (size_t i = 0; i < a.shape().size(); ++i) {
    str += std::to_string(a.shape(i)) + ",";
  }
  str += ") [ ";
  if (a.shape().size() == 0) {
    return str + "]";
  }

  size_t len = std::min(static_cast<size_t>(a.size()), maxDumpLen);
  auto rv = data::get(a);
  for (size_t i = 0; i < len; ++i) {
    str += data_to_string(rv[i]) + " ";
  }
  if (len < a.shape().area()) {
    str += "... ";
  }
  str += "]";
  HType sumVal = HType();
  T maxVal = std::numeric_limits<T>::lowest();
  T minVal = std::numeric_limits<T>::max();
  for (auto v : rv) {
    sumVal += v;
    maxVal = std::max(maxVal, v);
    minVal = std::min(minVal, v);
  }
  str += " sum: " + data_to_string(sumVal);
  HType avgVal = sumVal / static_cast<HType>(a.size());
  str += " avg: " + data_to_string(avgVal);
  HType variance = HType();
  for (auto v : rv) {
    if (v > avgVal) {
      variance += (v - avgVal) * (v - avgVal);
    } else {
      variance += (avgVal - v) * (avgVal - v);
    }
  }
  variance /= static_cast<HType>(a.size());
  str += " var: " + data_to_string(variance);
  str += " max: " + data_to_string(maxVal);
  str += " min: " + data_to_string(minVal);
  return str;
}

std::string debug_string(const Shape& s) {
  std::string str("(");
  for (size_t i = 0; i < s.size(); ++i) {
    str += std::to_string(s[i]) + ",";
  }
  str += ")";
  return str;
}

std::ostream& operator<<(std::ostream& stream, const Shape& s) {
  stream << debug_string(s);
  return stream;
}

template <class T>
typename std::enable_if<std::is_integral<T>::value, bool>::type data_is_similar(T a, T b) {
  return a == b;
}

template <class T>
typename std::enable_if<std::is_floating_point<T>::value, bool>::type data_is_similar(T a, T b) {
  // return std::fabs(a - b) < std::numeric_limits<T>::epsilon() * std::fabs(a + b) * 4
  return std::fabs(a - b) < 0.001 * std::fabs(a + b) || std::fabs(a - b) < 0.00001;
}

template <class T>
bool is_similar(const Tensor<T>& a, const Tensor<T>& b) {
  if (a.shape() != b.shape()) {
    return false;
  }
  auto av = data::get(a);
  auto bv = data::get(b);
  assert(av.size() == bv.size());
  for (size_t i = 0; i < av.size(); ++i) {
    if (!data_is_similar(av[i], bv[i])) {
      return false;
    }
  }
  return true;
}

#define DEFINE_FUNC(type)                                                      \
  template std::string debug_string(const Tensor<type>& a, size_t maxDumpLen); \
  template bool is_similar(const Tensor<type>& a, const Tensor<type>& b);

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
