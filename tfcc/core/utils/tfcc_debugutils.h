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

#include <ostream>
#include <string>
#include <vector>

#include "framework/tfcc_shape.h"
#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"

namespace tfcc {

template <class T>
std::string debug_string(const Tensor<T>& a, size_t maxDumpLen);

template <class T>
std::string debug_string(const Tensor<T>& a) {
  return debug_string(a, 6);
}

std::string debug_string(const Shape& s);

template <class T>
bool is_similar(const Tensor<T>& a, const Tensor<T>& b);

template <class T>
std::ostream& operator<<(std::ostream& stream, const Tensor<T>& a) {
  stream << debug_string(a);
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const Shape& s);

}  // namespace tfcc
