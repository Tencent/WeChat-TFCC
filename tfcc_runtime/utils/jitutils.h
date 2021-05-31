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

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tfcc {
namespace Xbyak {
class CodeGenerator;
}
}  // namespace tfcc

namespace tfcc {
namespace runtime {
namespace jit {

inline constexpr size_t args_sum() { return 0; }

template <class T, class... Args>
inline constexpr size_t args_sum(T v, Args... values) {
  return v + args_sum(values...);
}

inline float convert_argument(float value) { return value; }

inline double convert_argument(double value) { return value; }

template <class T>
inline uint64_t convert_argument(T value) {
  union ValueWrapper {
    uint64_t uint64Value;
    T value;
  };
  static_assert(
      std::is_arithmetic<T>::value || std::is_pointer<T>::value,
      "Parameters must be arithmetic or pointer");
  static_assert(sizeof(value) <= sizeof(uint64_t), "Invalid paramters");
  static_assert(sizeof(ValueWrapper) <= sizeof(uint64_t), "Invalid paramters");
  ValueWrapper result;
  result.uint64Value = 0;
  result.value = value;
  return result.uint64Value;
}

void call_function_inner(Xbyak::CodeGenerator& jit, uintptr_t func, size_t& pos, size_t& fpos);
void set_argument_inner(Xbyak::CodeGenerator& jit, size_t& pos, float value);
void set_argument_inner(Xbyak::CodeGenerator& jit, size_t& pos, double value);
void set_argument_inner(Xbyak::CodeGenerator& jit, size_t& pos, uint64_t value);
void mov_rax_to_rdi(Xbyak::CodeGenerator& jit);

template <class T, class... Args>
inline void call_function_inner(
    Xbyak::CodeGenerator& jit, uintptr_t func, size_t& pos, size_t& fpos, T value,
    Args... arguments) {
  if (std::is_floating_point<T>::value) {
    set_argument_inner(jit, fpos, convert_argument(value));
  } else {
    set_argument_inner(jit, pos, convert_argument(value));
  }
  call_function_inner(jit, func, pos, fpos, arguments...);
}

template <class Func, class... Args>
inline void call_function(Xbyak::CodeGenerator& jit, Func func, Args... arguments) {
  static_assert(
      args_sum(std::is_floating_point<Args>::value...) < 8,
      "Floating point argument count must less than 8");
  static_assert(sizeof...(arguments) < 16, "Arguments count must less then 16");
  using RetType = typename std::result_of<Func(Args...)>::type;
  static_assert(
      std::is_same<RetType, const char*>::value || std::is_void<RetType>::value,
      "The result of func must be void or const char*");
  size_t pos = 0;
  size_t fpos = 0;
  call_function_inner(jit, reinterpret_cast<uintptr_t>(func), pos, fpos, arguments...);
}

template <class Func, class... Args>
inline void call_function_with_rax_as_1st_string_param(
    Xbyak::CodeGenerator& jit, Func func, Args... arguments) {
  static_assert(
      args_sum(std::is_floating_point<Args>::value...) < 8,
      "Floating point argument count must less than 8");
  static_assert(sizeof...(arguments) + 1 < 16, "Arguments count must less then 15");
  using RetType = typename std::result_of<Func(const char*, Args...)>::type;
  static_assert(
      std::is_same<RetType, const char*>::value || std::is_void<RetType>::value,
      "The result of func must be void or const char*");
  size_t pos = 1;
  size_t fpos = 0;
  mov_rax_to_rdi(jit);
  call_function_inner(jit, reinterpret_cast<uintptr_t>(func), pos, fpos, arguments...);
}

}  // namespace jit
}  // namespace runtime
}  // namespace tfcc
