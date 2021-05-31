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

#include <functional>
#include <utility>

#include "framework/tfcc_mklinstruction.h"
#include "framework/tfcc_mklsession.h"

namespace tfcc {

#ifdef TFCC_COMPILER_IS_SUPPORT_PPLC
template <class Func, class... Args>
inline void mkl_async_wrapper(std::string name, Func func, Args... args) {
  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  session->addTask([func, args...]() { func(args...); }, std::move(name));
}

template <class Runner, class... Args>
inline void mkl_async_auto_switch_wrapper(std::string name, Runner runner, Args... args) {
  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  auto instruction = session->getCPUInstructionFlags();
  if (instruction & MKLInstruction::AVX512) {
    MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
    session->addTask([args...]() { Runner::runAVX512(args...); }, std::move(name));
  } else if (instruction & MKLInstruction::AVX256) {
    MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
    session->addTask([args...]() { Runner::runAVX256(args...); }, std::move(name));
  } else {
    MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
    session->addTask([args...]() { Runner::runNormal(args...); }, std::move(name));
  }
}
#else
template <size_t... Ints>
struct IntegerSequence {};

template <size_t I, size_t... Ints>
struct IndexSequence : public IndexSequence<I - 1, I - 1, Ints...> {};

template <size_t... Ints>
struct IndexSequence<0, Ints...> {
  using type = IntegerSequence<Ints...>;
};

template <class Func, class Tuple, size_t... Ints>
auto _apply_impl(Func&& func, Tuple&& args, IntegerSequence<Ints...>)
    -> decltype(func(std::get<Ints>(std::forward<Tuple>(args))...)) {
  return func(std::get<Ints>(std::forward<Tuple>(args))...);
}

template <class Func, class Tuple>
auto apply(Func&& func, Tuple&& args) -> decltype(_apply_impl(
    std::forward<Func>(func), std::forward<Tuple>(args),
    typename IndexSequence<std::tuple_size<typename std::decay<Tuple>::type>::value>::type())) {
  return _apply_impl(
      std::forward<Func>(func), std::forward<Tuple>(args),
      typename IndexSequence<std::tuple_size<typename std::decay<Tuple>::type>::value>::type());
}

template <class Func, class... Args>
inline void mkl_async_wrapper(std::string name, Func func, Args... args) {
  auto data = std::make_tuple(args...);
  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  session->addTask([func, data]() { apply(func, data); }, std::move(name));
}

template <class Runner, class Tuple, size_t... Ints>
auto _apply_runner_run_normal_impl(Runner, Tuple&& args, IntegerSequence<Ints...>)
    -> decltype(Runner::runNormal(std::get<Ints>(std::forward<Tuple>(args))...)) {
  return Runner::runNormal(std::get<Ints>(std::forward<Tuple>(args))...);
}

template <class Runner, class Tuple, size_t... Ints>
auto _apply_runner_run_avx256_impl(Runner, Tuple&& args, IntegerSequence<Ints...>)
    -> decltype(Runner::runAVX256(std::get<Ints>(std::forward<Tuple>(args))...)) {
  return Runner::runAVX256(std::get<Ints>(std::forward<Tuple>(args))...);
}

template <class Runner, class Tuple, size_t... Ints>
auto _apply_runner_run_avx512_impl(Runner, Tuple&& args, IntegerSequence<Ints...>)
    -> decltype(Runner::runAVX512(std::get<Ints>(std::forward<Tuple>(args))...)) {
  return Runner::runAVX512(std::get<Ints>(std::forward<Tuple>(args))...);
}

template <class Runner, class... Args>
inline void mkl_async_auto_switch_wrapper(std::string name, Runner runner, Args... args) {
  auto data = std::make_tuple(args...);
  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  auto instruction = session->getCPUInstructionFlags();
  if (instruction & MKLInstruction::AVX512) {
    MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
    session->addTask(
        [runner, data]() {
          _apply_runner_run_avx512_impl(
              runner, data, typename IndexSequence<sizeof...(Args)>::type());
        },
        std::move(name));
  } else if (instruction & MKLInstruction::AVX256) {
    MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
    session->addTask(
        [runner, data]() {
          _apply_runner_run_avx256_impl(
              runner, data, typename IndexSequence<sizeof...(Args)>::type());
        },
        std::move(name));
  } else {
    MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
    session->addTask(
        [runner, data]() {
          _apply_runner_run_normal_impl(
              runner, data, typename IndexSequence<sizeof...(Args)>::type());
        },
        std::move(name));
  }
}
#endif

// auto switch helper
#define TFCC_MKL_FUNCTION_TRAITS_HELPER(FUNCTION_NAME)                           \
  template <class T>                                                             \
  class _CheckHelper_##FUNCTION_NAME {                                           \
   public:                                                                       \
    template <class U>                                                           \
    static constexpr auto check(int) -> decltype(&U::FUNCTION_NAME != nullptr) { \
      return true;                                                               \
    }                                                                            \
    template <class U>                                                           \
    static constexpr bool check(...) {                                           \
      return false;                                                              \
    }                                                                            \
    static constexpr bool value = check<T>(0);                                   \
  }

#define TFCC_MKL_FUNCTION_RUNNER(FUNCTION_NAME)                                       \
  template <class Normal, class AVX256, class AVX512, bool HasAVX256, bool hasAVX512> \
  class _RunHelper_##FUNCTION_NAME {};                                                \
  template <class Normal, class AVX256, class AVX512>                                 \
  class _RunHelper_##FUNCTION_NAME<Normal, AVX256, AVX512, false, false> {            \
   public:                                                                            \
    template <class... Args>                                                          \
    static auto runNormal(Args&&... args)                                             \
        -> decltype(Normal::FUNCTION_NAME(std::forward<Args>(args)...)) {             \
      return Normal::FUNCTION_NAME(std::forward<Args>(args)...);                      \
    }                                                                                 \
    template <class... Args>                                                          \
    static auto runAVX256(Args&&... args)                                             \
        -> decltype(Normal::FUNCTION_NAME(std::forward<Args>(args)...)) {             \
      return Normal::FUNCTION_NAME(std::forward<Args>(args)...);                      \
    }                                                                                 \
    template <class... Args>                                                          \
    static auto runAVX512(Args&&... args)                                             \
        -> decltype(Normal::FUNCTION_NAME(std::forward<Args>(args)...)) {             \
      return Normal::FUNCTION_NAME(std::forward<Args>(args)...);                      \
    }                                                                                 \
  };                                                                                  \
  template <class Normal, class AVX256, class AVX512>                                 \
  class _RunHelper_##FUNCTION_NAME<Normal, AVX256, AVX512, true, false> {             \
   public:                                                                            \
    template <class... Args>                                                          \
    static auto runNormal(Args&&... args)                                             \
        -> decltype(Normal::FUNCTION_NAME(std::forward<Args>(args)...)) {             \
      return Normal::FUNCTION_NAME(std::forward<Args>(args)...);                      \
    }                                                                                 \
    template <class... Args>                                                          \
    static auto runAVX256(Args&&... args)                                             \
        -> decltype(AVX256::FUNCTION_NAME(std::forward<Args>(args)...)) {             \
      return AVX256::FUNCTION_NAME(std::forward<Args>(args)...);                      \
    }                                                                                 \
    template <class... Args>                                                          \
    static auto runAVX512(Args&&... args)                                             \
        -> decltype(AVX256::FUNCTION_NAME(std::forward<Args>(args)...)) {             \
      return AVX256::FUNCTION_NAME(std::forward<Args>(args)...);                      \
    }                                                                                 \
  };                                                                                  \
  template <class Normal, class AVX256, class AVX512>                                 \
  class _RunHelper_##FUNCTION_NAME<Normal, AVX256, AVX512, false, true> {             \
   public:                                                                            \
    template <class... Args>                                                          \
    static auto runNormal(Args&&... args)                                             \
        -> decltype(Normal::FUNCTION_NAME(std::forward<Args>(args)...)) {             \
      return Normal::FUNCTION_NAME(std::forward<Args>(args)...);                      \
    }                                                                                 \
    template <class... Args>                                                          \
    static auto runAVX256(Args&&... args)                                             \
        -> decltype(Normal::FUNCTION_NAME(std::forward<Args>(args)...)) {             \
      return Normal::FUNCTION_NAME(std::forward<Args>(args)...);                      \
    }                                                                                 \
    template <class... Args>                                                          \
    static auto runAVX512(Args&&... args)                                             \
        -> decltype(AVX512::FUNCTION_NAME(std::forward<Args>(args)...)) {             \
      return AVX512::FUNCTION_NAME(std::forward<Args>(args)...);                      \
    }                                                                                 \
  };                                                                                  \
  template <class Normal, class AVX256, class AVX512>                                 \
  class _RunHelper_##FUNCTION_NAME<Normal, AVX256, AVX512, true, true> {              \
   public:                                                                            \
    template <class... Args>                                                          \
    static auto runNormal(Args&&... args)                                             \
        -> decltype(Normal::FUNCTION_NAME(std::forward<Args>(args)...)) {             \
      return Normal::FUNCTION_NAME(std::forward<Args>(args)...);                      \
    }                                                                                 \
    template <class... Args>                                                          \
    static auto runAVX256(Args&&... args)                                             \
        -> decltype(AVX256::FUNCTION_NAME(std::forward<Args>(args)...)) {             \
      return AVX256::FUNCTION_NAME(std::forward<Args>(args)...);                      \
    }                                                                                 \
    template <class... Args>                                                          \
    static auto runAVX512(Args&&... args)                                             \
        -> decltype(AVX512::FUNCTION_NAME(std::forward<Args>(args)...)) {             \
      return AVX512::FUNCTION_NAME(std::forward<Args>(args)...);                      \
    }                                                                                 \
  }

#define _TFCC_MKL_GET_RUNNER_HELPER_INNER(                                           \
    BASE_NAME, TMPLATE_SUFFIX, FUNCTION_NAME, AVX256_SUFFIX, AVX512_SUFFIX)          \
  _RunHelper_##FUNCTION_NAME<                                                        \
      BASE_NAME<TMPLATE_SUFFIX>, BASE_NAME##AVX256_SUFFIX<TMPLATE_SUFFIX>,           \
      BASE_NAME##AVX512_SUFFIX<TMPLATE_SUFFIX>,                                      \
      _CheckHelper_##FUNCTION_NAME<BASE_NAME##AVX256_SUFFIX<TMPLATE_SUFFIX>>::value, \
      _CheckHelper_##FUNCTION_NAME<BASE_NAME##AVX512_SUFFIX<TMPLATE_SUFFIX>>::value>

#define TFCC_MKL_GET_RUNNER_HELPER(BASE_NAME, TMPLATE_SUFFIX, FUNCTION_NAME) \
  _TFCC_MKL_GET_RUNNER_HELPER_INNER(BASE_NAME, TMPLATE_SUFFIX, FUNCTION_NAME, AVX256, AVX512)

#define _TFCC_MKL_GET_RUNNER_HELPER_INNER_V2(                                                 \
    BASE_NAME, TMPLATE_SUFFIX1, TMPLATE_SUFFIX2, FUNCTION_NAME, AVX256_SUFFIX, AVX512_SUFFIX) \
  _RunHelper_##FUNCTION_NAME<                                                                 \
      BASE_NAME<TMPLATE_SUFFIX1, TMPLATE_SUFFIX2>,                                            \
      BASE_NAME##AVX256_SUFFIX<TMPLATE_SUFFIX1, TMPLATE_SUFFIX2>,                             \
      BASE_NAME##AVX512_SUFFIX<TMPLATE_SUFFIX1, TMPLATE_SUFFIX2>,                             \
      _CheckHelper_##FUNCTION_NAME<                                                           \
          BASE_NAME##AVX256_SUFFIX<TMPLATE_SUFFIX1, TMPLATE_SUFFIX2>>::value,                 \
      _CheckHelper_##FUNCTION_NAME<                                                           \
          BASE_NAME##AVX512_SUFFIX<TMPLATE_SUFFIX1, TMPLATE_SUFFIX2>>::value>

#define TFCC_MKL_GET_RUNNER_HELPER_V2(BASE_NAME, TMPLATE_SUFFIX1, TMPLATE_SUFFIX2, FUNCTION_NAME) \
  _TFCC_MKL_GET_RUNNER_HELPER_INNER_V2(                                                           \
      BASE_NAME, TMPLATE_SUFFIX1, TMPLATE_SUFFIX2, FUNCTION_NAME, AVX256, AVX512)

#define TFCC_MKL_HELPER_PRE_DEFINE(FUNCTION_NAME) \
  TFCC_MKL_FUNCTION_TRAITS_HELPER(FUNCTION_NAME); \
  TFCC_MKL_FUNCTION_RUNNER(FUNCTION_NAME)

}  // namespace tfcc
