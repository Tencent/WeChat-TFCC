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

#include "tfcc_mklactivationinterface.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <set>
#include <type_traits>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_mklsession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "interfaces/tfcc_mklinterfacehelper.h"
#include "kernel/tfcc_mklactivationkernel.avx256.h"
#include "kernel/tfcc_mklactivationkernel.avx512.h"
#include "kernel/tfcc_mklactivationkernel.hpp"

namespace tfcc {

TFCC_MKL_HELPER_PRE_DEFINE(sigmoid);
TFCC_MKL_HELPER_PRE_DEFINE(tanh);
TFCC_MKL_HELPER_PRE_DEFINE(relu);
TFCC_MKL_HELPER_PRE_DEFINE(leakyRelu);
TFCC_MKL_HELPER_PRE_DEFINE(softplus);
TFCC_MKL_HELPER_PRE_DEFINE(log);
TFCC_MKL_HELPER_PRE_DEFINE(rsqrt);
TFCC_MKL_HELPER_PRE_DEFINE(softmax);
TFCC_MKL_HELPER_PRE_DEFINE(softmaxV2);
TFCC_MKL_HELPER_PRE_DEFINE(sin);
TFCC_MKL_HELPER_PRE_DEFINE(cos);
TFCC_MKL_HELPER_PRE_DEFINE(pow);
TFCC_MKL_HELPER_PRE_DEFINE(powV2);
TFCC_MKL_HELPER_PRE_DEFINE(powV3);
TFCC_MKL_HELPER_PRE_DEFINE(gelu);
TFCC_MKL_HELPER_PRE_DEFINE(geluAccurate);
TFCC_MKL_HELPER_PRE_DEFINE(erf);
TFCC_MKL_HELPER_PRE_DEFINE(asin);
TFCC_MKL_HELPER_PRE_DEFINE(asinh);
TFCC_MKL_HELPER_PRE_DEFINE(acos);
TFCC_MKL_HELPER_PRE_DEFINE(acosh);
TFCC_MKL_HELPER_PRE_DEFINE(atan);
TFCC_MKL_HELPER_PRE_DEFINE(atanh);
TFCC_MKL_HELPER_PRE_DEFINE(sign);

// class function
template <class T>
Variable<T> MKLActivationInterface<T>::sigmoid(const Tensor<T>& a) {
  Variable<T> result(a.shape());

  mkl_async_auto_switch_wrapper(
      "sigmoid", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, sigmoid)(), a.data(), a.size(),
      result.data());
  return result;
}

template <class T>
Variable<T> MKLActivationInterface<T>::tanh(const Tensor<T>& a) {
  Variable<T> result(a.shape());
  mkl_async_auto_switch_wrapper(
      "tanh", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, tanh)(), a.data(), a.size(),
      result.data());
  return result;
}

template <class T>
Variable<T> MKLActivationInterface<T>::relu(const Tensor<T>& a) {
  Variable<T> result(a.shape());
  mkl_async_auto_switch_wrapper(
      "relu", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, relu)(), a.data(), a.size(),
      result.data());
  return result;
}

template <class T>
Variable<T> MKLActivationInterface<T>::leakyRelu(const Tensor<T>& a, T alpha) {
  Variable<T> result(a.shape());
  mkl_async_auto_switch_wrapper(
      "leaky_relu", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, leakyRelu)(), a.data(),
      a.size(), alpha, result.data());
  return result;
}

template <class T>
Variable<T> MKLActivationInterface<T>::softplus(const Tensor<T>& a) {
  Variable<T> result(a.shape());
  mkl_async_auto_switch_wrapper(
      "softplus", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, softplus)(), a.data(),
      a.size(), result.data());
  return result;
}

template <class T>
Variable<T> MKLActivationInterface<T>::log(const Tensor<T>& a) {
  Variable<T> result(a.shape());
  mkl_async_auto_switch_wrapper(
      "log", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, log)(), a.data(), a.size(),
      result.data());
  return result;
}

template <class T>
Variable<T> MKLActivationInterface<T>::rsqrt(const Tensor<T>& a) {
  Variable<T> result(a.shape());
  mkl_async_auto_switch_wrapper(
      "rsqrt", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, rsqrt)(), a.data(), a.size(),
      result.data());
  return result;
}

template <class T>
Variable<T> MKLActivationInterface<T>::softmax(const Tensor<T>& a, size_t axis) {
  Variable<T> result(a.shape());
  unsigned s1 = 1, s2 = 1, s3 = 1;
  s2 = a.shape(axis);
  for (size_t i = 0; i < axis; ++i) {
    s1 *= a.shape(i);
  }
  for (size_t i = axis + 1; i < a.shape().size(); ++i) {
    s3 *= a.shape(i);
  }

  if (s3 > 1) {
    mkl_async_auto_switch_wrapper(
        "softmax", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, softmax)(), a.data(), s1, s2,
        s3, result.data());
  } else {
    mkl_async_auto_switch_wrapper(
        "softmax", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, softmaxV2)(), a.data(), s1,
        s2, result.data());
  }
  return result;
}

template <class T>
Variable<T> MKLActivationInterface<T>::sin(const Tensor<T>& a) {
  Variable<T> result(a.shape());
  mkl_async_auto_switch_wrapper(
      "sin", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, sin)(), a.data(), a.size(),
      result.data());
  return result;
}

template <class T>
Variable<T> MKLActivationInterface<T>::cos(const Tensor<T>& a) {
  Variable<T> result(a.shape());
  mkl_async_auto_switch_wrapper(
      "cos", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, cos)(), a.data(), a.size(),
      result.data());
  return result;
}

template <class T>
Variable<T> MKLActivationInterface<T>::pow(const Tensor<T>& a, T exponent) {
  Variable<T> result(a.shape());
  mkl_async_auto_switch_wrapper(
      "pow", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, pow)(), a.data(), a.size(),
      exponent, result.data());
  return result;
}

template <class T>
Variable<T> MKLActivationInterface<T>::pow(const Tensor<T>& a, const Tensor<T>& exponent) {
  Variable<T> result(a.shape());
  mkl_async_auto_switch_wrapper(
      "pow_v2", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, powV2)(), a.data(),
      exponent.data(), result.data(), a.size());
  return result;
}

template <class T>
Variable<T> MKLActivationInterface<T>::pow(T a, const Tensor<T>& exponent) {
  Variable<T> result(exponent.shape());
  mkl_async_auto_switch_wrapper(
      "pow_v3", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, powV3)(), exponent.data(),
      exponent.size(), a, result.data());
  return result;
}

template <class T>
Variable<T> MKLActivationInterface<T>::gelu(const Tensor<T>& a) {
  Variable<T> result(a.shape());
  mkl_async_auto_switch_wrapper(
      "gelu", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, gelu)(), a.data(), a.size(),
      result.data());
  return result;
}

template <class T>
Variable<T> MKLActivationInterface<T>::geluAccurate(const Tensor<T>& a) {
  Variable<T> result(a.shape());
  mkl_async_auto_switch_wrapper(
      "gelu_accurate", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, geluAccurate)(),
      a.data(), a.size(), result.data());
  return result;
}

template <class T>
Variable<T> MKLActivationInterface<T>::erf(const Tensor<T>& a) {
  Variable<T> result(a.shape());
  mkl_async_auto_switch_wrapper(
      "erf", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, erf)(), a.data(), a.size(),
      result.data());
  return result;
}

template <class T>
Variable<T> MKLActivationInterface<T>::asin(const Tensor<T>& a) {
  Variable<T> result(a.shape());
  mkl_async_auto_switch_wrapper(
      "asin", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, asin)(), a.data(), a.size(),
      result.data());
  return result;
}

template <class T>
Variable<T> MKLActivationInterface<T>::asinh(const Tensor<T>& a) {
  Variable<T> result(a.shape());
  mkl_async_auto_switch_wrapper(
      "asinh", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, asinh)(), a.data(), a.size(),
      result.data());
  return result;
}

template <class T>
Variable<T> MKLActivationInterface<T>::acos(const Tensor<T>& a) {
  Variable<T> result(a.shape());
  mkl_async_auto_switch_wrapper(
      "acos", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, acos)(), a.data(), a.size(),
      result.data());
  return result;
}

template <class T>
Variable<T> MKLActivationInterface<T>::acosh(const Tensor<T>& a) {
  Variable<T> result(a.shape());
  mkl_async_auto_switch_wrapper(
      "acosh", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, acosh)(), a.data(), a.size(),
      result.data());
  return result;
}

template <class T>
Variable<T> MKLActivationInterface<T>::atan(const Tensor<T>& a) {
  Variable<T> result(a.shape());
  mkl_async_auto_switch_wrapper(
      "atan", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, atan)(), a.data(), a.size(),
      result.data());
  return result;
}

template <class T>
Variable<T> MKLActivationInterface<T>::atanh(const Tensor<T>& a) {
  Variable<T> result(a.shape());
  mkl_async_auto_switch_wrapper(
      "atanh", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, atanh)(), a.data(), a.size(),
      result.data());
  return result;
}

template <class T>
Variable<T> MKLActivationInterface<T>::sign(const Tensor<T>& a) {
  Variable<T> result(a.shape());
  mkl_async_auto_switch_wrapper(
      "sign", TFCC_MKL_GET_RUNNER_HELPER(_MKLActivationKernel, T, sign)(), a.data(), a.size(),
      result.data());
  return result;
}

#define DEFINE_FUNC(type) template class MKLActivationInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
