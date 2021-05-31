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

#include "tfcc_mkltransformationinterface.h"

#include <algorithm>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_types.h"
#include "framework/tfcc_view.h"
#include "interfaces/tfcc_mklinterfacehelper.h"
#include "kernel/tfcc_mkltransformationkernel.avx256.h"
#include "kernel/tfcc_mkltransformationkernel.avx512.h"
#include "kernel/tfcc_mkltransformationkernel.hpp"

namespace tfcc {

TFCC_MKL_HELPER_PRE_DEFINE(transform);
TFCC_MKL_HELPER_PRE_DEFINE(transform2);
TFCC_MKL_HELPER_PRE_DEFINE(transform3);
TFCC_MKL_HELPER_PRE_DEFINE(transform4);
TFCC_MKL_HELPER_PRE_DEFINE(transform5);
TFCC_MKL_HELPER_PRE_DEFINE(transform6);

template <class T>
Variable<T> MKLTransformationInterface<T>::transform(const Tensor<T>& a, T alpha, T beta) {
  Variable<T> result(a.shape());

  mkl_async_auto_switch_wrapper(
      "transform", TFCC_MKL_GET_RUNNER_HELPER(_MKLTransformationKernel, T, transform)(), a.data(),
      a.size(), alpha, beta, result.data());

  return result;
}

template <class T>
Variable<T> MKLTransformationInterface<T>::transform2(const Tensor<T>& a, T alpha, T beta) {
  Variable<T> result(a.shape());

  mkl_async_auto_switch_wrapper(
      "transform2", TFCC_MKL_GET_RUNNER_HELPER(_MKLTransformationKernel, T, transform2)(), a.data(),
      a.size(), alpha, beta, result.data());

  return result;
}

template <class T>
Variable<T> MKLTransformationInterface<T>::transform3(const Tensor<T>& a, T alpha, T beta) {
  Variable<T> result(a.shape());

  mkl_async_auto_switch_wrapper(
      "transform3", TFCC_MKL_GET_RUNNER_HELPER(_MKLTransformationKernel, T, transform3)(), a.data(),
      a.size(), alpha, beta, result.data());

  return result;
}

template <class T>
Variable<T> MKLTransformationInterface<T>::transform4(const Tensor<T>& a, T alpha, T beta) {
  Variable<T> result(a.shape());

  mkl_async_auto_switch_wrapper(
      "transform4", TFCC_MKL_GET_RUNNER_HELPER(_MKLTransformationKernel, T, transform4)(), a.data(),
      a.size(), alpha, beta, result.data());

  return result;
}

template <class T>
Variable<T> MKLTransformationInterface<T>::transform5(const Tensor<T>& a, T alpha, T beta) {
  Variable<T> result(a.shape());

  mkl_async_auto_switch_wrapper(
      "transform5", TFCC_MKL_GET_RUNNER_HELPER(_MKLTransformationKernel, T, transform5)(), a.data(),
      a.size(), alpha, beta, result.data());

  return result;
}

template <class T>
Variable<T> MKLTransformationInterface<T>::transform6(const Tensor<T>& a, T alpha, T beta) {
  Variable<T> result(a.shape());

  mkl_async_auto_switch_wrapper(
      "transform6", TFCC_MKL_GET_RUNNER_HELPER(_MKLTransformationKernel, T, transform6)(), a.data(),
      a.size(), alpha, beta, result.data());

  return result;
}

#define DEFINE_FUNC(type) template class MKLTransformationInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
