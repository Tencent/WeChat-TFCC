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

#include "tfcc_mklsignalinterface.h"

#include <omp.h>
#include <algorithm>
#include <cstring>

#ifdef TFCC_USE_MKL
#  include "mkl_dfti.h"
#endif

#include "exceptions/tfcc_notimplementederror.h"
#include "exceptions/tfcc_runtimeerror.h"
#include "framework/tfcc_mklsession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "interfaces/tfcc_mklinterfacehelper.h"

namespace tfcc {

template <class T>
Variable<Complex<T>> MKLSignalInterface<T>::rfft(const Tensor<T>& a, unsigned length) {
  throw NotImplementedError();
}

template <class T>
Variable<T> MKLSignalInterface<T>::irfft(const Tensor<Complex<T>>& a, unsigned length) {
  throw NotImplementedError();
}

#ifdef TFCC_USE_MKL

static void _mkl_rfft_with_padding(
    DFTI_DESCRIPTOR_HANDLE hand, const float* a, unsigned batch, unsigned chunk, unsigned length,
    Complex<float>* result, unsigned resultChunk) {
  std::vector<float> buffer(length * omp_get_max_threads(), 0.f);
#  pragma omp parallel for
  for (unsigned i = 0; i < batch; ++i) {
    memcpy(
        buffer.data() + omp_get_thread_num() * length, a + i * chunk,
        std::min(chunk, length) * sizeof(float));
    DftiComputeForward(
        hand, buffer.data() + omp_get_thread_num() * length, result + i * resultChunk);
  }
  DftiFreeDescriptor(&hand);
}

static void _mkl_irfft_with_padding(
    DFTI_DESCRIPTOR_HANDLE hand, const Complex<float>* a, unsigned batch, unsigned chunk,
    unsigned length, float* result, unsigned resultChunk) {
  std::vector<Complex<float>> buffer(length * omp_get_max_threads(), {0.f, 0.f});
#  pragma omp parallel for
  for (unsigned i = 0; i < batch; ++i) {
    memcpy(
        buffer.data() + omp_get_thread_num() * length, a + i * chunk,
        std::min(chunk, length) * sizeof(Complex<float>));
    DftiComputeBackward(
        hand, buffer.data() + omp_get_thread_num() * length, result + i * resultChunk);
  }
  DftiFreeDescriptor(&hand);
}

template <>
Variable<Complex<float>> MKLSignalInterface<float>::rfft(const Tensor<float>& a, unsigned length) {
  std::vector<unsigned> shape = a.shape().toVector();
  shape[shape.size() - 1] = length / 2 + 1;
  Variable<Complex<float>> result(shape);

  DFTI_DESCRIPTOR_HANDLE hand = nullptr;
  MKL_LONG status = DftiCreateDescriptor(&hand, DFTI_SINGLE, DFTI_REAL, 1, length);
  if (status != DFTI_NO_ERROR) {
    DftiFreeDescriptor(&hand);
    throw RuntimeError("Create DFT desc error");
  }
  status = DftiSetValue(hand, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  if (status != DFTI_NO_ERROR) {
    DftiFreeDescriptor(&hand);
    throw RuntimeError("Set DFT properties error");
  }
  status = DftiSetValue(hand, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
  if (status != DFTI_NO_ERROR) {
    DftiFreeDescriptor(&hand);
    throw RuntimeError("Set DFT properties error");
  }
  status = DftiCommitDescriptor(hand);
  if (status != DFTI_NO_ERROR) {
    DftiFreeDescriptor(&hand);
    throw RuntimeError("Initialize DFT desc error");
  }

  mkl_async_wrapper(
      "rfft", _mkl_rfft_with_padding, hand, a.data(), a.size() / a.shape(a.shape().size() - 1),
      a.shape(a.shape().size() - 1), length, result.data(),
      result.shape(result.shape().size() - 1));
  return result;
}

template <>
Variable<float> MKLSignalInterface<float>::irfft(const Tensor<Complex<float>>& a, unsigned length) {
  std::vector<unsigned> shape = a.shape().toVector();
  shape[shape.size() - 1] = length;
  Variable<float> result(shape);

  DFTI_DESCRIPTOR_HANDLE hand = nullptr;
  MKL_LONG status = DftiCreateDescriptor(&hand, DFTI_SINGLE, DFTI_REAL, 1, length);
  if (status != DFTI_NO_ERROR) {
    DftiFreeDescriptor(&hand);
    throw RuntimeError("Create DFT desc error");
  }
  status = DftiSetValue(hand, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  if (status != DFTI_NO_ERROR) {
    DftiFreeDescriptor(&hand);
    throw RuntimeError("Set DFT properties error");
  }
  status = DftiSetValue(hand, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
  if (status != DFTI_NO_ERROR) {
    DftiFreeDescriptor(&hand);
    throw RuntimeError("Set DFT properties error");
  }
  status = DftiSetValue(hand, DFTI_BACKWARD_SCALE, (1.0f / static_cast<float>(length)));
  if (status != DFTI_NO_ERROR) {
    DftiFreeDescriptor(&hand);
    throw RuntimeError("Set DFT properties error");
  }
  status = DftiCommitDescriptor(hand);
  if (status != DFTI_NO_ERROR) {
    DftiFreeDescriptor(&hand);
    throw RuntimeError("Initialize DFT desc error");
  }

  mkl_async_wrapper(
      "irfft", _mkl_irfft_with_padding, hand, a.data(), a.size() / a.shape(a.shape().size() - 1),
      a.shape(a.shape().size() - 1), length / 2 + 1, result.data(),
      result.shape(result.shape().size() - 1));
  return result;
}

#endif

#define DEFINE_FUNC(type) template class MKLSignalInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
