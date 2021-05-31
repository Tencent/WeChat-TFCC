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

#include "tfcc_mklarithmeticinterface.h"

#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_mklsession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "framework/tfcc_view.h"
#include "interfaces/tfcc_mklbroadcasthelper.h"
#include "interfaces/tfcc_mklinterfacehelper.h"
#include "kernel/tfcc_mklarithmetickernel.avx256.h"
#include "kernel/tfcc_mklarithmetickernel.avx512.h"
#include "kernel/tfcc_mklarithmetickernel.hpp"

namespace tfcc {

TFCC_MKL_HELPER_PRE_DEFINE(batchAdd);
TFCC_MKL_HELPER_PRE_DEFINE(batchSub);
TFCC_MKL_HELPER_PRE_DEFINE(batchMul);
TFCC_MKL_HELPER_PRE_DEFINE(batchDiv);

template <class T>
Variable<T> MKLArithmeticInterface<T>::add(const Tensor<T>& a, const Tensor<T>& b) {
  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  auto instruction = session->getCPUInstructionFlags();
  if (instruction & MKLInstruction::AVX512) {
    return _mkl_process_broadcast_op(
        a, b, [](T a, T b) { return a + b; },
        [](const T* a, const T* b, T* c, unsigned total) {
          return TFCC_MKL_GET_RUNNER_HELPER(_MKLArithmeticKernel, T, batchAdd)::runAVX512(
              a, b, c, total);
        },
        "add");
  }
  if (instruction & MKLInstruction::AVX256) {
    return _mkl_process_broadcast_op(
        a, b, [](T a, T b) { return a + b; },
        [](const T* a, const T* b, T* c, unsigned total) {
          return TFCC_MKL_GET_RUNNER_HELPER(_MKLArithmeticKernel, T, batchAdd)::runAVX256(
              a, b, c, total);
        },
        "add");
  } else {
    return _mkl_process_broadcast_op(
        a, b, [](T a, T b) { return a + b; },
        [](const T* a, const T* b, T* c, unsigned total) {
          return TFCC_MKL_GET_RUNNER_HELPER(_MKLArithmeticKernel, T, batchAdd)::runNormal(
              a, b, c, total);
        },
        "add");
  }
}

template <class T>
Variable<T> MKLArithmeticInterface<T>::sub(const Tensor<T>& a, const Tensor<T>& b) {
  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  auto instruction = session->getCPUInstructionFlags();
  if (instruction & MKLInstruction::AVX512) {
    return _mkl_process_broadcast_op(
        a, b, [](T a, T b) { return a - b; },
        [](const T* a, const T* b, T* c, unsigned total) {
          return TFCC_MKL_GET_RUNNER_HELPER(_MKLArithmeticKernel, T, batchSub)::runAVX512(
              a, b, c, total);
        },
        "sub");
  }
  if (instruction & MKLInstruction::AVX256) {
    return _mkl_process_broadcast_op(
        a, b, [](T a, T b) { return a - b; },
        [](const T* a, const T* b, T* c, unsigned total) {
          return TFCC_MKL_GET_RUNNER_HELPER(_MKLArithmeticKernel, T, batchSub)::runAVX256(
              a, b, c, total);
        },
        "sub");
  } else {
    return _mkl_process_broadcast_op(
        a, b, [](T a, T b) { return a - b; },
        [](const T* a, const T* b, T* c, unsigned total) {
          return TFCC_MKL_GET_RUNNER_HELPER(_MKLArithmeticKernel, T, batchSub)::runNormal(
              a, b, c, total);
        },
        "sub");
  }
}

template <class T>
Variable<T> MKLArithmeticInterface<T>::mul(const Tensor<T>& a, const Tensor<T>& b) {
  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  auto instruction = session->getCPUInstructionFlags();
  if (instruction & MKLInstruction::AVX512) {
    return _mkl_process_broadcast_op(
        a, b, [](T a, T b) { return a * b; },
        [](const T* a, const T* b, T* c, unsigned total) {
          return TFCC_MKL_GET_RUNNER_HELPER(_MKLArithmeticKernel, T, batchMul)::runAVX512(
              a, b, c, total);
        },
        "mul");
  }
  if (instruction & MKLInstruction::AVX256) {
    return _mkl_process_broadcast_op(
        a, b, [](T a, T b) { return a * b; },
        [](const T* a, const T* b, T* c, unsigned total) {
          return TFCC_MKL_GET_RUNNER_HELPER(_MKLArithmeticKernel, T, batchMul)::runAVX256(
              a, b, c, total);
        },
        "mul");
  } else {
    return _mkl_process_broadcast_op(
        a, b, [](T a, T b) { return a * b; },
        [](const T* a, const T* b, T* c, unsigned total) {
          return TFCC_MKL_GET_RUNNER_HELPER(_MKLArithmeticKernel, T, batchMul)::runNormal(
              a, b, c, total);
        },
        "mul");
  }
}

template <class T>
Variable<T> MKLArithmeticInterface<T>::div(const Tensor<T>& a, const Tensor<T>& b) {
  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  auto instruction = session->getCPUInstructionFlags();
  if (instruction & MKLInstruction::AVX512) {
    return _mkl_process_broadcast_op(
        a, b, [](T a, T b) { return a / b; },
        [](const T* a, const T* b, T* c, unsigned total) {
          return TFCC_MKL_GET_RUNNER_HELPER(_MKLArithmeticKernel, T, batchDiv)::runAVX512(
              a, b, c, total);
        },
        "div");
  }
  if (instruction & MKLInstruction::AVX256) {
    return _mkl_process_broadcast_op(
        a, b, [](T a, T b) { return a / b; },
        [](const T* a, const T* b, T* c, unsigned total) {
          return TFCC_MKL_GET_RUNNER_HELPER(_MKLArithmeticKernel, T, batchDiv)::runAVX256(
              a, b, c, total);
        },
        "div");
  } else {
    return _mkl_process_broadcast_op(
        a, b, [](T a, T b) { return a / b; },
        [](const T* a, const T* b, T* c, unsigned total) {
          return TFCC_MKL_GET_RUNNER_HELPER(_MKLArithmeticKernel, T, batchDiv)::runNormal(
              a, b, c, total);
        },
        "div");
  }
}

// complex
template <class T>
Variable<Complex<T>> MKLArithmeticInterface<Complex<T>>::mul(
    const Tensor<Complex<T>>& a, const Tensor<Complex<T>>& b) {
  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  auto instruction = session->getCPUInstructionFlags();
  if (instruction & MKLInstruction::AVX512) {
    return _mkl_process_broadcast_op(
        a, b,
        [](Complex<T> a, Complex<T> b) {
          Complex<T> result;
          result.real = a.real * b.real - a.imag * b.imag;
          result.imag = a.real * b.imag + a.imag * b.real;
          return result;
        },
        [](const Complex<T>* a, const Complex<T>* b, Complex<T>* c, unsigned total) {
          return TFCC_MKL_GET_RUNNER_HELPER(_MKLArithmeticKernel, Complex<T>, batchMul)::runAVX512(
              a, b, c, total);
        },
        "mul");
  }
  if (instruction & MKLInstruction::AVX256) {
    return _mkl_process_broadcast_op(
        a, b,
        [](Complex<T> a, Complex<T> b) {
          Complex<T> result;
          result.real = a.real * b.real - a.imag * b.imag;
          result.imag = a.real * b.imag + a.imag * b.real;
          return result;
        },
        [](const Complex<T>* a, const Complex<T>* b, Complex<T>* c, unsigned total) {
          return TFCC_MKL_GET_RUNNER_HELPER(_MKLArithmeticKernel, Complex<T>, batchMul)::runAVX256(
              a, b, c, total);
        },
        "mul");
  } else {
    return _mkl_process_broadcast_op(
        a, b,
        [](Complex<T> a, Complex<T> b) {
          Complex<T> result;
          result.real = a.real * b.real - a.imag * b.imag;
          result.imag = a.real * b.imag + a.imag * b.real;
          return result;
        },
        [](const Complex<T>* a, const Complex<T>* b, Complex<T>* c, unsigned total) {
          return TFCC_MKL_GET_RUNNER_HELPER(_MKLArithmeticKernel, Complex<T>, batchMul)::runNormal(
              a, b, c, total);
        },
        "mul");
  }
}

#define DEFINE_FUNC(type) template class MKLArithmeticInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);
TFCC_FOR_COMPLEX_TYPES(DEFINE_FUNC);

}  // namespace tfcc
