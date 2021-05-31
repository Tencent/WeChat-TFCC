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

#include "tfcc_mklquantizationinterface.h"

#include <omp.h>
#include <cmath>
#include <limits>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_mklsession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "interfaces/tfcc_mklinterfacehelper.h"
#include "utils/tfcc_quantizationutils.h"

#include "kernel/tfcc_mklquantizationkernel.avx256.h"
#include "kernel/tfcc_mklquantizationkernel.avx512.h"
#include "kernel/tfcc_mklquantizationkernel.hpp"

namespace tfcc {

TFCC_MKL_HELPER_PRE_DEFINE(quantize);
TFCC_MKL_HELPER_PRE_DEFINE(dequantize);
TFCC_MKL_HELPER_PRE_DEFINE(requantize);

template <class T>
static inline void _mkl_set_value(T value, unsigned total, T* result) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    result[i] = value;
  }
}

template <class T, class Func>
static inline std::tuple<float, float> _mkl_quantize(
    const float* a, unsigned total, T* result, Func func) {
  float minValue = std::numeric_limits<float>::max();
  float maxValue = std::numeric_limits<float>::lowest();
#pragma omp parallel for reduction(max : maxValue) reduction(min : minValue)
  for (unsigned i = 0; i < total; ++i) {
    minValue = std::min(a[i], minValue);
    maxValue = std::max(a[i], maxValue);
  }

  if (minValue == maxValue) {
    _mkl_set_value(std::numeric_limits<T>::lowest(), total, result);
    return std::make_tuple(minValue, maxValue);
  }

  double scale;
  int64_t offset;
  std::tie(scale, offset) = get_quantized_scale_info<T>(minValue, maxValue);

  func(a, total, scale, offset, result);
  return std::make_tuple(minValue, maxValue);
}

// helper
template <class TO, class TI>
Variable<TO> _requantize_helper(
    const Tensor<TI>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
    const Tensor<float>& minOutput, const Tensor<float>& maxOutput) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  if (minInput.size() != 1 || minInput.shape().size() != 1) {
    throw InvalidArgumentError("invalid minInput");
  }
  if (maxInput.size() != 1 || maxInput.shape().size() != 1) {
    throw InvalidArgumentError("invalid maxInput");
  }
  if (minOutput.size() != 1 || minOutput.shape().size() != 1) {
    throw InvalidArgumentError("invalid minOutput");
  }
  if (maxOutput.size() != 1 || maxOutput.shape().size() != 1) {
    throw InvalidArgumentError("invalid maxOutput");
  }

  std::function<void(const TI*, unsigned, double, double, double, int64_t, TO*)> func;
  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  auto instruction = session->getCPUInstructionFlags();
  if (instruction & MKLInstruction::AVX512) {
    func = [](const TI* a, unsigned total, double inputScale, double inputMinRounded,
              double outputScale, int64_t outputOffset, TO* result) {
      TFCC_MKL_GET_RUNNER_HELPER_V2(_MKLRequantizationKernel, TI, TO, requantize)::runAVX512(
          a, total, inputScale, inputMinRounded, outputScale, outputOffset, result);
    };
  } else if (instruction & MKLInstruction::AVX256) {
    func = [](const TI* a, unsigned total, double inputScale, double inputMinRounded,
              double outputScale, int64_t outputOffset, TO* result) {
      TFCC_MKL_GET_RUNNER_HELPER_V2(_MKLRequantizationKernel, TI, TO, requantize)::runAVX256(
          a, total, inputScale, inputMinRounded, outputScale, outputOffset, result);
    };
  } else {
    func = [](const TI* a, unsigned total, double inputScale, double inputMinRounded,
              double outputScale, int64_t outputOffset, TO* result) {
      TFCC_MKL_GET_RUNNER_HELPER_V2(_MKLRequantizationKernel, TI, TO, requantize)::runNormal(
          a, total, inputScale, inputMinRounded, outputScale, outputOffset, result);
    };
  }

  Variable<TO> result(a.shape());
  mkl_async_wrapper(
      "requantize",
      [func](
          const TI* a, unsigned total, const float* minInput, const float* maxInput,
          const float* minOutput, const float* maxOutput, TO* result) {
        if (*minOutput >= *maxOutput) {
          _mkl_set_value(std::numeric_limits<TO>::lowest(), total, result);
          return;
        }
        double inputScale, inputMinRounded;
        std::tie(inputScale, inputMinRounded) =
            get_dequantized_scale_info<TI>(*minInput, *maxInput);
        double outputScale;
        int64_t outputOffset;
        std::tie(outputScale, outputOffset) = get_quantized_scale_info<TO>(*minOutput, *maxOutput);

        if (*minInput >= *maxInput) {
          int64_t quantized = static_cast<int64_t>(round((*minInput) * outputScale)) - outputOffset;
          quantized += static_cast<int64_t>(std::numeric_limits<TO>::lowest());
          quantized = std::max(quantized, static_cast<int64_t>(std::numeric_limits<TO>::lowest()));
          quantized = std::min(quantized, static_cast<int64_t>(std::numeric_limits<TO>::max()));
          _mkl_set_value(static_cast<TO>(quantized), total, result);
          return;
        }

        func(a, total, inputScale, inputMinRounded, outputScale, outputOffset, result);
      },
      a.data(), a.size(), minInput.data(), maxInput.data(), minOutput.data(), maxOutput.data(),
      result.data());

  return result;
}

// class functions
template <class T>
std::tuple<Variable<T>, Variable<float>, Variable<float>> MKLQuantizationInterface<T>::quantize(
    const Tensor<float>& a) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Variable<T> result(a.shape());
  Variable<float> minValue({1});
  Variable<float> maxValue({1});

  std::function<void(const float*, unsigned, double, int64_t, T*)> func;
  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  auto instruction = session->getCPUInstructionFlags();
  if (instruction & MKLInstruction::AVX512) {
    func = [](const float* a, unsigned total, double scale, int64_t offset, T* result) {
      TFCC_MKL_GET_RUNNER_HELPER(_MKLQuantizationKernel, T, quantize)::runAVX512(
          a, total, scale, offset, result);
    };
  } else if (instruction & MKLInstruction::AVX256) {
    func = [](const float* a, unsigned total, double scale, int64_t offset, T* result) {
      TFCC_MKL_GET_RUNNER_HELPER(_MKLQuantizationKernel, T, quantize)::runAVX256(
          a, total, scale, offset, result);
    };
  } else {
    func = [](const float* a, unsigned total, double scale, int64_t offset, T* result) {
      TFCC_MKL_GET_RUNNER_HELPER(_MKLQuantizationKernel, T, quantize)::runNormal(
          a, total, scale, offset, result);
    };
  }

  mkl_async_wrapper(
      "quantize",
      [func](const float* a, unsigned total, T* result, float* minValue, float* maxValue) {
        std::tie(*minValue, *maxValue) = _mkl_quantize(a, total, result, func);
      },
      a.data(), a.size(), result.data(), minValue.data(), maxValue.data());

  return std::make_tuple(std::move(result), std::move(minValue), std::move(maxValue));
}

template <class T>
Variable<T> MKLQuantizationInterface<T>::quantize(
    const Tensor<float>& a, const Tensor<float>& minValue, const Tensor<float>& maxValue) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  if (minValue.size() != 1 || minValue.shape().size() != 1) {
    throw InvalidArgumentError("invalid minValue");
  }
  if (maxValue.size() != 1 || maxValue.shape().size() != 1) {
    throw InvalidArgumentError("invalid maxValue");
  }

  std::function<void(const float*, unsigned, double, int64_t, T*)> func;
  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  auto instruction = session->getCPUInstructionFlags();
  if (instruction & MKLInstruction::AVX512) {
    func = [](const float* a, unsigned total, double scale, int64_t offset, T* result) {
      TFCC_MKL_GET_RUNNER_HELPER(_MKLQuantizationKernel, T, quantize)::runAVX512(
          a, total, scale, offset, result);
    };
  } else if (instruction & MKLInstruction::AVX256) {
    func = [](const float* a, unsigned total, double scale, int64_t offset, T* result) {
      TFCC_MKL_GET_RUNNER_HELPER(_MKLQuantizationKernel, T, quantize)::runAVX256(
          a, total, scale, offset, result);
    };
  } else {
    func = [](const float* a, unsigned total, double scale, int64_t offset, T* result) {
      TFCC_MKL_GET_RUNNER_HELPER(_MKLQuantizationKernel, T, quantize)::runNormal(
          a, total, scale, offset, result);
    };
  }

  Variable<T> result(a.shape());
  mkl_async_wrapper(
      "quantize",
      [func](
          const float* a, unsigned total, const float* minValue, const float* maxValue, T* result) {
        if (*minValue >= *maxValue) {
          _mkl_set_value(std::numeric_limits<T>::lowest(), total, result);
          return;
        }
        double scale;
        int64_t offset;
        std::tie(scale, offset) = get_quantized_scale_info<T>(*minValue, *maxValue);
        func(a, total, scale, offset, result);
      },
      a.data(), a.size(), minValue.data(), maxValue.data(), result.data());
  return result;
}

template <class T>
Variable<T> MKLQuantizationInterface<T>::requantize(
    const Tensor<int8_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
    const Tensor<float>& minOutput, const Tensor<float>& maxOutput) {
  return _requantize_helper<T, int8_t>(a, minInput, maxInput, minOutput, maxOutput);
}

template <class T>
Variable<T> MKLQuantizationInterface<T>::requantize(
    const Tensor<uint8_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
    const Tensor<float>& minOutput, const Tensor<float>& maxOutput) {
  return _requantize_helper<T, uint8_t>(a, minInput, maxInput, minOutput, maxOutput);
}

template <class T>
Variable<T> MKLQuantizationInterface<T>::requantize(
    const Tensor<int16_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
    const Tensor<float>& minOutput, const Tensor<float>& maxOutput) {
  return _requantize_helper<T, int16_t>(a, minInput, maxInput, minOutput, maxOutput);
}

template <class T>
Variable<T> MKLQuantizationInterface<T>::requantize(
    const Tensor<uint16_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
    const Tensor<float>& minOutput, const Tensor<float>& maxOutput) {
  return _requantize_helper<T, uint16_t>(a, minInput, maxInput, minOutput, maxOutput);
}

template <class T>
Variable<T> MKLQuantizationInterface<T>::requantize(
    const Tensor<int32_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
    const Tensor<float>& minOutput, const Tensor<float>& maxOutput) {
  return _requantize_helper<T, int32_t>(a, minInput, maxInput, minOutput, maxOutput);
}

template <class T>
Variable<T> MKLQuantizationInterface<T>::requantize(
    const Tensor<uint32_t>& a, const Tensor<float>& minInput, const Tensor<float>& maxInput,
    const Tensor<float>& minOutput, const Tensor<float>& maxOutput) {
  return _requantize_helper<T, uint32_t>(a, minInput, maxInput, minOutput, maxOutput);
}

template <class T>
Variable<float> MKLQuantizationInterface<T>::dequantize(
    const Tensor<T>& a, const Tensor<float>& minValue, const Tensor<float>& maxValue) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  if (minValue.size() != 1 || minValue.shape().size() != 1) {
    throw InvalidArgumentError("invalid minValue");
  }
  if (maxValue.size() != 1 || maxValue.shape().size() != 1) {
    throw InvalidArgumentError("invalid maxValue");
  }

  std::function<void(const T*, unsigned, double, double, float*)> func;
  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  auto instruction = session->getCPUInstructionFlags();
  if (instruction & MKLInstruction::AVX512) {
    func = [](const T* a, unsigned total, double scale, double minRounded, float* result) {
      TFCC_MKL_GET_RUNNER_HELPER(_MKLQuantizationKernel, T, dequantize)::runAVX512(
          a, total, scale, minRounded, result);
    };
  } else if (instruction & MKLInstruction::AVX256) {
    func = [](const T* a, unsigned total, double scale, double minRounded, float* result) {
      TFCC_MKL_GET_RUNNER_HELPER(_MKLQuantizationKernel, T, dequantize)::runAVX256(
          a, total, scale, minRounded, result);
    };
  } else {
    func = [](const T* a, unsigned total, double scale, double minRounded, float* result) {
      TFCC_MKL_GET_RUNNER_HELPER(_MKLQuantizationKernel, T, dequantize)::runNormal(
          a, total, scale, minRounded, result);
    };
  }

  Variable<float> result(a.shape());
  mkl_async_wrapper(
      "dequantize",
      [func](
          const T* a, unsigned total, const float* minValue, const float* maxValue, float* result) {
        if (*minValue >= *maxValue) {
          _mkl_set_value(*minValue, total, result);
          return;
        }
        double scale, minRounded;
        std::tie(scale, minRounded) = get_dequantized_scale_info<T>(*minValue, *maxValue);
        func(a, total, scale, minRounded, result);
      },
      a.data(), a.size(), minValue.data(), maxValue.data(), result.data());

  return result;
}

#define DEFINE_FUNC(type) template class MKLQuantizationInterface<type>;

TFCC_FOR_QUANTIZATION_TYPES(DEFINE_FUNC);

}  // namespace tfcc
