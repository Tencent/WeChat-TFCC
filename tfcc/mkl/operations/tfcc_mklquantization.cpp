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

#include "tfcc_mklquantization.h"

#include <omp.h>
#include <cassert>
#include <cstdlib>

#include "utils/tfcc_debugutils.h"

#include "exceptions/tfcc_invalidargumenterror.h"
#include "framework/tfcc_types.h"
#include "interfaces/tfcc_mklinterfacehelper.h"
#include "kernel/tfcc_mklquantizationmatmul.avx256.h"
#include "utils/tfcc_quantizationutils.h"

namespace tfcc {
namespace mkl {
namespace quantization {

static inline void _mkl_create_offset_matrix(
    unsigned batch, const int32_t* a, unsigned totalA, unsigned strideA, const int32_t* b,
    unsigned totalB, unsigned strideB, int32_t offsetA, int32_t offsetB, int32_t k, int32_t* c,
    unsigned strideC) {
  int32_t globalOffset = offsetA * offsetB * k;
  if (static_cast<int>(batch) < omp_get_max_threads()) {
    for (unsigned i = 0; i < batch; ++i) {
      const int32_t* ra = a + i * strideA;
      const int32_t* rb = b + i * strideB;
      int32_t* rc = c + i * strideC;
#pragma omp parallel for
      for (unsigned j = 0; j < totalA; ++j) {
        int32_t* pc = rc + j * totalB;
        int32_t oa = ra[j] * offsetB;
        for (unsigned k = 0; k < totalB; ++k) {
          int32_t ob = rb[k] * offsetA;
          pc[k] = globalOffset + oa + ob;
        }
      }
    }
  } else {
#pragma omp parallel for
    for (unsigned i = 0; i < batch; ++i) {
      const int32_t* ra = a + i * strideA;
      const int32_t* rb = b + i * strideB;
      int32_t* rc = c + i * strideC;
      for (unsigned j = 0; j < totalA; ++j) {
        int32_t* pc = rc + j * totalB;
        int32_t oa = ra[j] * offsetB;
        for (unsigned k = 0; k < totalB; ++k) {
          int32_t ob = rb[k] * offsetA;
          pc[k] = globalOffset + oa + ob;
        }
      }
    }
  }
}

std::tuple<Variable<int32_t>, Variable<float>, Variable<float>> matmul(
    const Tensor<uint8_t>& a, const Tensor<int32_t>& reduceSumA, const Tensor<float>& minA,
    const Tensor<float>& maxA, const Tensor<int8_t>& b, const Tensor<int32_t>& reduceSumB,
    const Tensor<float>& minB, const Tensor<float>& maxB) {
  if (a.size() == 0 || a.shape().size() < 2) {
    throw InvalidArgumentError("invalid input tensor a");
  }
  if (reduceSumA.size() == 0) {
    throw InvalidArgumentError("invalid tensor reduceSumA");
  }
  if (minA.size() != 1 || minA.shape().size() != 1) {
    throw InvalidArgumentError("invalid tensor minA");
  }
  if (maxA.size() != 1 || maxA.shape().size() != 1) {
    throw InvalidArgumentError("invalid tensor maxA");
  }
  if (b.size() == 0 || b.shape().size() < 2) {
    throw InvalidArgumentError("invalid input tensor b");
  }
  if (reduceSumB.size() == 0) {
    throw InvalidArgumentError("invalid tensor reduceSumB");
  }
  if (minB.size() != 1 || minB.shape().size() != 1) {
    throw InvalidArgumentError("invalid tensor minB");
  }
  if (maxB.size() != 1 || maxB.shape().size() != 1) {
    throw InvalidArgumentError("invalid tensor maxB");
  }
  if (a.shape(a.shape().size() - 1) != b.shape(b.shape().size() - 2)) {
    throw InvalidArgumentError("tensor a and b don't match");
  }
  if (a.shape().size() != b.shape().size() && a.shape().size() != 2 && b.shape().size() != 2) {
    throw InvalidArgumentError("tensor a and b don't match");
  }
  if (a.shape(a.shape().size() - 2) != reduceSumA.shape(reduceSumA.shape().size() - 1)) {
    throw InvalidArgumentError("tensor a and reduceSumA don't match");
  }
  if (b.shape(b.shape().size() - 1) != reduceSumB.shape(reduceSumB.shape().size() - 1)) {
    throw InvalidArgumentError("tensor b and reduceSumB don't match");
  }

  if (a.shape().size() == b.shape().size()) {
    for (size_t i = 0; i < a.shape().size() - 2; ++i) {
      if (a.shape(i) != b.shape(i)) {
        throw InvalidArgumentError("tensor a and b don't match");
      }
    }
  }

  std::vector<unsigned> newS =
      a.shape().size() > b.shape().size() ? a.shape().toVector() : b.shape().toVector();
  newS[newS.size() - 2] = a.shape(a.shape().size() - 2);
  newS[newS.size() - 1] = b.shape(b.shape().size() - 1);

  Variable<int32_t> result(newS);
  unsigned batch = result.size() / result.shape(result.shape().size() - 1) /
                   result.shape(result.shape().size() - 2);
  unsigned m = a.shape(a.shape().size() - 2);
  unsigned n = b.shape(b.shape().size() - 1);
  unsigned k = a.shape(a.shape().size() - 1);
  unsigned strideReduceA = reduceSumA.shape().size() == 1 ? 0 : a.shape(a.shape().size() - 2);
  unsigned strideReduceB = reduceSumB.shape().size() == 1 ? 0 : b.shape(b.shape().size() - 1);
  unsigned strideA =
      a.shape().size() == 2 ? 0 : a.shape(a.shape().size() - 2) * a.shape(a.shape().size() - 1);
  unsigned strideB =
      b.shape().size() == 2 ? 0 : b.shape(b.shape().size() - 2) * b.shape(b.shape().size() - 1);
  unsigned strideC =
      result.shape(result.shape().size() - 2) * result.shape(result.shape().size() - 1);

  if (b.size() == b.shape(b.shape().size() - 2) * b.shape(b.shape().size() - 1)) {
    m *= batch;
    strideReduceA *= batch;
    strideA *= batch;
    batch = 1;
  }
  Variable<float> minResult({1}), maxResult({1});

  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  auto instruction = session->getCPUInstructionFlags();

  if (n == 1 && batch == 1 && (instruction & MKLInstruction::AVX256)) {
    mkl_async_wrapper(
        "quantization_matmul",
        [](const uint8_t* a, const int32_t* reduceA, const float* minA, const float* maxA,
           const int8_t* b, const int32_t* reduceB, const float* minB, const float* maxB,
           unsigned m, unsigned k, int32_t* c, float* minC, float* maxC) {
          int64_t offsetA;
          int64_t offsetB;
          std::tie(std::ignore, offsetA) = get_quantized_scale_info<uint8_t>(*minA, *maxA);
          std::tie(std::ignore, offsetB) = get_quantized_scale_info<int8_t>(*minB, *maxB);
          offsetA -= static_cast<int64_t>(std::numeric_limits<uint8_t>::lowest());
          offsetB -= static_cast<int64_t>(std::numeric_limits<int8_t>::lowest());

          _MKLQuantizationMatmulAVX256::quantizedMatmulN1(
              a, reduceA, offsetA, b, reduceB[0], offsetB, m, k, c);

          std::tie(*minC, *maxC) =
              get_quantization_range_for_multiplication<uint8_t, int8_t, int32_t>(
                  *minA, *maxA, *minB, *maxB);
        },
        a.data(), reduceSumA.data(), minA.data(), maxA.data(), b.data(), reduceSumB.data(),
        minB.data(), maxB.data(), m, k, result.data(), minResult.data(), maxResult.data());
    return std::make_tuple(std::move(result), std::move(minResult), std::move(maxResult));
  }

  mkl_async_wrapper(
      "create_offset_matrix",
      [](unsigned batch, const int32_t* a, unsigned totalA, unsigned strideA, const float* minA,
         const float* maxA, const int32_t* b, unsigned totalB, unsigned strideB, const float* minB,
         const float* maxB, int32_t k, int32_t* c, float* minC, float* maxC) {
        int64_t offsetA;
        int64_t offsetB;
        std::tie(std::ignore, offsetA) = get_quantized_scale_info<uint8_t>(*minA, *maxA);
        std::tie(std::ignore, offsetB) = get_quantized_scale_info<int8_t>(*minB, *maxB);
        offsetA -= static_cast<int64_t>(std::numeric_limits<uint8_t>::lowest());
        offsetB -= static_cast<int64_t>(std::numeric_limits<int8_t>::lowest());

        _mkl_create_offset_matrix(
            batch, a, totalA, strideA, b, totalB, strideB, static_cast<int32_t>(offsetA),
            static_cast<int32_t>(offsetB), k, c, totalA * totalB);
        std::tie(*minC, *maxC) =
            get_quantization_range_for_multiplication<uint8_t, int8_t, int32_t>(
                *minA, *maxA, *minB, *maxB);
      },
      batch, reduceSumA.data(), m, strideReduceA, minA.data(), maxA.data(), reduceSumB.data(), n,
      strideReduceB, minB.data(), maxB.data(), static_cast<int32_t>(a.shape(a.shape().size() - 1)),
      result.data(), minResult.data(), maxResult.data());

  mkl_async_wrapper(
      "quantization_matmul", _MKLQuantizationMatmulAVX256::quantizedMatmulColMajor, batch, b.data(),
      strideB, a.data(), strideA, n, m, k, result.data(), strideC);
  return std::make_tuple(std::move(result), std::move(minResult), std::move(maxResult));
}

template <class T>
static inline void _mkl_reduce_sum_row(
    const T* a, unsigned batch, unsigned row, unsigned column, int32_t* b) {
  if (static_cast<int>(batch) < omp_get_max_threads()) {
    for (unsigned i = 0; i < batch; ++i) {
      const T* ra = a + i * row * column;
      int32_t* rb = b + i * row;
#pragma omp parallel for
      for (unsigned j = 0; j < row; ++j) {
        int32_t tmp = 0;
        const T* pa = ra + j * column;
        for (unsigned k = 0; k < column; ++k) {
          tmp += static_cast<int32_t>(pa[k]);
        }
        rb[j] = tmp;
      }
    }
  } else {
#pragma omp parallel for
    for (unsigned i = 0; i < batch; ++i) {
      const T* ra = a + i * row * column;
      int32_t* rb = b + i * row;
      for (unsigned j = 0; j < row; ++j) {
        int32_t tmp = 0;
        const T* pa = ra + j * column;
        for (unsigned k = 0; k < column; ++k) {
          tmp += static_cast<int32_t>(pa[k]);
        }
        rb[j] = tmp;
      }
    }
  }
}

template <class T>
static inline void _mkl_reduce_sum_column(
    const T* a, unsigned batch, unsigned row, unsigned column, int32_t* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < batch * column; ++i) {
    b[i] = 0;
  }
  if (static_cast<int>(batch) < omp_get_max_threads()) {
#pragma omp parallel
    {
      unsigned pn = omp_get_num_threads();
      unsigned tid = omp_get_thread_num();
      unsigned tbatch = (column + pn - 1) / pn;
      unsigned start = std::min(tid * tbatch, column);
      unsigned end = std::min((tid + 1) * tbatch, column);
      unsigned total = end - start;

      for (unsigned i = 0; i < batch; ++i) {
        const T* ra = a + i * row * column;
        int32_t* rb = b + i * column;
        for (unsigned j = 0; j < row; ++j) {
          const T* pa = ra + start + j * column;
          int32_t* pb = rb + start;
          for (unsigned k = 0; k < total; ++k) {
            pb[k] += static_cast<int32_t>(pa[k]);
          }
        }
      }
    }
  } else {
#pragma omp parallel for
    for (unsigned i = 0; i < batch; ++i) {
      const T* ra = a + i * row * column;
      int32_t* rb = b + i * column;
      for (unsigned j = 0; j < row; ++j) {
        const T* pa = ra + j * column;
        int32_t* pb = rb;
        for (unsigned k = 0; k < column; ++k) {
          pb[k] += static_cast<int32_t>(pa[k]);
        }
      }
    }
  }
}

template <class T>
Variable<int32_t> reduce_sum(const Tensor<T>& a, bool row) {
  if (a.size() == 0 || a.shape().size() < 2) {
    throw InvalidArgumentError("invalid input tensor a");
  }

  Variable<int32_t> result;
  std::vector<unsigned> newS = a.shape().toVector();
  newS.pop_back();
  newS.pop_back();

  unsigned r = a.shape(a.shape().size() - 2);
  unsigned c = a.shape(a.shape().size() - 1);
  unsigned batch = a.size() / r / c;
  if (row) {
    newS.push_back(r);
    result = Variable<int32_t>(newS);
    mkl_async_wrapper(
        "mkl_reduce_sum_row",
        [](const T* a, unsigned batch, unsigned row, unsigned column, int32_t* b) {
          _mkl_reduce_sum_row(a, batch, row, column, b);
        },
        a.data(), batch, r, c, result.data());
  } else {
    newS.push_back(c);
    result = Variable<int32_t>(newS);
    mkl_async_wrapper(
        "mkl_reduce_sum_column",
        [](const T* a, unsigned batch, unsigned row, unsigned column, int32_t* b) {
          _mkl_reduce_sum_column(a, batch, row, column, b);
        },
        a.data(), batch, r, c, result.data());
  }
  return result;
}

#define DEFINE_FUNC(type) template Variable<int32_t> reduce_sum(const Tensor<type>& a, bool row);

DEFINE_FUNC(int8_t);
DEFINE_FUNC(uint8_t);
DEFINE_FUNC(int16_t);

}  // namespace quantization
}  // namespace mkl
}  // namespace tfcc
