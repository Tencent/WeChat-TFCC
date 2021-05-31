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

#include "tfcc_mklquantizationmatmul.avx256.h"

#include <omp.h>
#include <vector>

#include "gemm/tfcc_mklsinglethreadgemm.h"
#include "utils/tfcc_commutils.h"

namespace tfcc {

static int32_t _mkl_quantized_matmul_1_1_k(const uint8_t* a, const int8_t* b, unsigned batch)
    __attribute__((noinline));

static int32_t _mkl_quantized_matmul_1_1_k(const uint8_t* a, const int8_t* b, unsigned batch) {
  int32_t result;
  asm volatile(
      "mt_1_1_k_loop_start_start:\n\t"
      "movq r13, %[a_ptr]\n\t"
      "movq r14, %[b_ptr]\n\t"
      "vpxor ymm2, ymm2, ymm2\n\t"
      "mt_1_1_k_loop_start:\n\t"
      "vpmovzxbw ymm0, [r13]\n\t"
      "vpmovsxbw ymm1, [r14]\n\t"
      "vpmaddwd ymm1, ymm0, ymm1\n\t"
      "vpaddd ymm2, ymm1, ymm2\n\t"
      "add r13, 0x10\n\t"
      "add r14, 0x10\n\t"
      "dec %[batch]\n\t"
      "jnz mt_1_1_k_loop_start\n\t"
      // horizontal add
      "vphaddd ymm2, ymm2, ymm2\n\t"
      "vphaddd ymm2, ymm2, ymm2\n\t"
      "vextracti128 xmm3, ymm2, 1\n\t"
      "paddd xmm3, xmm2\n\t"
      "movd %[result], xmm3\n\t"
      "mt_1_1_k_loop_start_end:\n\t"
      : [ result ] "=r"(result)
      : [ a_ptr ] "r"(a), [ b_ptr ] "r"(b), [ batch ] "r"(batch)
      : "r13", "r14",
#if __GNUC__ >= 5
        "ymm0", "ymm1", "ymm2", "ymm3", "cc"
#else
        "xmm0", "xmm1", "xmm2", "xmm3", "cc"
#endif
  );
  return result;
}

void _MKLQuantizationMatmulAVX256::quantizedMatmulN1(
    const uint8_t* a, const int32_t* reduceA, int32_t offsetA, const int8_t* b, int32_t reduceB,
    int32_t offsetB, unsigned m, unsigned k, int32_t* c) {
  unsigned batch = k / 16;
  unsigned tailStart = batch * 16;
  int32_t globalOffset = offsetA * offsetB * k + reduceB * offsetA;
#pragma omp parallel for
  for (unsigned i = 0; i < m; ++i) {
    int32_t result = 0;
    if (batch > 0) {
      result += _mkl_quantized_matmul_1_1_k(a + i * k, b, batch);
    }
    for (unsigned j = tailStart; j < k; ++j) {
      result += a[i * k + j] * b[j];
    }
    c[i] = result + globalOffset + reduceA[i] * offsetB;
  }
}

static void _quantization_matmul_once(
    MKLColMajorStrideOutput<int32_t> dst, MKLStrideInput<int8_t, false> a,
    MKLStrideInput<uint8_t, true> b) {
  thread_local std::vector<int8_t> cacheBufferA;
  thread_local std::vector<uint8_t> cacheBufferB;
  single_thread_matmul(dst, a, b, cacheBufferA, cacheBufferB);
}

void _MKLQuantizationMatmulAVX256::quantizedMatmulColMajor(
    unsigned batch, const int8_t* a, unsigned strideA, const uint8_t* b, unsigned strideB,
    unsigned m, unsigned n, unsigned k, int32_t* c, unsigned strideC) {
  unsigned maxThread = omp_get_max_threads();
  if (batch > maxThread * 4) {
#pragma omp parallel for
    for (unsigned i = 0; i < batch; ++i) {
      const auto* ra = a + i * strideA;
      const auto* rb = b + i * strideB;
      auto* rc = c + i * strideC;
      MKLColMajorStrideOutput<int32_t> dst(rc, m, n, m);
      MKLStrideInput<int8_t, false> sa(ra, m, k, m);
      MKLStrideInput<uint8_t, true> sb(rb, n, k, k);

      _quantization_matmul_once(dst, sa, sb);
    }
  }

  if (m / 24 > n / 4) {
    for (unsigned i = 0; i < batch; ++i) {
      const auto* ra = a + i * strideA;
      const auto* rb = b + i * strideB;
      auto* rc = c + i * strideC;

      MKLColMajorStrideOutput<int32_t> dst(rc, m, n, m);
      MKLStrideInput<int8_t, false> sa(ra, m, k, m);
      MKLStrideInput<uint8_t, true> sb(rb, n, k, k);

#pragma omp parallel
      {
        unsigned pn = omp_get_num_threads();
        unsigned tid = omp_get_thread_num();
        unsigned subM = roundUp(m, 24 * pn) / pn;
        unsigned ms = 0;
        if (subM * tid < sa.width()) {
          ms = std::min(subM, sa.width() - subM * tid);
        }

        if (ms > 0) {
          MKLStrideInput<int8_t, false> rsa = sa.subInput(subM * tid, 0, ms, sa.depth());
          MKLColMajorStrideOutput<int32_t> rdst = dst.subOutput(subM * tid, 0, ms, dst.col());

          _quantization_matmul_once(rdst, rsa, sb);
        }
      }
    }
  } else {
    for (unsigned i = 0; i < batch; ++i) {
      const auto* ra = a + i * strideA;
      const auto* rb = b + i * strideB;
      auto* rc = c + i * strideC;

      MKLColMajorStrideOutput<int32_t> dst(rc, m, n, m);
      MKLStrideInput<int8_t, false> sa(ra, m, k, m);
      MKLStrideInput<uint8_t, true> sb(rb, n, k, k);

#pragma omp parallel
      {
        unsigned pn = omp_get_num_threads();
        unsigned tid = omp_get_thread_num();
        unsigned subN = roundUp(n, 4 * pn) / pn;
        unsigned ns = 0;
        if (subN * tid < sb.width()) {
          ns = std::min(subN, sb.width() - subN * tid);
        }

        if (ns > 0) {
          MKLStrideInput<uint8_t, true> rsb = sb.subInput(subN * tid, 0, ns, sb.depth());
          MKLColMajorStrideOutput<int32_t> rdst = dst.subOutput(0, subN * tid, dst.row(), ns);

          _quantization_matmul_once(rdst, sa, rsb);
        }
      }
    }
  }
}

}  // namespace tfcc
