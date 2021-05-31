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

#include "tfcc_mkltransposekernel.avx256.h"

#include <immintrin.h>
#include <omp.h>
#include <algorithm>

#include "framework/tfcc_types.h"

namespace tfcc {

template <size_t ROW, size_t COL>
class Transpose8X8Kernel {
 public:
  static inline void transpose(const float* a, unsigned lda, unsigned ldb, float* result) {
    static_assert(ROW <= 8 && COL <= 8 && ROW > 0 && COL > 0, "invalid transpose row col");
    __m256 r0, r1, r2, r3, r4, r5, r6, r7;
    __m256 t0, t1, t2, t3, t4, t5, t6, t7;

    if (ROW > 4) {
      r0 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(a + 0 * lda + 0)), _mm_loadu_ps(a + 4 * lda + 0), 1);
      r4 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(a + 0 * lda + 4)), _mm_loadu_ps(a + 4 * lda + 4), 1);
    } else if (ROW > 0) {
      r0 = _mm256_castps128_ps256(_mm_loadu_ps(a + 0 * lda + 0));
      r4 = _mm256_castps128_ps256(_mm_loadu_ps(a + 0 * lda + 4));
    }
    if (ROW > 5) {
      r1 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(a + 1 * lda + 0)), _mm_loadu_ps(a + 5 * lda + 0), 1);
      r5 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(a + 1 * lda + 4)), _mm_loadu_ps(a + 5 * lda + 4), 1);
    } else if (ROW > 1) {
      r1 = _mm256_castps128_ps256(_mm_loadu_ps(a + 1 * lda + 0));
      r5 = _mm256_castps128_ps256(_mm_loadu_ps(a + 1 * lda + 4));
    }
    if (ROW > 6) {
      r2 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(a + 2 * lda + 0)), _mm_loadu_ps(a + 6 * lda + 0), 1);
      r6 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(a + 2 * lda + 4)), _mm_loadu_ps(a + 6 * lda + 4), 1);
    } else if (ROW > 2) {
      r2 = _mm256_castps128_ps256(_mm_loadu_ps(a + 2 * lda + 0));
      r6 = _mm256_castps128_ps256(_mm_loadu_ps(a + 2 * lda + 4));
    }
    if (ROW > 7) {
      r3 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(a + 3 * lda + 0)), _mm_loadu_ps(a + 7 * lda + 0), 1);
      r7 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(_mm_loadu_ps(a + 3 * lda + 4)), _mm_loadu_ps(a + 7 * lda + 4), 1);
    } else if (ROW > 3) {
      r3 = _mm256_castps128_ps256(_mm_loadu_ps(a + 3 * lda + 0));
      r7 = _mm256_castps128_ps256(_mm_loadu_ps(a + 3 * lda + 4));
    }

    t0 = _mm256_unpacklo_ps(r0, r1);
    t1 = _mm256_unpackhi_ps(r0, r1);
    t2 = _mm256_unpacklo_ps(r2, r3);
    t3 = _mm256_unpackhi_ps(r2, r3);
    t4 = _mm256_unpacklo_ps(r4, r5);
    t5 = _mm256_unpackhi_ps(r4, r5);
    t6 = _mm256_unpacklo_ps(r6, r7);
    t7 = _mm256_unpackhi_ps(r6, r7);

    __m256 v;

    v = _mm256_shuffle_ps(t0, t2, 0x4e);
    r0 = _mm256_blend_ps(t0, v, 0xcc);
    r1 = _mm256_blend_ps(t2, v, 0x33);

    v = _mm256_shuffle_ps(t1, t3, 0x4e);
    r2 = _mm256_blend_ps(t1, v, 0xcc);
    r3 = _mm256_blend_ps(t3, v, 0x33);

    v = _mm256_shuffle_ps(t4, t6, 0x4e);
    r4 = _mm256_blend_ps(t4, v, 0xcc);
    r5 = _mm256_blend_ps(t6, v, 0x33);

    v = _mm256_shuffle_ps(t5, t7, 0x4e);
    r6 = _mm256_blend_ps(t5, v, 0xcc);
    r7 = _mm256_blend_ps(t7, v, 0x33);

    if (ROW == 8) {
      switch (COL) {
        case 8:
          _mm256_storeu_ps(result + 7 * ldb, r7);
        case 7:
          _mm256_storeu_ps(result + 6 * ldb, r6);
        case 6:
          _mm256_storeu_ps(result + 5 * ldb, r5);
        case 5:
          _mm256_storeu_ps(result + 4 * ldb, r4);
        case 4:
          _mm256_storeu_ps(result + 3 * ldb, r3);
        case 3:
          _mm256_storeu_ps(result + 2 * ldb, r2);
        case 2:
          _mm256_storeu_ps(result + 1 * ldb, r1);
        case 1:
          _mm256_storeu_ps(result + 0 * ldb, r0);
      }
    } else {
      __m256i mask;
      switch (ROW) {
        case 1:
          mask = _mm256_setr_epi32(-1, 0, 0, 0, 0, 0, 0, 0);
          break;
        case 2:
          mask = _mm256_setr_epi32(-1, -1, 0, 0, 0, 0, 0, 0);
          break;
        case 3:
          mask = _mm256_setr_epi32(-1, -1, -1, 0, 0, 0, 0, 0);
          break;
        case 4:
          mask = _mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0);
          break;
        case 5:
          mask = _mm256_setr_epi32(-1, -1, -1, -1, -1, 0, 0, 0);
          break;
        case 6:
          mask = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0);
          break;
        case 7:
          mask = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, 0);
          break;
      }

      switch (COL) {
        case 8:
          _mm256_maskstore_ps(result + 7 * ldb, mask, r7);
        case 7:
          _mm256_maskstore_ps(result + 6 * ldb, mask, r6);
        case 6:
          _mm256_maskstore_ps(result + 5 * ldb, mask, r5);
        case 5:
          _mm256_maskstore_ps(result + 4 * ldb, mask, r4);
        case 4:
          _mm256_maskstore_ps(result + 3 * ldb, mask, r3);
        case 3:
          _mm256_maskstore_ps(result + 2 * ldb, mask, r2);
        case 2:
          _mm256_maskstore_ps(result + 1 * ldb, mask, r1);
        case 1:
          _mm256_maskstore_ps(result + 0 * ldb, mask, r0);
      }
    }
  }
};

class TransposeSelector {
  template <size_t ROW>
  class Helper {
   public:
    template <class... Args>
    void operator()(size_t col, Args... args) {
      switch (col) {
        case 1:
          Transpose8X8Kernel<ROW, 1>::transpose(args...);
          break;
        case 2:
          Transpose8X8Kernel<ROW, 2>::transpose(args...);
          break;
        case 3:
          Transpose8X8Kernel<ROW, 3>::transpose(args...);
          break;
        case 4:
          Transpose8X8Kernel<ROW, 4>::transpose(args...);
          break;
        case 5:
          Transpose8X8Kernel<ROW, 5>::transpose(args...);
          break;
        case 6:
          Transpose8X8Kernel<ROW, 6>::transpose(args...);
          break;
        case 7:
          Transpose8X8Kernel<ROW, 7>::transpose(args...);
          break;
        case 8:
          Transpose8X8Kernel<ROW, 8>::transpose(args...);
          break;
        default:
          break;
      }
    }
  };

 public:
  template <class... Args>
  void operator()(size_t row, size_t col, Args... args) {
    switch (row) {
      case 1:
        Helper<1>()(col, args...);
        break;
      case 2:
        Helper<2>()(col, args...);
        break;
      case 3:
        Helper<3>()(col, args...);
        break;
      case 4:
        Helper<4>()(col, args...);
        break;
      case 5:
        Helper<5>()(col, args...);
        break;
      case 6:
        Helper<6>()(col, args...);
        break;
      case 7:
        Helper<7>()(col, args...);
        break;
      case 8:
        Helper<8>()(col, args...);
        break;
    }
  }
};

// float
void _MKLTransposeKernelAVX256<float>::transposeBA(
    const float* a, unsigned row, unsigned col, float* b) {
  unsigned rowBatch = (row + 8 - 1) / 8;
  unsigned colBatch = (col + 8 - 1) / 8;

#pragma omp parallel for
  for (unsigned i = 0; i < rowBatch * colBatch; ++i) {
    unsigned r = i / colBatch * 8;
    unsigned c = i % colBatch * 8;

    unsigned nowRow = std::min(8u, row - r);
    unsigned nowCol = std::min(8u, col - c);
    if (nowRow == 8 && nowCol == 8) {
      Transpose8X8Kernel<8, 8>::transpose(a + r * col + c, col, row, b + c * row + r);
    } else {
      TransposeSelector()(nowRow, nowCol, a + r * col + c, col, row, b + c * row + r);
    }
  }
}

void _MKLTransposeKernelAVX256<float>::transposeACB(
    const float* a, unsigned depth, unsigned row, unsigned col, float* b) {
  unsigned rowBatch = (row + 8 - 1) / 8;
  unsigned colBatch = (col + 8 - 1) / 8;

#pragma omp parallel for
  for (unsigned d = 0; d < depth; ++d) {
    const float* ra = a + row * col * d;
    float* rb = b + row * col * d;
    for (unsigned i = 0; i < rowBatch * colBatch; ++i) {
      unsigned r = i / colBatch * 8;
      unsigned c = i % colBatch * 8;

      unsigned nowRow = std::min(8u, row - r);
      unsigned nowCol = std::min(8u, col - c);
      if (nowRow == 8 && nowCol == 8) {
        Transpose8X8Kernel<8, 8>::transpose(ra + r * col + c, col, row, rb + c * row + r);
      } else {
        TransposeSelector()(nowRow, nowCol, ra + r * col + c, col, row, rb + c * row + r);
      }
    }
  }
}

void _MKLTransposeKernelAVX256<float>::transposeBAC(
    const float* a, unsigned depth, unsigned row, unsigned col, float* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < depth * row; ++i) {
    unsigned d = i / row;
    unsigned r = i % row;
    const float* ra = a + d * row * col + r * col;
    float* rb = b + r * depth * col + d * col;
    for (unsigned j = 0; j < col; ++j) {
      rb[j] = ra[j];
    }
  }
}

#define DEFINE_FUNC(type) template class _MKLTransposeKernelAVX256<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
