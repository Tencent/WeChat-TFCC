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

#include "tfcc_mklbasicinterface.h"

#include <omp.h>
#include <algorithm>
#include <cstdlib>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_mklsession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "framework/tfcc_view.h"
#include "interfaces/tfcc_mklinterfacehelper.h"
#include "kernel/tfcc_mkltransposekernel.avx256.h"

namespace tfcc {

// base
template <class T>
static void _mkl_slice(
    const T* a, unsigned s1, unsigned s2, unsigned s3, T* b, unsigned start, unsigned length) {
  if (s1 * s2 * s3 < 1024) {
    for (unsigned i = 0; i < s1; ++i) {
      const T* a1 = a + i * s2 * s3;
      T* b1 = b + i * length * s3;
      for (unsigned j = start; j < start + length; ++j) {
        const T* a2 = a1 + j * s3;
        T* b2 = b1 + (j - start) * s3;
        for (unsigned k = 0; k < s3; ++k) {
          b2[k] = a2[k];
        }
      }
    }
  } else if (s1 > 4) {
#pragma omp parallel for
    for (unsigned i = 0; i < s1; ++i) {
      const T* a1 = a + i * s2 * s3;
      T* b1 = b + i * length * s3;
      for (unsigned j = start; j < start + length; ++j) {
        const T* a2 = a1 + j * s3;
        T* b2 = b1 + (j - start) * s3;
        for (unsigned k = 0; k < s3; ++k) {
          b2[k] = a2[k];
        }
      }
    }
  } else if (length > 4) {
    for (unsigned i = 0; i < s1; ++i) {
      const T* a1 = a + i * s2 * s3;
      T* b1 = b + i * length * s3;
#pragma omp parallel for
      for (unsigned j = start; j < start + length; ++j) {
        const T* a2 = a1 + j * s3;
        T* b2 = b1 + (j - start) * s3;
        for (unsigned k = 0; k < s3; ++k) {
          b2[k] = a2[k];
        }
      }
    }
  } else {
    for (unsigned i = 0; i < s1; ++i) {
      const T* a1 = a + i * s2 * s3;
      T* b1 = b + i * length * s3;
      for (unsigned j = start; j < start + length; ++j) {
        const T* a2 = a1 + j * s3;
        T* b2 = b1 + (j - start) * s3;
#pragma omp parallel for
        for (unsigned k = 0; k < s3; ++k) {
          b2[k] = a2[k];
        }
      }
    }
  }
}

template <class T>
static void _mkl_assign_to(
    const T* a, unsigned s1, unsigned s2, unsigned s3, T* b, unsigned start, unsigned length) {
  if (s1 * s2 * s3 < 1024) {
    for (unsigned i = 0; i < s1; ++i) {
      const T* a1 = a + i * s2 * s3;
      T* b1 = b + i * length * s3;
      for (unsigned j = 0; j < s2; ++j) {
        const T* a2 = a1 + j * s3;
        T* b2 = b1 + (j + start) * s3;
        for (unsigned k = 0; k < s3; ++k) {
          b2[k] = a2[k];
        }
      }
    }
  } else if (s1 > 4) {
#pragma omp parallel for
    for (unsigned i = 0; i < s1; ++i) {
      const T* a1 = a + i * s2 * s3;
      T* b1 = b + i * length * s3;
      for (unsigned j = 0; j < s2; ++j) {
        const T* a2 = a1 + j * s3;
        T* b2 = b1 + (j + start) * s3;
        for (unsigned k = 0; k < s3; ++k) {
          b2[k] = a2[k];
        }
      }
    }
  } else if (s2 > 4) {
    for (unsigned i = 0; i < s1; ++i) {
      const T* a1 = a + i * s2 * s3;
      T* b1 = b + i * length * s3;
#pragma omp parallel for
      for (unsigned j = 0; j < s2; ++j) {
        const T* a2 = a1 + j * s3;
        T* b2 = b1 + (j + start) * s3;
        for (unsigned k = 0; k < s3; ++k) {
          b2[k] = a2[k];
        }
      }
    }
  } else {
    for (unsigned i = 0; i < s1; ++i) {
      const T* a1 = a + i * s2 * s3;
      T* b1 = b + i * length * s3;
      for (unsigned j = 0; j < s2; ++j) {
        const T* a2 = a1 + j * s3;
        T* b2 = b1 + (j + start) * s3;
#pragma omp parallel for
        for (unsigned k = 0; k < s3; ++k) {
          b2[k] = a2[k];
        }
      }
    }
  }
}

template <class T>
static void _mkl_clip(const T* a, unsigned total, T minValue, T maxValue, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    T v = a[i];
    v = std::min(v, maxValue);
    v = std::max(v, minValue);
    b[i] = v;
  }
}

template <class T>
static void _mkl_top_k_batch_parallel(
    const T* a, unsigned batch, unsigned n, unsigned k, T* b, unsigned* indices) {
  struct ValueIndex {
    T value;
    unsigned index;
  };
  auto sorter = [](const ValueIndex& v1, const ValueIndex& v2) { return v1.value > v2.value; };

#pragma omp parallel for
  for (unsigned i = 0; i < batch; ++i) {
    const T* ra = a + n * i;
    T* rb = b + k * i;
    unsigned* ri = indices + k * i;
    size_t currentCount = 0;
    std::vector<ValueIndex> vList(k + 1);

    for (unsigned j = 0; j < n; ++j) {
      vList[currentCount] = {ra[j], j};
      ++currentCount;
      std::push_heap(vList.begin(), vList.begin() + currentCount, sorter);
      if (currentCount == vList.size()) {
        std::pop_heap(vList.begin(), vList.begin() + currentCount, sorter);
        --currentCount;
      }
    }

    std::sort_heap(vList.begin(), vList.begin() + currentCount, sorter);

    for (size_t j = 0; j < static_cast<size_t>(k); ++j) {
      rb[j] = vList[j].value;
      ri[j] = vList[j].index;
    }
  }
}

template <class T>
static void _mkl_top_k_top_parallel(
    const T* a, unsigned batch, unsigned n, unsigned k, T* b, unsigned* indices) {
  struct ValueIndex {
    T value;
    unsigned index;
  };
  auto sorter = [](const ValueIndex& v1, const ValueIndex& v2) { return v1.value > v2.value; };

  for (unsigned i = 0; i < batch; ++i) {
    const T* ra = a + n * i;
    T* rb = b + k * i;
    unsigned* ri = indices + k * i;

    std::vector<std::vector<ValueIndex>> heapList(omp_get_max_threads());
#pragma omp parallel for
    for (unsigned j = 0; j < n; ++j) {
      unsigned tid = omp_get_thread_num();
      std::vector<ValueIndex>& vList = heapList[tid];
      if (vList.size() >= k && vList[0].value >= ra[j]) {
        continue;
      }
      vList.push_back({ra[j], j});
      std::push_heap(vList.begin(), vList.end(), sorter);
      if (vList.size() > k) {
        std::pop_heap(vList.begin(), vList.end(), sorter);
        vList.pop_back();
      }
    }

    std::vector<ValueIndex> resultHeap;
    for (auto& heap : heapList) {
      if (heap.size() == 0) {
        continue;
      }
      std::sort_heap(heap.begin(), heap.end(), sorter);
      size_t currentLength = resultHeap.size();
      resultHeap.insert(resultHeap.end(), heap.begin(), heap.end());
      std::inplace_merge(
          resultHeap.begin(), resultHeap.begin() + currentLength, resultHeap.end(), sorter);
    }

    for (size_t j = 0; j < static_cast<size_t>(k); ++j) {
      rb[j] = resultHeap[j].value;
      ri[j] = resultHeap[j].index;
    }
  }
}

template <class T>
static inline void _mkl_real_transpose(
    const T* a, const std::vector<unsigned>& shape, const std::vector<unsigned>& oldOffsets,
    const std::vector<unsigned>& newOffsets, T* b, size_t deep, bool useOMP) {
  unsigned total = shape[deep];
  unsigned oldOffset = oldOffsets[deep];
  unsigned newOffset = newOffsets[deep];
  if (deep == shape.size() - 1) {
    if (useOMP && total > 1) {
#pragma omp parallel for
      for (unsigned i = 0; i < total; ++i) {
        b[newOffset * i] = a[oldOffset * i];
      }
    } else {
      for (unsigned i = 0; i < total; ++i) {
        b[newOffset * i] = a[oldOffset * i];
      }
    }
  } else {
    if (useOMP && total > 1) {
#pragma omp parallel for
      for (unsigned i = 0; i < total; ++i) {
        _mkl_real_transpose(
            a + oldOffset * i, shape, oldOffsets, newOffsets, b + newOffset * i, deep + 1, false);
      }
    } else {
      for (unsigned i = 0; i < total; ++i) {
        _mkl_real_transpose(
            a + oldOffset * i, shape, oldOffsets, newOffsets, b + newOffset * i, deep + 1, useOMP);
      }
    }
  }
}

template <class T>
static inline void _mkl_transpose(
    const T* a, const std::vector<unsigned>& shape, const std::vector<unsigned>& offsets, T* b) {
  std::vector<unsigned> oldOffsets(shape.size(), 0);
  unsigned currentOffset = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    oldOffsets[shape.size() - 1 - i] = currentOffset;
    currentOffset *= shape[shape.size() - 1 - i];
  }

  if (currentOffset > 1024) {
    _mkl_real_transpose(a, shape, oldOffsets, offsets, b, 0, true);
  } else {
    _mkl_real_transpose(a, shape, oldOffsets, offsets, b, 0, false);
  }
}

template <class T, unsigned ROW, unsigned COL>
static inline void _mkl_transpose_dim2_once_fix(
    const T* a, unsigned strideA, unsigned strideB, T* b) {
  T buffer[ROW * COL];
  for (unsigned i = 0; i < ROW; ++i) {
    const T* ra = a + strideA * i;
    T* rb = buffer + i;
    for (unsigned j = 0; j < COL; ++j) {
      *rb = *ra;
      ++ra;
      rb += ROW;
    }
  }

  for (unsigned i = 0; i < COL; ++i) {
    T* rb = b + i * strideB;
    for (unsigned j = 0; j < ROW; ++j) {
      rb[j] = buffer[i * ROW + j];
    }
  }
}

template <class T>
static inline void _mkl_transpose_dim2_once_comm(
    const T* a, unsigned row, unsigned col, unsigned strideA, unsigned strideB, T* b) {
  for (unsigned i = 0; i < row; ++i) {
    const T* ra = a + strideA * i;
    T* rb = b + i;
    for (unsigned j = 0; j < col; ++j) {
      *rb = *ra;
      ++ra;
      rb += strideB;
    }
  }
}

template <class T>
static inline void _mkl_transpose_dim2(const T* a, unsigned row, unsigned col, T* b) {
  constexpr unsigned ROW_SIZE_PER_BATCH = 32 / sizeof(T);
  constexpr unsigned COL_SIZE_PER_BATCH = 32 / sizeof(T);
  unsigned rowBatch = (row + ROW_SIZE_PER_BATCH - 1) / ROW_SIZE_PER_BATCH;
  unsigned colBatch = (col + COL_SIZE_PER_BATCH - 1) / COL_SIZE_PER_BATCH;

#pragma omp parallel for
  for (unsigned i = 0; i < rowBatch * colBatch; ++i) {
    unsigned r = i / colBatch * ROW_SIZE_PER_BATCH;
    unsigned c = i % colBatch * COL_SIZE_PER_BATCH;

    unsigned nowRow = std::min(ROW_SIZE_PER_BATCH, row - r);
    unsigned nowCol = std::min(COL_SIZE_PER_BATCH, col - c);
    if (nowRow == ROW_SIZE_PER_BATCH && nowCol == COL_SIZE_PER_BATCH) {
      _mkl_transpose_dim2_once_fix<T, ROW_SIZE_PER_BATCH, COL_SIZE_PER_BATCH>(
          a + r * col + c, col, row, b + c * row + r);
    } else {
      _mkl_transpose_dim2_once_comm<T>(a + r * col + c, nowRow, nowCol, col, row, b + c * row + r);
    }
  }
}

template <class T>
static inline void _mkl_concat_v1(
    const std::vector<const T*>& datas, const std::vector<unsigned>& chunkList,
    const std::vector<unsigned>& offsetList, T* result, unsigned batch, unsigned totalChunk) {
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < datas.size(); ++i) {
    const T* data = datas[i];
    T* r = result + offsetList[i];
    for (size_t j = 0; j < batch; ++j) {
      for (size_t k = 0; k < chunkList[i]; ++k) {
        r[k] = *data;
        ++data;
      }
      r += totalChunk;
    }
  }
}

template <class T>
static inline void _mkl_concat_v2(
    const std::vector<const T*>& datas, const std::vector<unsigned>& chunkList,
    const std::vector<unsigned>& offsetList, T* result, unsigned batch, unsigned totalChunk) {
#pragma omp parallel
  {
    unsigned maxLength = (batch + omp_get_max_threads() - 1) / omp_get_max_threads();
    unsigned startBatch = maxLength * omp_get_thread_num();
    unsigned endBatch = std::min(startBatch + maxLength, batch);
    std::vector<const T*> tmpDatas(datas.size());
    for (size_t i = 0; i < tmpDatas.size(); ++i) {
      tmpDatas[i] = datas[i] + chunkList[i] * startBatch;
    }
    T* r = result + startBatch * totalChunk;
    for (unsigned i = startBatch; i < endBatch; ++i) {
      for (unsigned j = 0; j < tmpDatas.size(); ++j) {
        for (unsigned k = 0; k < chunkList[j]; ++k) {
          *r = *tmpDatas[j];
          ++r;
          ++tmpDatas[j];
        }
      }
    }
  }
}

template <class T>
static inline void _mkl_where(
    const uint8_t* condition, unsigned total, const T* x, const T* y, T* result) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    result[i] = condition[i] != 0 ? x[i] : y[i];
  }
}

template <class T>
static inline void _mkl_where(
    const uint8_t* condition, unsigned total, T x, const T* y, T* result) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    result[i] = condition[i] != 0 ? x : y[i];
  }
}

template <class T>
static inline void _mkl_where(
    const uint8_t* condition, unsigned total, const T* x, T y, T* result) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    result[i] = condition[i] != 0 ? x[i] : y;
  }
}

template <class T>
static inline void _mkl_abs(const T* a, unsigned total, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    b[i] = a[i] < static_cast<T>(0) ? -a[i] : a[i];
  }
}

template <class T>
static inline void _mkl_tril(const T* a, unsigned row, unsigned col, int64_t k, T* b) {
#pragma omp parallel for
  for (unsigned r = 0; r < row; ++r) {
    const T* ra = a + r * col;
    T* rb = b + r * col;
    int64_t x = std::min(static_cast<int64_t>(r) + k + 1, static_cast<int64_t>(col));
    int64_t c = 0;
    for (; c < x; ++c) {
      rb[c] = ra[c];
    }

    for (; c < static_cast<int64_t>(col); ++c) {
      rb[c] = static_cast<T>(0);
    }
  }
}

template <class T>
static inline void _mkl_triu(const T* a, unsigned row, unsigned col, int64_t k, T* b) {
#pragma omp parallel for
  for (unsigned r = 0; r < row; ++r) {
    const T* ra = a + r * col;
    T* rb = b + r * col;
    int64_t x = std::min(static_cast<int64_t>(r) + k, static_cast<int64_t>(col));
    int64_t c = 0;
    for (; c < x; ++c) {
      rb[c] = static_cast<T>(0);
    }

    for (; c < static_cast<int64_t>(col); ++c) {
      rb[c] = ra[c];
    }
  }
}

template <class T>
static inline void _mkl_argmax(const T* a, unsigned s1, unsigned s2, unsigned s3, int64_t* b) {
  unsigned batch = s1 * s3;
#pragma omp parallel for
  for (unsigned i = 0; i < batch; ++i) {
    unsigned x = i / s3;
    unsigned z = i % s3;
    T value = a[z + x * s3 * s2];
    int64_t index = 0;
    for (unsigned y = 0; y < s2; ++y) {
      if (a[z + y * s3 + x * s3 * s2] > value) {
        value = a[z + y * s3 + x * s3 * s2];
        index = y;
      }
    }
    b[z + x * s3] = index;
  }
}

template <class T>
Variable<T> MKLBasicInterface<T>::slice(
    const Tensor<T>& a, size_t axis, unsigned start, unsigned end) {
  end = std::min(end, a.shape(axis));

  unsigned s1 = 1;
  unsigned s2 = a.shape(axis);
  unsigned s3 = 1;

  for (size_t i = 0; i < axis; ++i) {
    s1 *= a.shape(i);
  }
  for (size_t i = axis + 1; i < a.shape().size(); ++i) {
    s3 *= a.shape(i);
  }

  std::vector<unsigned> s = a.shape().toVector();
  s[axis] = end - start;
  Variable<T> result(s);

  mkl_async_wrapper(
      "slice",
      [](const T* a, unsigned s1, unsigned s2, unsigned s3, T* b, unsigned start, unsigned length) {
        _mkl_slice(a, s1, s2, s3, b, start, length);
      },
      a.data(), s1, s2, s3, result.data(), start, end - start);

  return result;
}

template <class T>
void MKLBasicInterface<T>::assignTo(
    const Tensor<T>& a, size_t axis, unsigned start, Variable<T>& b) {
  unsigned s1 = 1;
  unsigned s2 = a.shape(axis);
  unsigned s3 = 1;

  for (size_t i = 0; i < axis; ++i) {
    s1 *= a.shape(i);
  }
  for (size_t i = axis + 1; i < a.shape().size(); ++i) {
    s3 *= a.shape(i);
  }

  mkl_async_wrapper(
      "assign_to",
      [](const T* a, unsigned s1, unsigned s2, unsigned s3, T* b, unsigned start, unsigned length) {
        _mkl_assign_to(a, s1, s2, s3, b, start, length);
      },
      a.data(), s1, s2, s3, b.data(), start, b.shape(axis));
}

template <class T>
Variable<T> MKLBasicInterface<T>::transpose(const Tensor<T>& a, const std::vector<size_t>& perm) {
  std::vector<unsigned> newS;
  newS.reserve(a.shape().size());
  for (size_t i = 0; i < a.shape().size(); ++i) {
    newS.emplace_back(a.shape(perm[i]));
  }
  Variable<T> result(std::move(newS));
  unsigned lastOffset = 1;
  std::vector<unsigned> newOffsets(a.shape().size(), 0u);
  for (size_t i = 0; i < a.shape().size(); ++i) {
    newOffsets[perm[perm.size() - i - 1]] = lastOffset;
    lastOffset *= result.shape(result.shape().size() - i - 1);
  }

  // speed up avx2
  MKLSession* session = static_cast<MKLSession*>(Session::getThreadDefault());
  auto instruction = session->getCPUInstructionFlags();
  if (sizeof(T) == sizeof(float) && instruction & MKLInstruction::AVX256) {
    // ba
    if (perm.size() == 2 && perm[0] == 1 && perm[1] == 0) {
      mkl_async_wrapper(
          "transpose",
          [](const float* a, unsigned row, unsigned col, float* b) {
            _MKLTransposeKernelAVX256<float>::transposeBA(a, row, col, b);
          },
          reinterpret_cast<const float*>(a.data()), a.shape(0), a.shape(1),
          reinterpret_cast<float*>(result.data()));
      return result;
    }
    // acb
    if (perm.size() == 3 && perm[0] == 0 && perm[1] == 2 && perm[2] == 1) {
      mkl_async_wrapper(
          "transpose",
          [](const float* a, unsigned depth, unsigned row, unsigned col, float* b) {
            _MKLTransposeKernelAVX256<float>::transposeACB(a, depth, row, col, b);
          },
          reinterpret_cast<const float*>(a.data()), a.shape(0), a.shape(1), a.shape(2),
          reinterpret_cast<float*>(result.data()));
      return result;
    }
    // bac
    if (perm.size() == 3 && perm[0] == 1 && perm[1] == 0 && perm[2] == 2) {
      mkl_async_wrapper(
          "transpose",
          [](const float* a, unsigned depth, unsigned row, unsigned col, float* b) {
            _MKLTransposeKernelAVX256<float>::transposeBAC(a, depth, row, col, b);
          },
          reinterpret_cast<const float*>(a.data()), a.shape(0), a.shape(1), a.shape(2),
          reinterpret_cast<float*>(result.data()));
      return result;
    }
    // abdc
    if (perm.size() == 4 && perm[0] == 0 && perm[1] == 1 && perm[2] == 3 && perm[3] == 2) {
      mkl_async_wrapper(
          "transpose",
          [](const float* a, unsigned depth, unsigned row, unsigned col, float* b) {
            _MKLTransposeKernelAVX256<float>::transposeACB(a, depth, row, col, b);
          },
          reinterpret_cast<const float*>(a.data()), a.shape(0) * a.shape(1), a.shape(2), a.shape(3),
          reinterpret_cast<float*>(result.data()));
      return result;
    }
    // adbc
    if (perm.size() == 4 && perm[0] == 0 && perm[1] == 3 && perm[2] == 1 && perm[3] == 2) {
      mkl_async_wrapper(
          "transpose",
          [](const float* a, unsigned depth, unsigned row, unsigned col, float* b) {
            _MKLTransposeKernelAVX256<float>::transposeACB(a, depth, row, col, b);
          },
          reinterpret_cast<const float*>(a.data()), a.shape(0), a.shape(1) * a.shape(2), a.shape(3),
          reinterpret_cast<float*>(result.data()));
      return result;
    }
  }

  // speed up transpose
  if (perm.size() == 2 && perm[0] == 1 && perm[1] == 0) {
    mkl_async_wrapper(
        "transpose",
        [](const T* a, unsigned row, unsigned col, T* b) { _mkl_transpose_dim2(a, row, col, b); },
        a.data(), a.shape(0), a.shape(1), result.data());
    return result;
  }

  mkl_async_wrapper(
      "transpose",
      [](const T* a, const std::vector<unsigned>& shape, const std::vector<unsigned>& offsets,
         T* b) { _mkl_transpose(a, shape, offsets, b); },
      a.data(), a.shape().toVector(), newOffsets, result.data());

  return result;
}

template <class T>
Variable<T> MKLBasicInterface<T>::clip(const Tensor<T>& a, T minValue, T maxValue) {
  Variable<T> result(a.shape());

  mkl_async_wrapper(
      "clip",
      [](const T* a, unsigned total, T minValue, T maxValue, T* b) {
        _mkl_clip(a, total, minValue, maxValue, b);
      },
      a.data(), a.size(), minValue, maxValue, result.data());

  return result;
}

template <class T>
Variable<T> MKLBasicInterface<T>::concat(const std::vector<const Tensor<T>*>& values, size_t axis) {
  unsigned batch = 1;
  unsigned chunk = 1;
  for (size_t i = 0; i < values[0]->shape().size() && i < axis; ++i) {
    batch *= values[0]->shape(i);
  }
  for (size_t i = axis + 1; i < values[0]->shape().size(); ++i) {
    chunk *= values[0]->shape(i);
  }
  std::vector<const T*> datas;
  std::vector<unsigned> chunkList;
  std::vector<unsigned> offsetList;

  std::vector<unsigned> shape = values[0]->shape().toVector();
  shape[axis] = 0;
  unsigned offset = 0;
  for (const auto* tensor : values) {
    unsigned currentChunk = tensor->shape(axis) * chunk;
    datas.push_back(tensor->data());
    chunkList.push_back(currentChunk);
    offsetList.push_back(offset);
    offset += currentChunk;
    shape[axis] += tensor->shape(axis);
  }

  Variable<T> result(shape);

  mkl_async_wrapper(
      "concat",
      [](const std::vector<const T*>& datas, const std::vector<unsigned>& chunkList,
         const std::vector<unsigned>& offsetList, T* result, unsigned batch, unsigned totalChunk) {
        unsigned threadCnt = omp_get_max_threads();
        if (threadCnt > batch) {
          _mkl_concat_v1(datas, chunkList, offsetList, result, batch, totalChunk);
        } else {
          _mkl_concat_v2(datas, chunkList, offsetList, result, batch, totalChunk);
        }
      },
      datas, chunkList, offsetList, result.data(), batch, offset);
  return result;
}

template <class T>
Variable<T> MKLBasicInterface<T>::where(
    const Tensor<uint8_t>& condition, const Tensor<T>& x, const Tensor<T>& y) {
  Variable<T> result(condition.shape());
  mkl_async_wrapper(
      "where",
      [](const uint8_t* cond, unsigned length, const T* a, const T* b, T* r) {
        _mkl_where(cond, length, a, b, r);
      },
      condition.data(), condition.size(), x.data(), y.data(), result.data());
  return result;
}

template <class T>
Variable<T> MKLBasicInterface<T>::where(const Tensor<uint8_t>& condition, T x, const Tensor<T>& y) {
  Variable<T> result(condition.shape());
  mkl_async_wrapper(
      "where",
      [](const uint8_t* cond, unsigned length, T a, const T* b, T* r) {
        _mkl_where(cond, length, a, b, r);
      },
      condition.data(), condition.size(), x, y.data(), result.data());
  return result;
}

template <class T>
Variable<T> MKLBasicInterface<T>::where(const Tensor<uint8_t>& condition, const Tensor<T>& x, T y) {
  Variable<T> result(condition.shape());
  mkl_async_wrapper(
      "where",
      [](const uint8_t* cond, unsigned length, const T* a, T b, T* r) {
        _mkl_where(cond, length, a, b, r);
      },
      condition.data(), condition.size(), x.data(), y, result.data());
  return result;
}

template <class T>
Variable<T> MKLBasicInterface<T>::abs(const Tensor<T>& a) {
  Variable<T> result(a.shape());
  mkl_async_wrapper(
      "abs", [](const T* a, unsigned total, T* b) { _mkl_abs(a, total, b); }, a.data(), a.size(),
      result.data());
  return result;
}

template <class T>
Variable<T> MKLBasicInterface<T>::tril(const Tensor<T>& a, int64_t k) {
  Variable<T> result(a.shape());
  mkl_async_wrapper(
      "tril",
      [](const T* a, unsigned row, unsigned col, int64_t k, T* b) { _mkl_tril(a, row, col, k, b); },
      a.data(), a.shape(0), a.shape(1), k, result.data());
  return result;
}

template <class T>
Variable<T> MKLBasicInterface<T>::triu(const Tensor<T>& a, int64_t k) {
  Variable<T> result(a.shape());
  mkl_async_wrapper(
      "triu",
      [](const T* a, unsigned row, unsigned col, int64_t k, T* b) { _mkl_triu(a, row, col, k, b); },
      a.data(), a.shape(0), a.shape(1), k, result.data());
  return result;
}

template <class T>
Variable<int64_t> MKLBasicInterface<T>::argmax(const Tensor<T>& a, size_t axis) {
  std::vector<unsigned> resultShape = a.shape().toVector();
  resultShape[axis] = 1;
  Variable<int64_t> result(resultShape);
  unsigned s1 = 1, s2 = 1, s3 = 1;
  s2 = a.shape(axis);
  for (size_t i = 0; i < axis; ++i) {
    s1 *= a.shape(i);
  }
  for (size_t i = axis + 1; i < a.shape().size(); ++i) {
    s3 *= a.shape(i);
  }
  mkl_async_wrapper(
      "argmax",
      [](const T* a, unsigned s1, unsigned s2, unsigned s3, int64_t* b) {
        _mkl_argmax(a, s1, s2, s3, b);
      },
      a.data(), s1, s2, s3, result.data());
  return result;
}

template <class T>
std::tuple<Variable<T>, Variable<uint32_t>> MKLBasicInterface<T>::topK(
    const Tensor<T>& a, unsigned k) {
  std::vector<unsigned> s = a.shape().toVector();
  s[s.size() - 1] = k;

  Variable<T> result(s);
  Variable<uint32_t> indices(s);

  unsigned n = a.shape(a.shape().size() - 1);
  unsigned batch = a.size() / n;
  unsigned maxBatch = omp_get_max_threads();
  if (batch < maxBatch) {
    mkl_async_wrapper(
        "top_k",
        [](const T* a, unsigned batch, unsigned n, unsigned k, T* b, unsigned* indices) {
          _mkl_top_k_top_parallel(a, batch, n, k, b, indices);
        },
        a.data(), batch, n, k, result.data(), indices.data());

  } else {
    mkl_async_wrapper(
        "top_k",
        [](const T* a, unsigned batch, unsigned n, unsigned k, T* b, unsigned* indices) {
          _mkl_top_k_batch_parallel(a, batch, n, k, b, indices);
        },
        a.data(), batch, n, k, result.data(), indices.data());
  }
  return std::make_tuple(std::move(result), std::move(indices));
}

#define DEFINE_FUNC(type) template class MKLBasicInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
