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

#include <xmmintrin.h>
#include <algorithm>

#include "tfcc_mklstrideinput.h"
#include "utils/tfcc_commutils.h"

namespace tfcc {

template <class T, bool WidthMajor, unsigned KernelWidth, unsigned KernelDepth, unsigned KernelCell>
class MKLInputPacker {
  unsigned _cacheWidth;
  unsigned _cacheDepth;

 public:
  MKLInputPacker(unsigned cacheWidth, unsigned cacheDepth)
      : _cacheWidth(cacheWidth), _cacheDepth(cacheDepth) {}

  void process(T* startDst, MKLStrideInput<T, WidthMajor> src) {
    for (unsigned d = 0; d < src.depth(); d += _cacheDepth) {
      T* dst = startDst + d * roundUp(src.width(), KernelWidth * KernelCell);
      unsigned ds = std::min(_cacheDepth, src.depth() - d);
      for (unsigned w = 0; w < src.width(); w += _cacheWidth) {
        unsigned ws = std::min(_cacheWidth, src.width() - w);
        auto subSrc = src.subInput(w, d, ws, ds);
        prefetchL1(subSrc);
        packL1(dst, subSrc);
        dst += roundUp(ds, KernelDepth) * _cacheWidth;
      }
    }
  }

 private:
  void prefetch(const T* p) { _mm_prefetch(p, _MM_HINT_T0); }

  void prefetchL1(MKLStrideInput<T, WidthMajor> src) {
    constexpr unsigned DEFAULT_CAHCE_LINE_SIZE = 64;
    if (WidthMajor) {
      for (unsigned d = 0; d < src.depth(); d += DEFAULT_CAHCE_LINE_SIZE) {
        for (unsigned w = 0; w < src.width(); ++w) {
          prefetch(src.data(w, d));
        }
      }
    } else {
      for (unsigned d = 0; d < src.depth(); ++d) {
        for (unsigned w = 0; w < src.width(); w += DEFAULT_CAHCE_LINE_SIZE) {
          prefetch(src.data(w, d));
        }
      }
    }
  }

  void packL1(T* dst, MKLStrideInput<T, WidthMajor> src) {
    unsigned ds = roundUp(src.depth(), KernelDepth);
    for (unsigned w = 0; w < src.width(); w += KernelWidth * KernelCell) {
      unsigned ws = std::min(KernelWidth * KernelCell, src.width() - w);
      auto subSrc = src.subInput(w, 0, ws, src.depth());
      packRun(dst, subSrc);
      dst += KernelWidth * KernelCell * ds;
    }
  }

  void packRun(T* dst, MKLStrideInput<T, WidthMajor> src) {
    for (unsigned d = 0; d < src.depth(); d += KernelDepth) {
      unsigned ds = std::min(KernelDepth, src.depth() - d);
      auto subSrc = src.subInput(0, d, src.width(), ds);
      packOnce(dst, subSrc);
      dst += KernelWidth * KernelCell * KernelDepth;
    }
  }

  void packOnce(T* dst, MKLStrideInput<T, WidthMajor> src) {
    unsigned w = 0;
    for (unsigned c = 0; c < KernelCell; ++c) {
      for (unsigned cw = 0; cw < KernelWidth && w < src.width(); ++w, ++cw) {
        unsigned d = 0;
        for (; d < src.depth(); ++d) {
          *dst = src(w, d);
          ++dst;
        }
        for (; d < KernelDepth; ++d) {
          *dst = 0;
          ++dst;
        }
      }
    }
    for (; w < KernelWidth * KernelCell; ++w) {
      for (unsigned d = 0; d < KernelDepth; ++d) {
        *dst = 0;
        ++dst;
      }
    }
  }
};

}  // namespace tfcc
