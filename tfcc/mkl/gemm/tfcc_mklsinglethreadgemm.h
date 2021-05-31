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

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "tfcc_mklcolmajorstrideoutput.h"
#include "tfcc_mklinputpacker.h"
#include "tfcc_mklinputpackermanager.h"
#include "tfcc_mklmatmulkernel.h"
#include "tfcc_mklstrideinput.h"

namespace tfcc {

template <unsigned BatchRow, unsigned BatchCol, unsigned BatchDepth>
struct MKLBlockInfo {
  unsigned l1Row;
  unsigned l1Col;
  unsigned l1Depth;

  unsigned l2Row;
  unsigned l2Col;
  unsigned l2Depth;

  MKLBlockInfo(unsigned row, unsigned col, unsigned depth) {
    constexpr unsigned L1_CACHE_SIZE = 32 * 1024;
    constexpr unsigned L2_CACHE_SIZE = 256 * 1024;

    unsigned k = roundDown(L2_CACHE_SIZE / (BatchRow + BatchCol), BatchDepth);
    if (k >= depth) {
      unsigned n = L2_CACHE_SIZE / (depth * (BatchRow + BatchCol));
      l2Row = BatchRow * n;
      l2Col = BatchCol * n;
      l2Depth = roundUp(depth, BatchDepth);
    } else {
      l2Row = BatchRow;
      l2Col = BatchCol;
      l2Depth = k;
    }
    if (l2Row > roundUp(row, BatchRow)) {
      l2Row = roundUp(row, BatchRow);
      l2Col = roundDown(L2_CACHE_SIZE - l2Row * l2Depth, BatchCol);
    } else if (l2Col > roundUp(col, BatchCol)) {
      l2Col = roundUp(col, BatchCol);
      l2Row = roundDown(L2_CACHE_SIZE - l2Col * l2Depth, BatchCol);
    }

    k = roundDown(L1_CACHE_SIZE / (BatchRow + BatchCol), BatchDepth);
    if (k >= l2Depth) {
      unsigned n = L1_CACHE_SIZE / (depth * (BatchRow + BatchCol));
      l1Row = BatchRow * n;
      l1Col = BatchCol * n;
      l1Depth = l2Depth;
    } else {
      l1Row = BatchRow;
      l1Col = BatchCol;
      l1Depth = k;
    }
  }
};

template <class PA, class PB, class Blk>
void matmul_compute(
    MKLColMajorStrideOutput<int32_t> dst, const PA* packA, const PB* packB, unsigned depth,
    Blk blockInfo) {
  for (unsigned d = 0; d < depth; d += blockInfo.l1Depth) {
    unsigned ds = std::min(blockInfo.l1Depth, depth - d);
    for (unsigned r = 0; r < dst.row(); r += 24) {
      for (unsigned c = 0; c < dst.col(); c += 4) {
        matmul_m24_n4_k2(
            packA + d * roundUp(dst.row(), 24u) + r * ds,
            packB + d * roundUp(dst.col(), 4u) + c * ds, dst.data(r, c), dst.stride(), ds / 2);
      }
    }
  }
}

template <class TA, bool AWidthMajor, class TB, bool BWidthMajor>
void single_thread_matmul(
    MKLColMajorStrideOutput<int32_t> originDst, MKLStrideInput<TA, AWidthMajor> a,
    MKLStrideInput<TB, BWidthMajor> b, std::vector<TA>& cacheBufferA,
    std::vector<TB>& cacheBufferB) {
  cacheBufferA.clear();
  cacheBufferB.clear();

  std::unique_ptr<int32_t> tmp;
  MKLColMajorStrideOutput<int32_t> dst;
  if (originDst.row() % 24 == 0 && originDst.col() % 4 == 0) {
    dst = originDst;
  } else {
    tmp = std::unique_ptr<int32_t>(
        new int32_t[roundUp(originDst.row(), 24u) * roundUp(originDst.col(), 4u)]);
    dst = MKLColMajorStrideOutput<int32_t>(
        tmp.get(), originDst.row(), originDst.col(), roundUp(originDst.row(), 24u));
    for (unsigned c = 0; c < originDst.col(); ++c) {
      memcpy(dst.data(0, c), originDst.data(0, c), originDst.row() * sizeof(int32_t));
    }
  }

  MKLBlockInfo<24, 4, 2> info(a.width(), b.width(), a.depth());

  MKLInputPackerManager<TA, AWidthMajor, 8, 2, 3> pa(info.l1Row, info.l1Depth, cacheBufferA);
  MKLInputPackerManager<TB, BWidthMajor, 4, 2, 1> pb(info.l1Col, info.l1Depth, cacheBufferB);

  for (unsigned d = 0; d < a.depth(); d += info.l2Depth) {
    unsigned ds = std::min(info.l2Depth, a.depth() - d);
    for (unsigned r = 0; r < a.width(); r += info.l2Row) {
      unsigned rs = std::min(info.l2Row, a.width() - r);
      auto subA = a.subInput(r, d, rs, ds);
      TA* packA = pa.pack(subA);
      for (unsigned c = 0; c < b.width(); c += info.l2Col) {
        unsigned cs = std::min(info.l2Col, b.width() - c);
        auto subB = b.subInput(c, d, cs, ds);
        auto subDst = dst.subOutput(r, c, rs, cs);
        TB* packB = pb.pack(subB);
        matmul_compute(subDst, packA, packB, roundUp(ds, 2u), info);
      }
    }
  }

  if (tmp) {
    for (unsigned c = 0; c < dst.col(); ++c) {
      memcpy(originDst.data(0, c), dst.data(0, c), dst.row() * sizeof(int32_t));
    }
  }
}

}  // namespace tfcc
