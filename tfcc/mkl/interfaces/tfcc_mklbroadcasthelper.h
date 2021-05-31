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

#include "tfcc_mklarithmeticinterface.h"

#include <algorithm>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"
#include "framework/tfcc_view.h"

namespace tfcc {

template <class T, class Func>
static inline void _mkl_real_broadcast(
    const T* src, T* dst, const std::vector<unsigned>& dstShape,
    const std::vector<unsigned>& srcSkipSize, const std::vector<unsigned>& dstSkipSize, size_t deep,
    bool useOMP, Func func) {
  unsigned total = dstShape[deep];
  if (deep == dstShape.size() - 1) {
    if (useOMP) {
#pragma omp parallel for
      for (unsigned i = 0; i < total; ++i) {
        dst[i] = func(dst[i], src[i * srcSkipSize[deep]]);
      }
    } else {
      for (unsigned i = 0; i < total; ++i) {
        dst[i] = func(dst[i], src[i * srcSkipSize[deep]]);
      }
    }
    return;
  }
  if (useOMP) {
#pragma omp parallel for
    for (unsigned i = 0; i < total; ++i) {
      _mkl_real_broadcast(
          src + srcSkipSize[deep] * i, dst + dstSkipSize[deep] * i, dstShape, srcSkipSize,
          dstSkipSize, deep + 1, false, func);
    }
  } else {
    for (unsigned i = 0; i < total; ++i) {
      _mkl_real_broadcast(
          src + srcSkipSize[deep] * i, dst + dstSkipSize[deep] * i, dstShape, srcSkipSize,
          dstSkipSize, deep + 1, false, func);
    }
  }
}

template <class T, class Func>
static inline void _mkl_broadcast(
    const T* src, T* dst, const std::vector<unsigned>& dstShape,
    const std::vector<bool>& broadcastMasks, Func func) {
  std::vector<unsigned> srcSkipSize(dstShape.size(), 0);
  std::vector<unsigned> dstSkipSize(dstShape.size(), 0);

  size_t srcCurrentSkip = 1;
  size_t dstCurrentSkip = 1;
  for (size_t i = 0; i < dstShape.size(); ++i) {
    srcSkipSize[dstShape.size() - i - 1] = srcCurrentSkip;
    dstSkipSize[dstShape.size() - i - 1] = dstCurrentSkip;

    srcCurrentSkip *=
        broadcastMasks[dstShape.size() - 1 - i] ? 1 : dstShape[dstShape.size() - 1 - i];
    dstCurrentSkip *= dstShape[dstShape.size() - 1 - i];
  }
  for (size_t i = 0; i < dstShape.size(); ++i) {
    if (broadcastMasks[i]) {
      srcSkipSize[i] = 0;
    }
  }

  _mkl_real_broadcast(src, dst, dstShape, srcSkipSize, dstSkipSize, 0, true, func);
}

template <class T, class Func>
static inline void _mkl_calculation_op(const T* a, const T* b, T* c, unsigned total, Func func) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; ++i) {
    c[i] = func(a[i], b[i]);
  }
}

template <class T, class Func, class BFunc>
static inline Variable<T> _mkl_process_broadcast_op(
    const Tensor<T>& a, const Tensor<T>& b, Func opFunc, BFunc batchFunc, std::string opName) {
  if (a.shape().size() != b.shape().size()) {
    throw InvalidArgumentError("tensor broadcast error");
  }
  std::vector<unsigned> s;
  std::vector<bool> aMasks, bMasks;
  for (size_t i = 0; i < a.shape().size(); ++i) {
    if (a.shape(i) != b.shape(i) && std::min(a.shape(i), b.shape(i)) != 1) {
      throw InvalidArgumentError("tensor broadcast error");
    }
    s.emplace_back(std::max(a.shape(i), b.shape(i)));
    aMasks.push_back(a.shape(i) != std::max(a.shape(i), b.shape(i)));
    bMasks.push_back(b.shape(i) != std::max(a.shape(i), b.shape(i)));
  }
  Shape resultShape(std::move(s));

  // Two broadcast
  if (a.shape() != resultShape && b.shape() != resultShape) {
    Variable<T> result(resultShape);
    mkl_async_wrapper(
        "broadcast",
        [](const T* src, T* dst, std::vector<unsigned> dstShape, std::vector<bool> broadcastMasks) {
          _mkl_broadcast(src, dst, dstShape, broadcastMasks, [](T dst, T src) { return src; });
        },
        a.data(), result.data(), result.shape().toVector(), aMasks);
    mkl_async_wrapper(
        "broadcast_" + opName,
        [](const T* src, T* dst, std::vector<unsigned> dstShape, std::vector<bool> broadcastMasks,
           Func func) { _mkl_broadcast(src, dst, dstShape, broadcastMasks, func); },
        b.data(), result.data(), result.shape().toVector(), bMasks, opFunc);
    return result;
  }

  Variable<T> tmpA;
  Variable<T> tmpB;
  Variable<T> tmpC;

  View<T> va;
  View<T> vb;

  if (a.shape() != resultShape) {
    tmpA = Variable<T>(resultShape);
    mkl_async_wrapper(
        "broadcast_" + opName,
        [](const T* src, T* dst, std::vector<unsigned> dstShape, std::vector<bool> broadcastMasks) {
          _mkl_broadcast(src, dst, dstShape, broadcastMasks, [](T dst, T src) { return src; });
        },
        a.data(), tmpA.data(), tmpA.shape().toVector(), aMasks);
    va = View<T>(tmpA);
  } else {
    va = View<T>(a);
  }

  if (b.shape() != resultShape) {
    tmpB = Variable<T>(resultShape);
    mkl_async_wrapper(
        "broadcast_" + opName,
        [](const T* src, T* dst, std::vector<unsigned> dstShape, std::vector<bool> broadcastMasks) {
          _mkl_broadcast(src, dst, dstShape, broadcastMasks, [](T dst, T src) { return src; });
        },
        b.data(), tmpB.data(), tmpB.shape().toVector(), bMasks);
    vb = View<T>(tmpB);
  } else {
    vb = View<T>(b);
  }

  if (a.shape() == resultShape && b.shape() == resultShape) {
    tmpC = Variable<T>(resultShape);
  }

  Variable<T>& result =
      tmpA.shape() == resultShape ? tmpA : tmpB.shape() == resultShape ? tmpB : tmpC;

  mkl_async_wrapper(
      std::move(opName), batchFunc, va.data(), vb.data(), result.data(), result.size());
  return std::move(result);
}

}  // namespace tfcc
