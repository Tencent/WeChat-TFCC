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

#include <vector>

#include "exceptions/tfcc_cudaruntimeerror.h"
#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"

namespace tfcc {

template <class T, class F>
static __global__ void _cuda_nobroadcast_op(const T* a, const T* b, T* c, unsigned total) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    c[i] = F::process(a[i], b[i]);
  }
}

template <class T, class F>
static __global__ void _cuda_broadcast_op_dim1(
    const T* a, const T* b, T* c, unsigned as1, unsigned bs1, unsigned cs1) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  const unsigned total = cs1;

  for (unsigned i = tid; i < total; i += skip) {
    unsigned s1 = i;

    unsigned ai = (as1 == 1 ? 0 : s1);
    unsigned bi = (bs1 == 1 ? 0 : s1);
    c[i] = F::process(a[ai], b[bi]);
  }
}

template <class T, class F>
static __global__ void _cuda_broadcast_op_dim2(
    const T* a, const T* b, T* c, unsigned as1, unsigned as2, unsigned bs1, unsigned bs2,
    unsigned cs1, unsigned cs2) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  const unsigned total = cs1 * cs2;

  for (unsigned i = tid; i < total; i += skip) {
    unsigned s1 = (i / cs2) % cs1;
    unsigned s2 = i % cs2;

    unsigned ai = (as1 == 1 ? 0 : s1) * as2 + (as2 == 1 ? 0 : s2);
    unsigned bi = (bs1 == 1 ? 0 : s1) * bs2 + (bs2 == 1 ? 0 : s2);
    c[i] = F::process(a[ai], b[bi]);
  }
}

template <class T, class F>
static __global__ void _cuda_broadcast_op_dim3(
    const T* a, const T* b, T* c, unsigned as1, unsigned as2, unsigned as3, unsigned bs1,
    unsigned bs2, unsigned bs3, unsigned cs1, unsigned cs2, unsigned cs3) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  const unsigned total = cs1 * cs2 * cs3;

  for (unsigned i = tid; i < total; i += skip) {
    unsigned s1 = (i / (cs2 * cs3)) % cs1;
    unsigned s2 = (i / cs3) % cs2;
    unsigned s3 = i % cs3;

    unsigned ai = (as1 == 1 ? 0 : s1) * as2 * as3 + (as2 == 1 ? 0 : s2) * as3 + (as3 == 1 ? 0 : s3);
    unsigned bi = (bs1 == 1 ? 0 : s1) * bs2 * bs3 + (bs2 == 1 ? 0 : s2) * bs3 + (bs3 == 1 ? 0 : s3);
    c[i] = F::process(a[ai], b[bi]);
  }
}

template <class T, class F>
static __global__ void _cuda_broadcast_op_dim4(
    const T* a, const T* b, T* c, unsigned as1, unsigned as2, unsigned as3, unsigned as4,
    unsigned bs1, unsigned bs2, unsigned bs3, unsigned bs4, unsigned cs1, unsigned cs2,
    unsigned cs3, unsigned cs4) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  const unsigned total = cs1 * cs2 * cs3 * cs4;

  for (unsigned i = tid; i < total; i += skip) {
    unsigned s1 = (i / (cs2 * cs3 * cs4)) % cs1;
    unsigned s2 = (i / (cs3 * cs4)) % cs2;
    unsigned s3 = (i / cs4) % cs3;
    unsigned s4 = i % cs4;

    unsigned ai = (as1 == 1 ? 0 : s1) * as2 * as3 * as4 + (as2 == 1 ? 0 : s2) * as3 * as4 +
                  (as3 == 1 ? 0 : s3) * as4 + (as4 == 1 ? 0 : s4);
    unsigned bi = (bs1 == 1 ? 0 : s1) * bs2 * bs3 * bs4 + (bs2 == 1 ? 0 : s2) * bs3 * bs4 +
                  (bs3 == 1 ? 0 : s3) * bs4 + (bs4 == 1 ? 0 : s4);
    c[i] = F::process(a[ai], b[bi]);
  }
}

template <class T, class F>
static __global__ void _cuda_broadcast_op_dim5(
    const T* a, const T* b, T* c, unsigned as1, unsigned as2, unsigned as3, unsigned as4,
    unsigned as5, unsigned bs1, unsigned bs2, unsigned bs3, unsigned bs4, unsigned bs5,
    unsigned cs1, unsigned cs2, unsigned cs3, unsigned cs4, unsigned cs5) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  const unsigned total = cs1 * cs2 * cs3 * cs4 * cs5;

  for (unsigned i = tid; i < total; i += skip) {
    unsigned s1 = (i / (cs2 * cs3 * cs4 * cs5)) % cs1;
    unsigned s2 = (i / (cs3 * cs4 * cs5)) % cs2;
    unsigned s3 = (i / (cs4 * cs5)) % cs3;
    unsigned s4 = (i / cs5) % cs4;
    unsigned s5 = i % cs5;

    unsigned ai = (as1 == 1 ? 0 : s1) * as2 * as3 * as4 * as5 +
                  (as2 == 1 ? 0 : s2) * as3 * as4 * as5 + (as3 == 1 ? 0 : s3) * as4 * as5 +
                  (as4 == 1 ? 0 : s4) * as5 + (as5 == 1 ? 0 : s5);
    unsigned bi = (bs1 == 1 ? 0 : s1) * bs2 * bs3 * bs4 * bs5 +
                  (bs2 == 1 ? 0 : s2) * bs3 * bs4 * bs5 + (bs3 == 1 ? 0 : s3) * bs4 * bs5 +
                  (bs4 == 1 ? 0 : s4) * bs5 + (bs5 == 1 ? 0 : s5);
    c[i] = F::process(a[ai], b[bi]);
  }
}

template <class T, class F>
static inline Variable<T> _process_can_broadcast_op(
    const Tensor<T>& a, const Tensor<T>& b, CUDADeviceProperty& property) {
  if (a.shape().size() != b.shape().size()) throw InvalidArgumentError("tensor broadcast error");
  bool isSame = a.shape() == b.shape();
  if (a.shape().size() > 5 && isSame) {
    throw NotImplementedError();
  }
  std::vector<unsigned> s;
  for (size_t i = 0; i < a.shape().size(); ++i) s.emplace_back(std::max(a.shape(i), b.shape(i)));
  Variable<T> result(std::move(s));

  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = property.getSuitableKernelSize(result.size());

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  if (isSame) {
    _cuda_nobroadcast_op<T, F><<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
        a.data(), b.data(), result.data(), result.size());
  } else {
    switch (result.shape().size()) {
      case 1:
        _cuda_broadcast_op_dim1<T, F>
            <<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
                a.data(), b.data(), result.data(), a.shape(0), b.shape(0), result.shape(0));
        break;
      case 2:
        _cuda_broadcast_op_dim2<T, F>
            <<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
                a.data(), b.data(), result.data(), a.shape(0), a.shape(1), b.shape(0), b.shape(1),
                result.shape(0), result.shape(1));
        break;
      case 3:
        _cuda_broadcast_op_dim3<T, F>
            <<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
                a.data(), b.data(), result.data(), a.shape(0), a.shape(1), a.shape(2), b.shape(0),
                b.shape(1), b.shape(2), result.shape(0), result.shape(1), result.shape(2));
        break;
      case 4:
        _cuda_broadcast_op_dim4<T, F>
            <<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
                a.data(), b.data(), result.data(), a.shape(0), a.shape(1), a.shape(2), a.shape(3),
                b.shape(0), b.shape(1), b.shape(2), b.shape(3), result.shape(0), result.shape(1),
                result.shape(2), result.shape(3));
        break;
      case 5:
        _cuda_broadcast_op_dim5<T, F>
            <<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
                a.data(), b.data(), result.data(), a.shape(0), a.shape(1), a.shape(2), a.shape(3),
                a.shape(4), b.shape(0), b.shape(1), b.shape(2), b.shape(3), b.shape(4),
                result.shape(0), result.shape(1), result.shape(2), result.shape(3),
                result.shape(4));
        break;
      default:
        abort();
    }
  }

  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess) throw CUDARuntimeError(ret);

  return result;
}

}  // namespace tfcc
