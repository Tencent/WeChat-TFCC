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

#include "kernel/tfcc_cudacommonkernel.hpp"

namespace tfcc {

template <class T, unsigned THREAD_COUNT>
static inline T __device__ _cuda_reduce_sum(T val) {
  static_assert(
      THREAD_COUNT >= 32 && THREAD_COUNT <= 1024 && THREAD_COUNT % 32 == 0, "invalid thread count");
  __shared__ T sdata[32];
  T result = val;
  unsigned int tid = threadIdx.x;
  if (THREAD_COUNT >= 32) {
    result += __shfl_down_sync(0xffffffff, result, 16);
    result += __shfl_down_sync(0xffffffff, result, 8);
    result += __shfl_down_sync(0xffffffff, result, 4);
    result += __shfl_down_sync(0xffffffff, result, 2);
    result += __shfl_down_sync(0xffffffff, result, 1);
    if (THREAD_COUNT > 32 && tid % 32 == 0) sdata[tid / 32] = result;
    __syncthreads();
  }
  if (THREAD_COUNT == 32) return result;
  constexpr unsigned L2_COUNT = THREAD_COUNT >= 32 ? THREAD_COUNT / 32 : THREAD_COUNT;

  if (tid < 32) {
    if (THREAD_COUNT > 32) result = tid < L2_COUNT ? sdata[tid] : static_cast<T>(0);

    if (L2_COUNT > 16) result += __shfl_down_sync(0xffffffff, result, 16);
    if (L2_COUNT > 8) result += __shfl_down_sync(0xffffffff, result, 8);
    if (L2_COUNT > 4) result += __shfl_down_sync(0xffffffff, result, 4);
    if (L2_COUNT > 2) result += __shfl_down_sync(0xffffffff, result, 2);
    if (L2_COUNT > 1) result += __shfl_down_sync(0xffffffff, result, 1);
  }
  return result;
}

template <class T, unsigned THREAD_COUNT>
static inline T __device__ _cuda_reduce_max(T val) {
  static_assert(
      THREAD_COUNT >= 32 && THREAD_COUNT <= 1024 && THREAD_COUNT % 32 == 0, "invalid thread count");
  __shared__ T sdata[32];
  T result = val;
  unsigned int tid = threadIdx.x;
  if (THREAD_COUNT >= 32) {
    result = _cuda_max(result, __shfl_down_sync(0xffffffff, result, 16));
    result = _cuda_max(result, __shfl_down_sync(0xffffffff, result, 8));
    result = _cuda_max(result, __shfl_down_sync(0xffffffff, result, 4));
    result = _cuda_max(result, __shfl_down_sync(0xffffffff, result, 2));
    result = _cuda_max(result, __shfl_down_sync(0xffffffff, result, 1));
    if (THREAD_COUNT > 32 && tid % 32 == 0) sdata[tid / 32] = result;
    __syncthreads();
  }
  if (THREAD_COUNT == 32) return result;
  constexpr unsigned L2_COUNT = THREAD_COUNT >= 32 ? THREAD_COUNT / 32 : THREAD_COUNT;

  if (tid < 32) {
    if (THREAD_COUNT > 32) result = tid < L2_COUNT ? sdata[tid] : static_cast<T>(0);

    if (L2_COUNT > 16) result = _cuda_max(result, __shfl_down_sync(0xffffffff, result, 16));
    if (L2_COUNT > 8) result = _cuda_max(result, __shfl_down_sync(0xffffffff, result, 8));
    if (L2_COUNT > 4) result = _cuda_max(result, __shfl_down_sync(0xffffffff, result, 4));
    if (L2_COUNT > 2) result = _cuda_max(result, __shfl_down_sync(0xffffffff, result, 2));
    if (L2_COUNT > 1) result = _cuda_max(result, __shfl_down_sync(0xffffffff, result, 1));
  }
  return result;
}

}  // namespace tfcc
