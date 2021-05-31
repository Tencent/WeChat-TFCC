

#include "tfcc_cudanormalizationinterface.h"

#include <type_traits>

#include "exceptions/tfcc_cudaruntimeerror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "exceptions/tfcc_runtimeerror.h"
#include "framework/tfcc_cudasession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "kernel/tfcc_cudaatomickernel.hpp"
#include "kernel/tfcc_cudareducekernel.hpp"

namespace tfcc {

// cuda functions
template <class T, unsigned THREAD_COUNT>
static void __global__ _cuda_layer_normalize_small(const T* a, const T* gamma, const T* beta, T epsilon, unsigned chunk, T* b) {
  // assume chunk <= THREAD_COUNT
  a += blockIdx.x * chunk;
  b += blockIdx.x * chunk;

  __shared__ T meanShared;
  __shared__ T invShared;

  unsigned int tid = threadIdx.x;

  T input = tid < chunk ? a[tid] : static_cast<T>(0);
  T mean = _cuda_reduce_sum<T, THREAD_COUNT>(input) / static_cast<T>(chunk);
  if (tid == 0)
    meanShared = mean;
  __syncthreads();
  mean = meanShared;
  T square = tid < chunk ? (input - mean) * (input - mean) : static_cast<T>(0);
  T variance = _cuda_reduce_sum<T, THREAD_COUNT>(square) / static_cast<T>(chunk);
  T inv;
  if (tid == 0) {
    inv = rsqrt(variance + epsilon);
    invShared = inv;
  }
  __syncthreads();
  inv = invShared;
  if (tid < chunk) {
    T x = inv * gamma[tid];
    b[tid] = (input - mean) * x + beta[tid];
  }
}

#define LAYER_NORM_CASE(NUM)                                                                                 \
  case NUM:                                                                                                  \
    while (offset < batchSize) {                                                                             \
      _cuda_layer_normalize_small<T, NUM><<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>( \
          a.data() + chunkSize * offset, gamma.data() + offset, beta.data() + offset,                        \
          epsilon,                                                                                           \
          chunkSize,                                                                                         \
          result.data() + chunkSize * offset);                                                               \
      offset += blockCount;                                                                                  \
    }                                                                                                        \
    break

#define LAYER_NORM_CASE_2(NUM1, NUM2) \
  LAYER_NORM_CASE(NUM1);              \
  LAYER_NORM_CASE(NUM2)

#define LAYER_NORM_CASE_4(NUM1, NUM2, NUM3, NUM4) \
  LAYER_NORM_CASE_2(NUM1, NUM2);                  \
  LAYER_NORM_CASE_2(NUM3, NUM4)

#define LAYER_NORM_CASE_8(NUM1, NUM2, NUM3, NUM4, NUM5, NUM6, NUM7, NUM8) \
  LAYER_NORM_CASE_4(NUM1, NUM2, NUM3, NUM4);                              \
  LAYER_NORM_CASE_4(NUM5, NUM6, NUM7, NUM8)

#define LAYER_NORM_SWITCH(THREADCNT)                                                           \
  unsigned offset = 0;                                                                         \
  switch (THREADCNT) {                                                                         \
    LAYER_NORM_CASE_8(32 * 1, 32 * 2, 32 * 3, 32 * 4, 32 * 5, 32 * 6, 32 * 7, 32 * 8);         \
    LAYER_NORM_CASE_8(32 * 9, 32 * 10, 32 * 11, 32 * 12, 32 * 13, 32 * 14, 32 * 15, 32 * 16);  \
    LAYER_NORM_CASE_8(32 * 17, 32 * 18, 32 * 19, 32 * 20, 32 * 21, 32 * 22, 32 * 23, 32 * 24); \
    LAYER_NORM_CASE_8(32 * 25, 32 * 26, 32 * 27, 32 * 28, 32 * 29, 32 * 30, 32 * 31, 32 * 32); \
  }

template <class T, unsigned THREAD_COUNT>
static void __global__ _cuda_layer_normalize_large(const T* a, const T* gamma, const T* beta, T epsilon, unsigned chunk, T* b) {
  // assume chunk > THREAD_COUNT
  a += blockIdx.x * chunk;
  b += blockIdx.x * chunk;

  __shared__ T meanShared;
  __shared__ T invShared;

  unsigned int tid = threadIdx.x;
  T inputSum = a[tid];
  for (unsigned i = tid + blockDim.x; i < chunk; i += blockDim.x)
    inputSum += a[i];

  T mean = _cuda_reduce_sum<T, THREAD_COUNT>(inputSum) / static_cast<T>(chunk);
  if (tid == 0)
    meanShared = mean;
  __syncthreads();
  mean = meanShared;

  T squareSum = (a[tid] - mean) * (a[tid] - mean);
  for (unsigned i = tid + blockDim.x; i < chunk; i += blockDim.x)
    squareSum += (a[i] - mean) * (a[i] - mean);

  T variance = _cuda_reduce_sum<T, THREAD_COUNT>(squareSum) / static_cast<T>(chunk);
  T inv;
  if (tid == 0) {
    inv = rsqrt(variance + epsilon);
    invShared = inv;
  }
  __syncthreads();
  inv = invShared;

  for (unsigned i = tid; i < chunk; i += blockDim.x) {
    T x = inv * gamma[i];
    b[i] = (a[i] - mean) * x + beta[i];
  }
}

// helper functions
template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value, Variable<T>>::type
_layer_normalize_helper(const Tensor<T>& a, const Tensor<T>& gamma, const Tensor<T>& beta, T epsilon, size_t beginNormAxis, size_t blockCount, size_t threadCount) {
  unsigned batchSize = 1;
  unsigned chunkSize = 1;
  for (size_t i = 0; i < beginNormAxis; ++i)
    batchSize *= a.shape(i);
  for (size_t i = beginNormAxis; i < a.shape().size(); ++i)
    chunkSize *= a.shape(i);

  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());

  if (chunkSize > threadCount) {
    if (threadCount != 512 && threadCount != 1024)
      throw RuntimeError("Unexpect thread count");
    if (threadCount == 512) {
      unsigned offset = 0;
      while (offset < batchSize) {
        _cuda_layer_normalize_large<T, 512><<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
            a.data() + chunkSize * offset, gamma.data() + offset, beta.data() + offset,
            epsilon,
            chunkSize,
            result.data() + chunkSize * offset);
        offset += blockCount;
      }
    } else {
      unsigned offset = 0;
      while (offset < batchSize) {
        _cuda_layer_normalize_large<T, 1024><<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
            a.data() + chunkSize * offset, gamma.data() + offset, beta.data() + offset,
            epsilon,
            chunkSize,
            result.data() + chunkSize * offset);
        offset += blockCount;
      }
    }

  } else {
    LAYER_NORM_SWITCH(threadCount);
  }

  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _layer_normalize_helper(const Tensor<T>& a, const Tensor<T>& gamma, const Tensor<T>& beta, T epsilon, size_t beginNormAxis, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

// class function
template <class T>
CUDANormalizationInterface<T>::CUDANormalizationInterface(const CUDADeviceProperty& property)
  : _property(property) {
}

template <class T>
CUDANormalizationInterface<T>::~CUDANormalizationInterface() {
}

template <class T>
Variable<T> CUDANormalizationInterface<T>::layerNormalize(const Tensor<T>& a, const Tensor<T>& gamma, const Tensor<T>& beta, T epsilon, size_t beginNormAxis) {
  unsigned batchSize = 1;
  unsigned chunkSize = 1;
  for (size_t i = 0; i < beginNormAxis; ++i)
    batchSize *= a.shape(i);
  for (size_t i = beginNormAxis; i < a.shape().size(); ++i)
    chunkSize *= a.shape(i);

  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(batchSize, chunkSize);
  threadCount = (threadCount + 31) / 32 * 32;
  threadCount = std::min(threadCount, 1024lu);

  return _layer_normalize_helper(a, gamma, beta, epsilon, beginNormAxis, blockCount, threadCount);
}

#define DEFINE_FUNC(type) template class CUDANormalizationInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
