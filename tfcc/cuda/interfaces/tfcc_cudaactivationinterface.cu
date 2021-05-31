

#include "tfcc_cudaactivationinterface.h"

#include <limits>
#include <type_traits>

#include "exceptions/tfcc_cudaruntimeerror.h"
#include "exceptions/tfcc_cudnnruntimeerror.h"
#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_cudasession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "kernel/tfcc_cudaatomickernel.hpp"
#include "kernel/tfcc_cudareducekernel.hpp"
#include "utils/tfcc_cudnnutils.h"

namespace tfcc {

// cuda functions
template <class T>
static void __global__ _cuda_sigmoid(const T* a, unsigned total, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T v = a[i];
    b[i] = 1 / (1 + exp(-v));
  }
}

template <class T>
static void __global__ _cuda_tanh(const T* a, unsigned total, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T v = a[i];
    b[i] = tanh(v);
  }
}

template <class T>
static void __global__ _cuda_relu(const T* a, unsigned total, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T v = a[i];
    b[i] = max(v, T(0));
  }
}

template <class T>
static void __global__ _cuda_leaky_relu(const T* a, unsigned total, T alpha, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T v = a[i];
    T v2 = alpha * v;
    b[i] = max(v, v2);
  }
}

template <class T>
static void __global__ _cuda_softplus(const T* a, unsigned total, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T v = a[i];
    b[i] = log(exp(v) + 1);
  }
}

template <class T>
static void __global__ _cuda_log(const T* a, unsigned total, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T v = a[i];
    b[i] = log(v);
  }
}

template <class T>
static void __global__ _cuda_rsqrt(const T* a, unsigned total, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T v = a[i];
    b[i] = rsqrt(v);
  }
}

constexpr float lowest_my = std::numeric_limits<float>::lowest();

template <class T, unsigned THREAD_COUNT>
static void __global__ _cuda_softmax_v1_small(const T* a, unsigned chunk, T* b) {
  a += blockIdx.x * chunk;
  b += blockIdx.x * chunk;

  __shared__ T maxShared;
  __shared__ T exSumShared;

  unsigned tid = threadIdx.x;
  T input = tid < chunk ? a[tid] : lowest_my;
  T maxVal = _cuda_reduce_max<T, THREAD_COUNT>(input);
  if (tid == 0)
    maxShared = maxVal;
  __syncthreads();
  maxVal = maxShared;
  T ex = tid < chunk ? exp(input - maxVal) : static_cast<T>(0);
  T exSum = _cuda_reduce_sum<T, THREAD_COUNT>(ex);
  if (tid == 0)
    exSumShared = exSum;
  __syncthreads();
  exSum = exSumShared;
  if (tid < chunk)
    b[tid] = ex / exSum;
}

#define SOFTMAX_CASE(NUM)                                                                                   \
  case NUM:                                                                                                 \
    while (offset < batchSize) {                                                                            \
      _cuda_softmax_v1_small<float, NUM><<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>( \
          a.data() + chunkSize * offset,                                                                    \
          chunkSize,                                                                                        \
          result.data() + chunkSize * offset);                                                              \
      offset += blockCount;                                                                                 \
    }                                                                                                       \
    break

#define SOFTMAX_CASE_2(NUM1, NUM2) \
  SOFTMAX_CASE(NUM1);              \
  SOFTMAX_CASE(NUM2)

#define SOFTMAX_CASE_4(NUM1, NUM2, NUM3, NUM4) \
  SOFTMAX_CASE_2(NUM1, NUM2);                  \
  SOFTMAX_CASE_2(NUM3, NUM4)

#define SOFTMAX_CASE_8(NUM1, NUM2, NUM3, NUM4, NUM5, NUM6, NUM7, NUM8) \
  SOFTMAX_CASE_4(NUM1, NUM2, NUM3, NUM4);                              \
  SOFTMAX_CASE_4(NUM5, NUM6, NUM7, NUM8)

#define SOFTMAX_SWITCH(THREADCNT)                                                           \
  unsigned offset = 0;                                                                      \
  switch (THREADCNT) {                                                                      \
    SOFTMAX_CASE_8(32 * 1, 32 * 2, 32 * 3, 32 * 4, 32 * 5, 32 * 6, 32 * 7, 32 * 8);         \
    SOFTMAX_CASE_8(32 * 9, 32 * 10, 32 * 11, 32 * 12, 32 * 13, 32 * 14, 32 * 15, 32 * 16);  \
    SOFTMAX_CASE_8(32 * 17, 32 * 18, 32 * 19, 32 * 20, 32 * 21, 32 * 22, 32 * 23, 32 * 24); \
    SOFTMAX_CASE_8(32 * 25, 32 * 26, 32 * 27, 32 * 28, 32 * 29, 32 * 30, 32 * 31, 32 * 32); \
  }

template <class T>
static void __global__ _cuda_sin(const T* a, unsigned total, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    auto v = a[i];
    b[i] = sin(v);
  }
}

template <class T>
static void __global__ _cuda_cos(const T* a, unsigned total, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T v = a[i];
    b[i] = cos(v);
  }
}

template <class T>
static void __global__ _cuda_pow(const T* a, unsigned total, T exponent, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T v = a[i];
    b[i] = pow(v, exponent);
  }
}

template <class T>
static void __global__ _cuda_pow_v2(const T* a, const T* exponent, T* b, unsigned total) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T v = a[i];
    T e = exponent[i];
    b[i] = pow(v, e);
  }
}

template <class T>
static void __global__ _cuda_pow_v3(const T* exponent, unsigned total, T a, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T e = exponent[i];
    b[i] = pow(a, e);
  }
}

template <class T>
static void __global__ _cuda_gelu(const T* a, unsigned total, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T v = a[i];
    T tmp = static_cast<T>(0.7978845608028654) * (v + 0.044715 * v * v * v);
    b[i] = v * static_cast<T>(0.5) * (static_cast<T>(1.0) + tanh(tmp));
  }
}

template <class T>
static void __global__ _cuda_gelu_accurate(const T* a, unsigned total, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T v = a[i];
    b[i] = 0.5 * v * (1 + erf(v / static_cast<T>(1.4142135623730951)));
  }
}

template <class T>
static void __global__ _cuda_erf(const T* a, unsigned total, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T v = a[i];
    b[i] = erf(v);
  }
}

template <class T>
static void __global__ _cuda_asin(const T* a, unsigned total, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T v = a[i];
    b[i] = asinf(v);
  }
}

template <class T>
static void __global__ _cuda_asinh(const T* a, unsigned total, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T v = a[i];
    b[i] = asinhf(v);
  }
}

template <class T>
static void __global__ _cuda_acos(const T* a, unsigned total, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T v = a[i];
    b[i] = acosf(v);
  }
}

template <class T>
static void __global__ _cuda_acosh(const T* a, unsigned total, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T v = a[i];
    b[i] = acoshf(v);
  }
}

template <class T>
static void __global__ _cuda_atan(const T* a, unsigned total, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T v = a[i];
    b[i] = atanf(v);
  }
}

template <class T>
static void __global__ _cuda_atanh(const T* a, unsigned total, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T v = a[i];
    b[i] = atanhf(v);
  }
}

template <class T>
static void __global__ _cuda_sign(const T* a, unsigned total, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T v = a[i];
    if (v < 0)
      b[i] = static_cast<T>(-1);
    else if (v > 0)
      b[i] = static_cast<T>(1);
    else
      b[i] = 0;
  }
}

// helper functions
template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type
_sigmoid_helper(const Tensor<T>& a, size_t blockCount, size_t threadCount) {
  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_sigmoid<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _sigmoid_helper(const Tensor<T>& a, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type
_tanh_helper(const Tensor<T>& a, size_t blockCount, size_t threadCount) {
  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_tanh<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _tanh_helper(const Tensor<T>& a, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type
_relu_helper(const Tensor<T>& a, size_t blockCount, size_t threadCount) {
  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_relu<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _relu_helper(const Tensor<T>& a, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type
_leaky_relu_helper(const Tensor<T>& a, T alpha, size_t blockCount, size_t threadCount) {
  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_leaky_relu<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      alpha,
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _leaky_relu_helper(const Tensor<T>& a, T alpha, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type
_softplus_helper(const Tensor<T>& a, size_t blockCount, size_t threadCount) {
  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_softplus<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _softplus_helper(const Tensor<T>& a, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type
_log_helper(const Tensor<T>& a, size_t blockCount, size_t threadCount) {
  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_log<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _log_helper(const Tensor<T>& a, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type
_rsqrt_helper(const Tensor<T>& a, size_t blockCount, size_t threadCount) {
  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_rsqrt<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _rsqrt_helper(const Tensor<T>& a, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline void _cudnn_softmax(cudnnDataType_t cudnnType, unsigned s1, unsigned s2, unsigned s3, const T* a, T* b) {
  cudnnTensorDescriptor_t aTensor;
  cudnnStatus_t ret = cudnnCreateTensorDescriptor(&aTensor);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);
  CudnnTensorDescriptorGuard aTensorGuard(&aTensor);

  cudnnSoftmaxMode_t softmaxModel;
  if (s3 == 1) {
    ret = cudnnSetTensor4dDescriptor(aTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, s1, 1, 1, s2);
    softmaxModel = CUDNN_SOFTMAX_MODE_INSTANCE;
  } else {
    ret = cudnnSetTensor4dDescriptor(aTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, s1, s2, s3, 1);
    softmaxModel = CUDNN_SOFTMAX_MODE_CHANNEL;
  }
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  T alpha = 1.0, beta = 0.0;
  ret = cudnnSoftmaxForward(
      session->getImpl()->cudnnHandle(),
      CUDNN_SOFTMAX_ACCURATE,
      softmaxModel,
      &alpha,
      aTensor, a,
      &beta,
      aTensor, b);
  if (ret != CUDNN_STATUS_SUCCESS)
    throw CUDNNRuntimeError(ret);
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type
_sin_helper(const Tensor<T>& a, size_t blockCount, size_t threadCount) {
  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_sin<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _sin_helper(const Tensor<T>& a, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type
_cos_helper(const Tensor<T>& a, size_t blockCount, size_t threadCount) {
  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_cos<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _cos_helper(const Tensor<T>& a, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type
_pow_helper(const Tensor<T>& a, T exponent, size_t blockCount, size_t threadCount) {
  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_pow<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      exponent,
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _pow_helper(const Tensor<T>& a, T exponent, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type
_pow_helper(const Tensor<T>& a, const Tensor<T>& exponent, size_t blockCount, size_t threadCount) {
  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_pow_v2<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), exponent.data(),
      result.data(),
      result.size());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _pow_helper(const Tensor<T>& a, const Tensor<T>& exponent, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type
_pow_helper(T a, const Tensor<T>& exponent, size_t blockCount, size_t threadCount) {
  Variable<T> result(exponent.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_pow_v3<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      exponent.data(), exponent.size(),
      a,
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _pow_helper(T a, const Tensor<T>& exponent, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type
_gelu_helper(const Tensor<T>& a, size_t blockCount, size_t threadCount) {
  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_gelu<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _gelu_helper(const Tensor<T>& a, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type
_gelu_accurate_helper(const Tensor<T>& a, size_t blockCount, size_t threadCount) {
  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_gelu_accurate<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _gelu_accurate_helper(const Tensor<T>& a, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type
_erf_helper(const Tensor<T>& a, size_t blockCount, size_t threadCount) {
  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_erf<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _erf_helper(const Tensor<T>& a, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type
_asin_helper(const Tensor<T>& a, size_t blockCount, size_t threadCount) {
  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_asin<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _asin_helper(const Tensor<T>& a, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type
_asinh_helper(const Tensor<T>& a, size_t blockCount, size_t threadCount) {
  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_asinh<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _asinh_helper(const Tensor<T>& a, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type
_acos_helper(const Tensor<T>& a, size_t blockCount, size_t threadCount) {
  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_acos<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _acos_helper(const Tensor<T>& a, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type
_acosh_helper(const Tensor<T>& a, size_t blockCount, size_t threadCount) {
  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_acosh<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _acosh_helper(const Tensor<T>& a, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type
_atan_helper(const Tensor<T>& a, size_t blockCount, size_t threadCount) {
  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_atan<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _atan_helper(const Tensor<T>& a, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type
_atanh_helper(const Tensor<T>& a, size_t blockCount, size_t threadCount) {
  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_atanh<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _atanh_helper(const Tensor<T>& a, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type
_sign_helper(const Tensor<T>& a, size_t blockCount, size_t threadCount) {
  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_sign<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _sign_helper(const Tensor<T>& a, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

// class function
template <class T>
CUDAActivationInterface<T>::CUDAActivationInterface(const CUDADeviceProperty& property)
  : _property(property) {
}

template <class T>
CUDAActivationInterface<T>::~CUDAActivationInterface() {
}

template <class T>
Variable<T> CUDAActivationInterface<T>::sigmoid(const Tensor<T>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _sigmoid_helper(a, blockCount, threadCount);
}

template <class T>
Variable<T> CUDAActivationInterface<T>::tanh(const Tensor<T>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _tanh_helper(a, blockCount, threadCount);
}

template <class T>
Variable<T> CUDAActivationInterface<T>::relu(const Tensor<T>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _relu_helper(a, blockCount, threadCount);
}

template <class T>
Variable<T> CUDAActivationInterface<T>::leakyRelu(const Tensor<T>& a, T alpha) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _leaky_relu_helper(a, alpha, blockCount, threadCount);
}

template <class T>
Variable<T> CUDAActivationInterface<T>::softplus(const Tensor<T>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _softplus_helper(a, blockCount, threadCount);
}

template <class T>
Variable<T> CUDAActivationInterface<T>::log(const Tensor<T>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _log_helper(a, blockCount, threadCount);
}

template <class T>
Variable<T> CUDAActivationInterface<T>::rsqrt(const Tensor<T>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _rsqrt_helper(a, blockCount, threadCount);
}

template <class T>
Variable<T> CUDAActivationInterface<T>::softmax(const Tensor<T>& a, size_t axis) {
  throw NotImplementedError();
}

template <>
Variable<float> CUDAActivationInterface<float>::softmax(const Tensor<float>& a, size_t axis) {
  Variable<float> result(a.shape());
  unsigned s1 = 1, s2 = 1, s3 = 1;
  s2 = a.shape(axis);
  for (size_t i = 0; i < axis; ++i)
    s1 *= a.shape(i);
  for (size_t i = axis + 1; i < a.shape().size(); ++i)
    s3 *= a.shape(i);

  cudnnDataType_t cudnnDType = CUDNN_DATA_FLOAT;

  if (s3 != 1 || s2 > 512) {
    _cudnn_softmax(cudnnDType, s1, s2, s3, a.data(), result.data());
    return result;
  }
  unsigned batchSize = s1;
  unsigned chunkSize = s2;

  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(batchSize, chunkSize);
  threadCount = (threadCount + 31) / 32 * 32;
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());

  if (chunkSize > threadCount) {
    _cudnn_softmax(cudnnDType, s1, s2, s3, a.data(), result.data());
    return result;
  }

  SOFTMAX_SWITCH(threadCount);

  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <>
Variable<double> CUDAActivationInterface<double>::softmax(const Tensor<double>& a, size_t axis) {
  Variable<double> result(a.shape());
  unsigned s1 = 1, s2 = 1, s3 = 1;
  s2 = a.shape(axis);
  for (size_t i = 0; i < axis; ++i)
    s1 *= a.shape(i);
  for (size_t i = axis + 1; i < a.shape().size(); ++i)
    s3 *= a.shape(i);

  cudnnDataType_t cudnnDType = CUDNN_DATA_DOUBLE;

  _cudnn_softmax(cudnnDType, s1, s2, s3, a.data(), result.data());
  return result;
}

template <class T>
Variable<T> CUDAActivationInterface<T>::sin(const Tensor<T>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _sin_helper(a, blockCount, threadCount);
}

template <class T>
Variable<T> CUDAActivationInterface<T>::cos(const Tensor<T>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _cos_helper(a, blockCount, threadCount);
}

template <class T>
Variable<T> CUDAActivationInterface<T>::pow(const Tensor<T>& a, T exponent) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _pow_helper(a, exponent, blockCount, threadCount);
}

template <class T>
Variable<T> CUDAActivationInterface<T>::pow(const Tensor<T>& a, const Tensor<T>& exponent) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _pow_helper(a, exponent, blockCount, threadCount);
}

template <class T>
Variable<T> CUDAActivationInterface<T>::pow(T a, const Tensor<T>& exponent) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(exponent.size());
  return _pow_helper(a, exponent, blockCount, threadCount);
}

template <class T>
Variable<T> CUDAActivationInterface<T>::gelu(const Tensor<T>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _gelu_helper(a, blockCount, threadCount);
}

template <class T>
Variable<T> CUDAActivationInterface<T>::geluAccurate(const Tensor<T>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _gelu_accurate_helper(a, blockCount, threadCount);
}

template <class T>
Variable<T> CUDAActivationInterface<T>::erf(const Tensor<T>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _erf_helper(a, blockCount, threadCount);
}

template <class T>
Variable<T> CUDAActivationInterface<T>::asin(const Tensor<T>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _asin_helper(a, blockCount, threadCount);
}

template <class T>
Variable<T> CUDAActivationInterface<T>::asinh(const Tensor<T>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _asinh_helper(a, blockCount, threadCount);
}

template <class T>
Variable<T> CUDAActivationInterface<T>::acos(const Tensor<T>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _acos_helper(a, blockCount, threadCount);
}

template <class T>
Variable<T> CUDAActivationInterface<T>::acosh(const Tensor<T>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _acosh_helper(a, blockCount, threadCount);
}

template <class T>
Variable<T> CUDAActivationInterface<T>::atan(const Tensor<T>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _atan_helper(a, blockCount, threadCount);
}

template <class T>
Variable<T> CUDAActivationInterface<T>::atanh(const Tensor<T>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _atanh_helper(a, blockCount, threadCount);
}

template <class T>
Variable<T> CUDAActivationInterface<T>::sign(const Tensor<T>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _sign_helper(a, blockCount, threadCount);
}

#define DEFINE_FUNC(type) template class CUDAActivationInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
