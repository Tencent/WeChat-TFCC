

#include "tfcc_cudablasinterface.h"

#include <iostream>
#include "exceptions/tfcc_cublasruntimeerror.h"
#include "exceptions/tfcc_cudaruntimeerror.h"
#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_cudadevice.h"
#include "framework/tfcc_cudasession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "utils/tfcc_debugutils.h"

namespace tfcc {

template <class T>
static __global__ void _cuda_broadcast_bias(const T* a, unsigned batch, unsigned chunk, T* b) {
  for (unsigned i = blockIdx.x; i < batch; i += gridDim.x) {
    T* rb = b + i * chunk;
    for (unsigned j = threadIdx.x; j < chunk; j += blockDim.x) {
      rb[j] = a[j];
    }
  }
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value, Variable<T>>::type
_broadcast_bias_helper(const Tensor<T>& a, const Shape& shape, size_t blockCount, size_t threadCount) {
  Variable<T> result(shape);
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_broadcast_bias<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(),
      result.size() / a.size(),
      a.size(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _broadcast_bias_helper(const Tensor<T>& a, const Shape& shape, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

// cublas matmul
#ifdef TFCC_USE_TENSOR_CORE
template <class T>
inline void cuBlasMatmul(
    cublasHandle_t handle,
    const T* a,
    const T* b,
    T* c,
    unsigned m, unsigned n, unsigned k,
    T alpha, T beta) {
  throw NotImplementedError();
}

inline void cuBlasMatmul(
    cublasHandle_t handle,
    const float* a,
    const float* b,
    float* c,
    unsigned m, unsigned n, unsigned k,
    float alpha, float beta) {
  cublasMath_t mode;
  cublasStatus_t stat = cublasGetMathMode(handle, &mode);
  if (stat != CUBLAS_STATUS_SUCCESS)
    throw CUBlasRuntimeError(stat);
  stat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  if (stat != CUBLAS_STATUS_SUCCESS)
    throw CUBlasRuntimeError(stat);

  stat = cublasGemmEx(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      m, n, k,
      &alpha,
      b, CUDA_R_32F, m,
      a, CUDA_R_32F, k,
      &beta,
      c, CUDA_R_32F, m,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT);
  if (stat != CUBLAS_STATUS_SUCCESS)
    throw CUBlasRuntimeError(stat);

  stat = cublasSetMathMode(handle, mode);
  if (stat != CUBLAS_STATUS_SUCCESS)
    throw CUBlasRuntimeError(stat);
}

inline void cuBlasMatmul(
    cublasHandle_t handle,
    const double* a,
    const double* b,
    double* c,
    unsigned m, unsigned n, unsigned k,
    double alpha, double beta) {
  cublasMath_t mode;
  cublasStatus_t stat = cublasGetMathMode(handle, &mode);
  if (stat != CUBLAS_STATUS_SUCCESS)
    throw CUBlasRuntimeError(stat);
  stat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  if (stat != CUBLAS_STATUS_SUCCESS)
    throw CUBlasRuntimeError(stat);

  stat = cublasGemmEx(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      m, n, k,
      &alpha,
      b, CUDA_R_64F, m,
      a, CUDA_R_64F, k,
      &beta,
      c, CUDA_R_64F, m,
      CUDA_R_64F,
      CUBLAS_GEMM_DEFAULT);
  if (stat != CUBLAS_STATUS_SUCCESS)
    throw CUBlasRuntimeError(stat);

  stat = cublasSetMathMode(handle, mode);
  if (stat != CUBLAS_STATUS_SUCCESS)
    throw CUBlasRuntimeError(stat);
}
#endif

template <class T>
inline void cuBlasMatmulBatched(
    cublasHandle_t handle,
    const T* a, unsigned strideA,
    const T* b, unsigned strideB,
    T* c, unsigned strideC,
    unsigned m, unsigned n, unsigned k,
    unsigned batchCount,
    T alpha, T beta) {
  throw NotImplementedError();
}

inline void cuBlasMatmulBatched(
    cublasHandle_t handle,
    const float* a, unsigned strideA,
    const float* b, unsigned strideB,
    float* c, unsigned strideC,
    unsigned m, unsigned n, unsigned k,
    unsigned batchCount,
    float alpha, float beta) {
  cublasStatus_t stat = cublasSgemmStridedBatched(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      m, n, k,
      &alpha,
      b, m, strideB,
      a, k, strideA,
      &beta,
      c, m, strideC,
      batchCount);
  if (stat != CUBLAS_STATUS_SUCCESS)
    throw CUBlasRuntimeError(stat);
}

inline void cuBlasMatmulBatched(
    cublasHandle_t handle,
    const double* a, unsigned strideA,
    const double* b, unsigned strideB,
    double* c, unsigned strideC,
    unsigned m, unsigned n, unsigned k,
    unsigned batchCount,
    double alpha, double beta) {
  cublasStatus_t stat = cublasDgemmStridedBatched(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      m, n, k,
      &alpha,
      b, m, strideB,
      a, k, strideA,
      &beta,
      c, m, strideC,
      batchCount);
  if (stat != CUBLAS_STATUS_SUCCESS)
    throw CUBlasRuntimeError(stat);
}

template <class T>
CUDABlasInterface<T>::CUDABlasInterface(const CUDADeviceProperty& property)
  : _property(property) {
}

template <class T>
CUDABlasInterface<T>::~CUDABlasInterface() {
}

template <class T>
Variable<T> CUDABlasInterface<T>::matmul(const Tensor<T>& a, const Tensor<T>& b) {
  unsigned m = b.shape(b.shape().size() - 1);
  unsigned n = a.shape(a.shape().size() - 2);
  unsigned k = a.shape(a.shape().size() - 1);
  unsigned strideA = n * k;
  unsigned strideB = m * k;
  unsigned strideC = m * n;
  if (a.shape().size() != b.shape().size()) {
    strideA = a.shape().size() == 2 ? 0 : strideA;
    strideB = b.shape().size() == 2 ? 0 : strideB;
  }
  std::vector<unsigned> resultS = a.shape().size() > b.shape().size() ? a.shape().toVector() : b.shape().toVector();
  resultS[resultS.size() - 2] = n;
  resultS[resultS.size() - 1] = m;

  Variable<T> result(std::move(resultS));
  unsigned batchCount = result.size() / (result.shape(result.shape().size() - 1) * result.shape(result.shape().size() - 2));

  unsigned bs = b.size() / (b.shape(b.shape().size() - 1) * b.shape(b.shape().size() - 2));

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  tfcc::CUDADevice* device = static_cast<tfcc::CUDADevice*>(tfcc::Device::getThreadDefault());

#ifdef TFCC_USE_TENSOR_CORE
  if (bs == 1 && device->isTensorCoreEnabled()) {
    n = a.size() / k;
    cuBlasMatmul(
        session->getImpl()->cublasHandle(),
        a.data(),
        b.data(),
        result.data(),
        m, n, k,
        static_cast<T>(1.0), static_cast<T>(0.0));
    return result;
  }
#endif

  cuBlasMatmulBatched(
      session->getImpl()->cublasHandle(),
      a.data(), strideA,
      b.data(), strideB,
      result.data(), strideC,
      m, n, k,
      batchCount,
      static_cast<T>(1.0), static_cast<T>(0.0));

  return result;
}

template <class T>
Variable<T> CUDABlasInterface<T>::matmul(const Tensor<T>& a, const Tensor<T>& b, const Tensor<T>& c) {
  unsigned m = b.shape(b.shape().size() - 1);
  unsigned n = a.shape(a.shape().size() - 2);
  unsigned k = a.shape(a.shape().size() - 1);
  unsigned strideA = n * k;
  unsigned strideB = m * k;
  unsigned strideC = m * n;
  if (a.shape().size() != b.shape().size()) {
    strideA = a.shape().size() == 2 ? 0 : strideA;
    strideB = b.shape().size() == 2 ? 0 : strideB;
  }
  std::vector<unsigned> resultSV = a.shape().size() > b.shape().size() ? a.shape().toVector() : b.shape().toVector();
  resultSV[resultSV.size() - 2] = n;
  resultSV[resultSV.size() - 1] = m;

  Shape resultS(std::move(resultSV));

  unsigned batchCount = resultS.area() / (resultS[resultS.size() - 1] * resultS[resultS.size() - 2]);

  unsigned bs = b.size() / (b.shape(b.shape().size() - 1) * b.shape(b.shape().size() - 2));

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  tfcc::CUDADevice* device = static_cast<tfcc::CUDADevice*>(tfcc::Device::getThreadDefault());

  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(resultS.area() / c.size(), c.size());
  Variable<T> result = _broadcast_bias_helper(c, resultS, blockCount, threadCount);

#ifdef TFCC_USE_TENSOR_CORE
  if (bs == 1 && device->isTensorCoreEnabled()) {
    n = a.size() / k;
    cuBlasMatmul(
        session->getImpl()->cublasHandle(),
        a.data(),
        b.data(),
        result.data(),
        m, n, k,
        static_cast<T>(1.0), static_cast<T>(1.0));
    return result;
  }
#endif

  cuBlasMatmulBatched(
      session->getImpl()->cublasHandle(),
      a.data(), strideA,
      b.data(), strideB,
      result.data(), strideC,
      m, n, k,
      batchCount,
      static_cast<T>(1.0), static_cast<T>(1.0));

  return result;
}

#define DEFINE_FUNC(type) template class CUDABlasInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
