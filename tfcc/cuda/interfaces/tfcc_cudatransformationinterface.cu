

#include "tfcc_cudatransformationinterface.h"

#include <algorithm>

#include "exceptions/tfcc_cudaruntimeerror.h"
#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_cudasession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"

namespace tfcc {

// transform
template <class T>
static __global__ void _cuda_transform(const T* a, unsigned total, T alpha, T beta, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    auto v = alpha * a[i] + beta;
    b[i] = v;
  }
}

// transform2
template <class T>
static __global__ void _cuda_transform2(const T* a, unsigned total, T alpha, T beta, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    auto v = a[i] / alpha + beta;
    b[i] = v;
  }
}

// transform3
template <class T>
static __global__ void _cuda_transform3(const T* a, unsigned total, T alpha, T beta, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    auto v = alpha / a[i] + beta;
    b[i] = v;
  }
}

// transform4
template <class T>
static __global__ void _cuda_transform4(const T* a, unsigned total, T alpha, T beta, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    auto v = beta - a[i] * alpha;
    b[i] = v;
  }
}

// transform5
template <class T>
static __global__ void _cuda_transform5(const T* a, unsigned total, T alpha, T beta, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    auto v = beta - a[i] / alpha;
    b[i] = v;
  }
}

// transform6
template <class T>
static __global__ void _cuda_transform6(const T* a, unsigned total, T alpha, T beta, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    auto v = beta - alpha / a[i];
    b[i] = v;
  }
}

template <class T>
CUDATransformationInterface<T>::CUDATransformationInterface(const CUDADeviceProperty& property)
  : _property(property) {
}

template <class T>
CUDATransformationInterface<T>::~CUDATransformationInterface() {
}

template <class T>
Variable<T> CUDATransformationInterface<T>::transform(const Tensor<T>& a, T alpha, T beta) {
  Variable<T> result(a.shape());
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_transform<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      alpha, beta,
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);

  return result;
}

template <class T>
Variable<T> CUDATransformationInterface<T>::transform2(const Tensor<T>& a, T alpha, T beta) {
  Variable<T> result(a.shape());
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_transform2<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      alpha, beta,
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);

  return result;
}

template <class T>
Variable<T> CUDATransformationInterface<T>::transform3(const Tensor<T>& a, T alpha, T beta) {
  Variable<T> result(a.shape());
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_transform3<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      alpha, beta,
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);

  return result;
}

template <class T>
Variable<T> CUDATransformationInterface<T>::transform4(const Tensor<T>& a, T alpha, T beta) {
  Variable<T> result(a.shape());
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_transform4<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      alpha, beta,
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);

  return result;
}

template <class T>
Variable<T> CUDATransformationInterface<T>::transform5(const Tensor<T>& a, T alpha, T beta) {
  Variable<T> result(a.shape());
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_transform5<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      alpha, beta,
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);

  return result;
}

template <class T>
Variable<T> CUDATransformationInterface<T>::transform6(const Tensor<T>& a, T alpha, T beta) {
  Variable<T> result(a.shape());
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_transform6<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      alpha, beta,
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);

  return result;
}

#define DEFINE_FUNC(type) template class CUDATransformationInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
