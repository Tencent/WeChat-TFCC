

#include "tfcc_cudacomparisoninterface.h"

#include "exceptions/tfcc_cudaruntimeerror.h"
#include "framework/tfcc_cudasession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"

namespace tfcc {

// cuda functions
template <class T>
static void __global__ _cuda_equal(const T* a, unsigned total, T b, uint8_t* result) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip)
    result[i] = a[i] == b ? 1 : 0;
}

template <class T>
static void __global__ _cuda_unequal(const T* a, unsigned total, T b, uint8_t* result) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip)
    result[i] = a[i] != b ? 1 : 0;
}

template <class T>
static void __global__ _cuda_greater(const T* a, unsigned total, T b, uint8_t* result) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip)
    result[i] = a[i] > b ? 1 : 0;
}

template <class T>
static void __global__ _cuda_greater_equal(const T* a, unsigned total, T b, uint8_t* result) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip)
    result[i] = a[i] >= b ? 1 : 0;
}

template <class T>
static void __global__ _cuda_less(const T* a, unsigned total, T b, uint8_t* result) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip)
    result[i] = a[i] < b ? 1 : 0;
}

template <class T>
static void __global__ _cuda_less_equal(const T* a, unsigned total, T b, uint8_t* result) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip)
    result[i] = a[i] <= b ? 1 : 0;
}

// class function
template <class T>
CUDAComparisonInterface<T>::CUDAComparisonInterface(const CUDADeviceProperty& property)
  : _property(property) {
}

template <class T>
CUDAComparisonInterface<T>::~CUDAComparisonInterface() {
}

template <class T>
Variable<uint8_t> CUDAComparisonInterface<T>::equal(const Tensor<T>& a, T b) {
  Variable<uint8_t> result(a.shape());
  CUDASession* session = static_cast<CUDASession*>(Session::getThreadDefault());
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  _cuda_equal<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(),
      a.size(),
      b,
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T>
Variable<uint8_t> CUDAComparisonInterface<T>::unequal(const Tensor<T>& a, T b) {
  Variable<uint8_t> result(a.shape());
  CUDASession* session = static_cast<CUDASession*>(Session::getThreadDefault());
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  _cuda_unequal<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(),
      a.size(),
      b,
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T>
Variable<uint8_t> CUDAComparisonInterface<T>::greater(const Tensor<T>& a, T b) {
  Variable<uint8_t> result(a.shape());
  CUDASession* session = static_cast<CUDASession*>(Session::getThreadDefault());
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  _cuda_greater<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(),
      a.size(),
      b,
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T>
Variable<uint8_t> CUDAComparisonInterface<T>::greaterEqual(const Tensor<T>& a, T b) {
  Variable<uint8_t> result(a.shape());
  CUDASession* session = static_cast<CUDASession*>(Session::getThreadDefault());
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  _cuda_greater_equal<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(),
      a.size(),
      b,
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T>
Variable<uint8_t> CUDAComparisonInterface<T>::less(const Tensor<T>& a, T b) {
  Variable<uint8_t> result(a.shape());
  CUDASession* session = static_cast<CUDASession*>(Session::getThreadDefault());
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  _cuda_less<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(),
      a.size(),
      b,
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T>
Variable<uint8_t> CUDAComparisonInterface<T>::lessEqual(const Tensor<T>& a, T b) {
  Variable<uint8_t> result(a.shape());
  CUDASession* session = static_cast<CUDASession*>(Session::getThreadDefault());
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  _cuda_less_equal<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(),
      a.size(),
      b,
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

#define DEFINE_FUNC(type) template class CUDAComparisonInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
