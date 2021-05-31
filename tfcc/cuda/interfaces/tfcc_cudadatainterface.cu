

#include "tfcc_cudadatainterface.h"

#include "exceptions/tfcc_cudaruntimeerror.h"
#include "framework/tfcc_cudasession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"

namespace tfcc {

// cuda functions
template <class T>
static void __global__ _cuda_ones(T* a, unsigned total) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip)
    a[i] = static_cast<T>(1);
}

template <class T1, class T2>
static void __global__ _cuda_cast(const T1* a, unsigned total, T2* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip)
    b[i] = static_cast<T1>(a[i]);
}

template <class T>
static void __global__ _cuda_cast_to_boolean(const T* a, unsigned total, uint8_t* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip)
    b[i] = a[i] ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
}

// helper function
template <class T1, class T2>
static inline Variable<T2> _cast_helper(const Tensor<T1>& a, T2, size_t blockCount, size_t threadCount) {
  Variable<T2> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_cast<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

// class function
template <class T>
CUDADataInterface<T>::CUDADataInterface(const CUDADeviceProperty& property)
  : _property(property) {
}

template <class T>
CUDADataInterface<T>::~CUDADataInterface() {
}

template <class T>
void CUDADataInterface<T>::set(T* dst, const T* data, size_t len) {
  cudaError_t ret = cudaMemcpy(dst, data, len * sizeof(T), cudaMemcpyHostToDevice);
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
}

template <class T>
void CUDADataInterface<T>::set(Variable<T>& a, const T* data) {
  a.sync();
  cudaError_t ret = cudaMemcpy(a.data(), data, a.size() * sizeof(T), cudaMemcpyHostToDevice);
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
}

template <class T>
void CUDADataInterface<T>::set(Variable<T>& a, std::vector<T>&& data) {
  a.sync();
  cudaError_t ret = cudaMemcpy(a.data(), data.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice);
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
}

template <class T>
void CUDADataInterface<T>::get(const Tensor<T>& a, T* data) {
  a.sync();
  cudaError_t ret = cudaMemcpy(data, a.data(), a.size() * sizeof(T), cudaMemcpyDeviceToHost);
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
}

template <class T>
void CUDADataInterface<T>::zeros(Variable<T>& a) {
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  cudaError_t ret = cudaMemsetAsync(a.data(), 0, a.size() * sizeof(T), session->getImpl()->cudaStream());
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
}

template <class T>
void CUDADataInterface<T>::ones(Variable<T>& a) {
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  _cuda_ones<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(),
      a.size());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
}

template <class T>
Variable<T> CUDADataInterface<T>::copy(const Tensor<T>& a) {
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  Variable<T> result(a.shape());
  cudaError_t ret = cudaMemcpyAsync(result.data(), a.data(), a.size() * sizeof(T), cudaMemcpyDeviceToDevice, session->getImpl()->cudaStream());
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T>
Variable<T> CUDADataInterface<T>::cast(const Tensor<float>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _cast_helper(a, T(), blockCount, threadCount);
}

template <class T>
Variable<T> CUDADataInterface<T>::cast(const Tensor<double>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _cast_helper(a, T(), blockCount, threadCount);
}

template <class T>
Variable<T> CUDADataInterface<T>::cast(const Tensor<int8_t>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _cast_helper(a, T(), blockCount, threadCount);
}

template <class T>
Variable<T> CUDADataInterface<T>::cast(const Tensor<uint8_t>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _cast_helper(a, T(), blockCount, threadCount);
}

template <class T>
Variable<T> CUDADataInterface<T>::cast(const Tensor<int16_t>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _cast_helper(a, T(), blockCount, threadCount);
}

template <class T>
Variable<T> CUDADataInterface<T>::cast(const Tensor<uint16_t>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _cast_helper(a, T(), blockCount, threadCount);
}

template <class T>
Variable<T> CUDADataInterface<T>::cast(const Tensor<int32_t>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _cast_helper(a, T(), blockCount, threadCount);
}

template <class T>
Variable<T> CUDADataInterface<T>::cast(const Tensor<uint32_t>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _cast_helper(a, T(), blockCount, threadCount);
}

template <class T>
Variable<T> CUDADataInterface<T>::cast(const Tensor<int64_t>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _cast_helper(a, T(), blockCount, threadCount);
}

template <class T>
Variable<T> CUDADataInterface<T>::cast(const Tensor<uint64_t>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _cast_helper(a, T(), blockCount, threadCount);
}

template <class T>
Variable<uint8_t> CUDADataInterface<T>::cast_to_boolean(const Tensor<T>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());

  Variable<uint8_t> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_cast_to_boolean<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

#define DEFINE_FUNC(type) template class CUDADataInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
