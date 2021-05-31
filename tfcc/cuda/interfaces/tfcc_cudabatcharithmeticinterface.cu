

#include "tfcc_cudabatcharithmeticinterface.h"

#include "exceptions/tfcc_cudaruntimeerror.h"
#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_cudasession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "kernel/tfcc_cudaatomickernel.hpp"

namespace tfcc {

// cuda functions
template <class T>
static void __global__ _cuda_batch_add(const uintptr_t* values, unsigned batch, unsigned total, T* b) {
  constexpr unsigned CACHE_SIZE = 4096;
  __shared__ T cache[CACHE_SIZE];
  for (unsigned offset = 0; offset < total; offset += CACHE_SIZE) {
    unsigned count = total - offset < CACHE_SIZE ? total - offset : CACHE_SIZE;
    for (unsigned i = threadIdx.x; i < count; i += blockDim.x)
      cache[i] = static_cast<T>(0);
    for (unsigned i = blockIdx.x; i < batch; i += gridDim.x) {
      const T* p = reinterpret_cast<const T*>(values[i]) + offset;
      for (unsigned j = threadIdx.x; j < count; j += blockDim.x)
        cache[j] += p[j];
    }
    for (unsigned i = threadIdx.x; i < count; i += blockDim.x)
      atomic_add_wrapper(b + offset + i, cache[i]);
  }
}

template <class T>
static void __global__ _cuda_batch_mul(const uintptr_t* values, unsigned batch, unsigned total, T* b) {
  constexpr unsigned CACHE_SIZE = 4096;
  __shared__ T cache[CACHE_SIZE];
  for (unsigned offset = 0; offset < total; offset += CACHE_SIZE) {
    unsigned count = total - offset < CACHE_SIZE ? total - offset : CACHE_SIZE;
    for (unsigned i = threadIdx.x; i < count; i += blockDim.x)
      cache[i] = static_cast<T>(1);
    for (unsigned i = blockIdx.x; i < batch; i += gridDim.x) {
      const T* p = reinterpret_cast<const T*>(values[i]) + offset;
      for (unsigned j = threadIdx.x; j < count; j += blockDim.x)
        cache[j] *= p[j];
    }
    for (unsigned i = threadIdx.x; i < count; i += blockDim.x)
      atomic_mul_wrapper(b + offset + i, cache[i]);
  }
}

template <class T>
static inline typename std::enable_if<
    std::is_same<float, T>::value ||
        std::is_same<uint32_t, T>::value ||
        std::is_same<int32_t, T>::value,
    Variable<T>>::type
_batch_add_helper(const std::vector<const Tensor<T>*>& values, size_t blockCount, size_t threadCount) {
  Variable<T> result(values[0]->shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());

  cudaError_t ret = cudaMemcpyAsync(result.data(), values[0]->data(), values[0]->size() * sizeof(T), cudaMemcpyDeviceToDevice, session->getImpl()->cudaStream());
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);

  if (values.size() > 1) {
    size_t count = values.size() - 1;
    std::vector<uintptr_t> vs;
    vs.reserve(count);
    for (size_t i = 0; i < count; ++i)
      vs.push_back(reinterpret_cast<uintptr_t>(values[i + 1]->data()));
    Variable<uintptr_t> ptrs({static_cast<unsigned>(count)});
    session->sync();
    ret = cudaMemcpy(ptrs.data(), vs.data(), vs.size() * sizeof(uintptr_t), cudaMemcpyHostToDevice);
    if (ret != cudaSuccess)
      throw CUDARuntimeError(ret);
    blockCount = (blockCount + 64 - 1) / 64;
    _cuda_batch_add<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
        ptrs.data(),
        count, result.size(),
        result.data());
  }
  return result;
}

template <class T, class ST>
static inline Variable<T> _batch_add_helper(const std::vector<const Tensor<T>*>& values, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<
    std::is_same<float, T>::value,
    Variable<T>>::type
_batch_mul_helper(const std::vector<const Tensor<T>*>& values, size_t blockCount, size_t threadCount) {
  Variable<T> result(values[0]->shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());

  cudaError_t ret = cudaMemcpyAsync(result.data(), values[0]->data(), values[0]->size() * sizeof(T), cudaMemcpyDeviceToDevice, session->getImpl()->cudaStream());
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);

  if (values.size() > 1) {
    size_t count = values.size() - 1;
    std::vector<uintptr_t> vs;
    vs.reserve(count);
    for (size_t i = 0; i < count; ++i)
      vs.push_back(reinterpret_cast<uintptr_t>(values[i + 1]->data()));
    Variable<uintptr_t> ptrs({static_cast<unsigned>(count)});
    session->sync();
    ret = cudaMemcpy(ptrs.data(), vs.data(), vs.size() * sizeof(uintptr_t), cudaMemcpyHostToDevice);
    if (ret != cudaSuccess)
      throw CUDARuntimeError(ret);
    blockCount = (blockCount + 64 - 1) / 64;
    _cuda_batch_mul<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
        ptrs.data(),
        count, result.size(),
        result.data());
  }
  return result;
}

template <class T, class ST>
static inline Variable<T> _batch_mul_helper(const std::vector<const Tensor<T>*>& values, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
CUDABatchArithmeticInterface<T>::CUDABatchArithmeticInterface(const CUDADeviceProperty& property)
  : _property(property) {
}

template <class T>
CUDABatchArithmeticInterface<T>::~CUDABatchArithmeticInterface() {
}

template <class T>
Variable<T> CUDABatchArithmeticInterface<T>::add(const std::vector<const Tensor<T>*>& values) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(values.size(), values[0]->size());
  return _batch_add_helper(values, blockCount, threadCount);
}

template <class T>
Variable<T> CUDABatchArithmeticInterface<T>::mul(const std::vector<const Tensor<T>*>& values) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(values.size(), values[0]->size());
  return _batch_mul_helper(values, blockCount, threadCount);
}

#define DEFINE_FUNC(type) template class CUDABatchArithmeticInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
