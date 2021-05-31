

#include "tfcc_cudaminmaxinterface.h"

#include "exceptions/tfcc_cudaruntimeerror.h"
#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_cudasession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "interfaces/tfcc_cudabroadcasthelper.h"

namespace tfcc {

template <class T>
static __global__ void _cuda_min(const T* a, unsigned total, T b, T* c) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    auto v = a[i] < b ? a[i] : b;
    c[i] = v;
  }
}

template <class T>
static __global__ void _cuda_max(const T* a, unsigned total, T b, T* c) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    auto v = a[i] < b ? b : a[i];
    c[i] = v;
  }
}

struct _MinOp {
  template <class T>
  static inline __device__ T process(T a, T b) { return a < b ? a : b; }
};

struct _MaxOp {
  template <class T>
  static inline __device__ T process(T a, T b) { return a < b ? b : a; }
};

template <class T>
CUDAMinMaxInterface<T>::CUDAMinMaxInterface(const CUDADeviceProperty& property)
  : _property(property) {
}

template <class T>
CUDAMinMaxInterface<T>::~CUDAMinMaxInterface() {
}

template <class T>
Variable<T> CUDAMinMaxInterface<T>::min(const Tensor<T>& a, const Tensor<T>& b) {
  return _process_can_broadcast_op<T, _MinOp>(a, b, _property);
}

template <class T>
Variable<T> CUDAMinMaxInterface<T>::min(const Tensor<T>& a, T b) {
  Variable<T> result(a.shape());
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_min<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
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
Variable<T> CUDAMinMaxInterface<T>::max(const Tensor<T>& a, const Tensor<T>& b) {
  return _process_can_broadcast_op<T, _MaxOp>(a, b, _property);
}

template <class T>
Variable<T> CUDAMinMaxInterface<T>::max(const Tensor<T>& a, T b) {
  Variable<T> result(a.shape());
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_max<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(),
      a.size(),
      b,
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);

  return result;
}

#define DEFINE_FUNC(type) template class CUDAMinMaxInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
