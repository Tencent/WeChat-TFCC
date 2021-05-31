

#include "tfcc_cudasegmentinterface.h"

#include "exceptions/tfcc_cudaruntimeerror.h"
#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_cudasession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"

namespace tfcc {

template <class T>
static __global__ void _cuda_unsorted_segment_sum(const T* a, unsigned batch, unsigned k, const int* ids, unsigned sum, T* b) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  const unsigned total = batch * k;
  for (unsigned i = tid; i < total; i += skip) {
    unsigned ps1 = i / k;
    unsigned ps2 = i % k;
    int idx = ids[ps1];
    if (idx < 0) {
      continue;
    }
    atomicAdd(b + idx * k + ps2, a[i]);
  }
}

// helper functions
template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<int32_t, T>::value || std::is_same<uint32_t, T>::value, Variable<T>>::type
_unsorted_segment_sum_helper(const Tensor<T>& a, const Tensor<int>& ids, unsigned num, size_t blockCount, size_t threadCount) {
  std::vector<unsigned> s = a.shape().toVector();
  s[0] = num;
  Variable<T> result(std::move(s));

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  cudaError_t ret = cudaMemsetAsync(result.data(), 0, result.size() * sizeof(T), session->getImpl()->cudaStream());
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);

  _cuda_unsorted_segment_sum<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(),
      a.shape(0),
      a.size() / a.shape(0),
      ids.data(),
      num,
      result.data());
  ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);

  return result;
}

template <class T, class ST>
static inline Variable<T> _unsorted_segment_sum_helper(const Tensor<T>& a, const Tensor<int>& ids, unsigned num, ST blockCount, ST threadCount) {
  throw NotImplementedError();
}

template <class T>
CUDASegmentInterface<T>::CUDASegmentInterface(const CUDADeviceProperty& property)
  : _property(property) {
}

template <class T>
CUDASegmentInterface<T>::~CUDASegmentInterface() {
}

template <class T>
Variable<T> CUDASegmentInterface<T>::unsortedSegmentSum(const Tensor<T>& a, const Tensor<int>& ids, unsigned num) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  return _unsorted_segment_sum_helper(a, ids, num, blockCount, threadCount);
}

#define DEFINE_FUNC(type) template class CUDASegmentInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
