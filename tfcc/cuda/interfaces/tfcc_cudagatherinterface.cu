

#include "tfcc_cudagatherinterface.h"

#include <algorithm>

#include "exceptions/tfcc_cudaruntimeerror.h"
#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_cudasession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"

namespace tfcc {

template <class T, class IDX>
static __global__ void _cuda_gather(const T* params, unsigned batch, unsigned chunkSize, const IDX* indices, unsigned length, T* result) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  const unsigned total = length * chunkSize;
  for (unsigned i = tid; i < total; i += skip) {
    unsigned p = i / chunkSize;
    unsigned x = i % chunkSize;

    IDX idx = indices[p];

    if (idx < 0 || idx >= batch) {
      result[i] = static_cast<T>(0);
    } else {
      result[i] = params[idx * chunkSize + x];
    }
  }
}

template <class T>
CUDAGatherInterface<T>::CUDAGatherInterface(const CUDADeviceProperty& property)
  : _property(property) {
}

template <class T>
CUDAGatherInterface<T>::~CUDAGatherInterface() {
}

template <class T, class IDX>
static Variable<T> _gather_helper(const CUDADeviceProperty& property, const Tensor<T>& params, const Tensor<IDX>& indices) {
  std::vector<unsigned> shape;
  for (size_t i = 0; i < indices.shape().size(); ++i)
    shape.push_back(indices.shape(i));
  for (size_t i = 1; i < params.shape().size(); ++i)
    shape.push_back(params.shape(i));

  Variable<T> result(shape);

  CUDASession* session = static_cast<CUDASession*>(Session::getThreadDefault());

  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = property.getSuitableKernelSize(result.size());
  _cuda_gather<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      params.data(),
      params.shape(0), params.size() / params.shape(0),
      indices.data(),
      indices.size(),
      result.data());

  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);

  return result;
}

template <class T>
Variable<T> CUDAGatherInterface<T>::gather(const Tensor<T>& params, const Tensor<uint32_t>& indices) {
  return _gather_helper(_property, params, indices);
}

template <class T>
Variable<T> CUDAGatherInterface<T>::gather(const Tensor<T>& params, const Tensor<int32_t>& indices) {
  return _gather_helper(_property, params, indices);
}

template <class T>
Variable<T> CUDAGatherInterface<T>::gather(const Tensor<T>& params, const Tensor<uint64_t>& indices) {
  return _gather_helper(_property, params, indices);
}

template <class T>
Variable<T> CUDAGatherInterface<T>::gather(const Tensor<T>& params, const Tensor<int64_t>& indices) {
  return _gather_helper(_property, params, indices);
}

#define DEFINE_FUNC(type) template class CUDAGatherInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
