

#include "tfcc_cudabasicinterface.h"

#include <algorithm>
#include <cstdlib>

#include "exceptions/tfcc_cudaruntimeerror.h"
#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_cudasession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"

namespace tfcc {

template <class T>
static __global__ void _cuda_slice(const T* a, unsigned s1, unsigned s2, unsigned s3, T* b, unsigned start, unsigned length) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  const unsigned total = s1 * length * s3;
  for (unsigned i = tid; i < total; i += skip) {
    unsigned ps1 = i / (length * s3);
    unsigned ps2 = (i / s3) % length + start;
    unsigned ps3 = i % s3;

    unsigned pos = ps3 + ps2 * s3 + ps1 * s2 * s3;
    b[i] = a[pos];
  }
}

template <class T>
static __global__ void _cuda_assign_to(const T* a, unsigned s1, unsigned s2, unsigned s3, T* b, unsigned start, unsigned length) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  const unsigned total = s1 * s2 * s3;
  for (unsigned i = tid; i < total; i += skip) {
    unsigned ps1 = i / (s2 * s3);
    unsigned ps2 = (i / s3) % s2 + start;
    unsigned ps3 = i % s3;

    unsigned pos = ps3 + ps2 * s3 + ps1 * length * s3;
    b[pos] = a[i];
  }
}

template <class T>
static __global__ void _cuda_transpose_2d(
    const T* a,
    unsigned s0, unsigned s1,
    unsigned f0, unsigned f1,
    T* b) {
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int skip = blockDim.x * gridDim.x;
  unsigned total = s0 * s1;
  for (unsigned i = tid; i < total; i += skip) {
    unsigned n0 = i / s1;
    unsigned n1 = i % s1;

    b[n0 * f0 + n1 * f1] = a[i];
  }
}

template <class T>
static __global__ void _cuda_transpose_3d(
    const T* a,
    unsigned s0, unsigned s1, unsigned s2,
    unsigned f0, unsigned f1, unsigned f2,
    T* b) {
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int skip = blockDim.x * gridDim.x;
  unsigned total = s0 * s1 * s2;
  for (unsigned i = tid; i < total; i += skip) {
    unsigned n0 = i / (s1 * s2);
    unsigned n1 = (i / s2) % s1;
    unsigned n2 = i % s2;

    b[n0 * f0 + n1 * f1 + n2 * f2] = a[i];
  }
}

template <class T>
static __global__ void _cuda_transpose_4d(
    const T* a,
    unsigned s0, unsigned s1, unsigned s2, unsigned s3,
    unsigned f0, unsigned f1, unsigned f2, unsigned f3,
    T* b) {
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int skip = blockDim.x * gridDim.x;
  unsigned total = s0 * s1 * s2 * s3;
  for (unsigned i = tid; i < total; i += skip) {
    unsigned n0 = i / (s1 * s2 * s3);
    unsigned n1 = (i / (s2 * s3)) % s1;
    unsigned n2 = (i / s3) % s2;
    unsigned n3 = i % s3;

    b[n0 * f0 + n1 * f1 + n2 * f2 + n3 * f3] = a[i];
  }
}

template <class T>
static __global__ void _cuda_transpose_5d(
    const T* a,
    unsigned s0, unsigned s1, unsigned s2, unsigned s3, unsigned s4,
    unsigned f0, unsigned f1, unsigned f2, unsigned f3, unsigned f4,
    T* b) {
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int skip = blockDim.x * gridDim.x;
  unsigned total = s0 * s1 * s2 * s3 * s4;
  for (unsigned i = tid; i < total; i += skip) {
    unsigned n0 = i / (s1 * s2 * s3 * s4);
    unsigned n1 = (i / (s2 * s3 * s4)) % s1;
    unsigned n2 = (i / (s3 * s4)) % s2;
    unsigned n3 = (i / s4) % s3;
    unsigned n4 = i % s4;

    b[n0 * f0 + n1 * f1 + n2 * f2 + n3 * f3 + n4 * f4] = a[i];
  }
}

template <class T>
static __global__ void _cuda_clip(const T* a, unsigned total, T minValue, T maxValue, T* b) {
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T v = a[i];
    v = max(v, minValue);
    v = min(v, maxValue);
    b[i] = v;
  }
}

template <class T>
static __global__ void _cuda_where(const uint8_t* condition, unsigned total, const T* x, const T* y, T* result) {
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T v = condition[i] != 0 ? x[i] : y[i];
    result[i] = v;
  }
}

template <class T>
static __global__ void _cuda_abs(const T* a, unsigned total, T* b) {
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip) {
    T v = a[i];
    b[i] = v < static_cast<T>(0) ? -v : v;
  }
}

template <class T>
CUDABasicInterface<T>::CUDABasicInterface(const CUDADeviceProperty& property)
  : _property(property) {
}

template <class T>
CUDABasicInterface<T>::~CUDABasicInterface() {
}

template <class T>
Variable<T> CUDABasicInterface<T>::slice(const Tensor<T>& a, size_t axis, unsigned start, unsigned end) {
  end = std::min(end, a.shape(axis));

  unsigned s1 = 1;
  unsigned s2 = a.shape(axis);
  unsigned s3 = 1;

  for (size_t i = 0; i < axis; ++i) {
    s1 *= a.shape(i);
  }
  for (size_t i = axis + 1; i < a.shape().size(); ++i) {
    s3 *= a.shape(i);
  }

  std::vector<unsigned> s = a.shape().toVector();
  s[axis] = end - start;
  Variable<T> result(s);

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  if (axis == 0) {
    cudaError_t ret = cudaMemcpyAsync(
        result.data(),
        a.data() + a.size() / a.shape(0) * start,
        result.size() * sizeof(T),
        cudaMemcpyDeviceToDevice,
        session->getImpl()->cudaStream());
    if (ret != cudaSuccess)
      throw CUDARuntimeError(ret);
    return result;
  }

  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(result.size());
  _cuda_slice<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(),
      s1, s2, s3,
      result.data(),
      start, end - start);

  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);

  return result;
}

template <class T>
void CUDABasicInterface<T>::assignTo(const Tensor<T>& a, size_t axis, unsigned start, Variable<T>& b) {
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  if (axis == 0) {
    cudaError_t ret = cudaMemcpyAsync(
        b.data() + b.size() / b.shape(0) * start,
        a.data(),
        a.size() * sizeof(T),
        cudaMemcpyDeviceToDevice,
        session->getImpl()->cudaStream());
    if (ret != cudaSuccess)
      throw CUDARuntimeError(ret);
    return;
  }

  unsigned s1 = 1;
  unsigned s2 = a.shape(axis);
  unsigned s3 = 1;

  for (size_t i = 0; i < axis; ++i) {
    s1 *= a.shape(i);
  }
  for (size_t i = axis + 1; i < a.shape().size(); ++i) {
    s3 *= a.shape(i);
  }

  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());
  _cuda_assign_to<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(),
      s1, s2, s3,
      b.data(),
      start, b.shape(axis));

  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
}

template <class T>
Variable<T> CUDABasicInterface<T>::transpose(const Tensor<T>& a, const std::vector<size_t>& perm) {
  std::vector<unsigned> newS;
  newS.reserve(a.shape().size());
  for (size_t i = 0; i < a.shape().size(); ++i)
    newS.emplace_back(a.shape(perm[i]));
  Variable<T> result(std::move(newS));
  unsigned lastOffset = 1;
  std::vector<unsigned> newOffsets(a.shape().size(), 0u);
  for (size_t i = 0; i < a.shape().size(); ++i) {
    newOffsets[perm[perm.size() - i - 1]] = lastOffset;
    lastOffset *= result.shape(result.shape().size() - i - 1);
  }

  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(result.size());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  switch (a.shape().size()) {
    case 2:
      _cuda_transpose_2d<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
          a.data(),
          a.shape(0), a.shape(1),
          newOffsets[0], newOffsets[1],
          result.data());
      break;
    case 3:
      _cuda_transpose_3d<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
          a.data(),
          a.shape(0), a.shape(1), a.shape(2),
          newOffsets[0], newOffsets[1], newOffsets[2],
          result.data());
      break;
    case 4:
      _cuda_transpose_4d<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
          a.data(),
          a.shape(0), a.shape(1), a.shape(2), a.shape(3),
          newOffsets[0], newOffsets[1], newOffsets[2], newOffsets[3],
          result.data());
      break;
    case 5:
      _cuda_transpose_5d<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
          a.data(),
          a.shape(0), a.shape(1), a.shape(2), a.shape(3), a.shape(4),
          newOffsets[0], newOffsets[1], newOffsets[2], newOffsets[3], newOffsets[4],
          result.data());
      break;
    default:
      throw NotImplementedError();
  }
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T>
Variable<T> CUDABasicInterface<T>::clip(const Tensor<T>& a, T minValue, T maxValue) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());

  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_clip<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      minValue, maxValue,
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T>
Variable<T> CUDABasicInterface<T>::concat(const std::vector<const Tensor<T>*>& values, size_t axis) {
  size_t total = 0;
  for (const Tensor<T>* tensor : values)
    total += tensor->shape(axis);
  std::vector<unsigned> s = (*values.begin())->shape().toVector();
  s[axis] = total;

  Variable<T> result(s);
  unsigned start = 0;
  for (const Tensor<T>* tensor : values) {
    this->assignTo(*tensor, axis, start, result);
    start += tensor->shape(axis);
  }

  return result;
}

template <class T>
Variable<T> CUDABasicInterface<T>::where(const Tensor<uint8_t>& condition, const Tensor<T>& x, const Tensor<T>& y) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(condition.size());

  Variable<T> result(condition.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_where<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      condition.data(),
      condition.size(),
      x.data(), y.data(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T>
Variable<T> CUDABasicInterface<T>::abs(const Tensor<T>& a) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(a.size());

  Variable<T> result(a.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_abs<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      a.data(), a.size(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

#define DEFINE_FUNC(type) template class CUDABasicInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
