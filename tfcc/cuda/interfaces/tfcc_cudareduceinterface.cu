

#include "tfcc_cudareduceinterface.h"

#include <limits>

#include "exceptions/tfcc_cudaruntimeerror.h"
#include "exceptions/tfcc_invalidargumenterror.h"
#include "framework/tfcc_cudasession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"

namespace tfcc {

template <class T>
struct CUDAReduceSumHelper {
  static __device__ T process(T a, T b) { return a + b; }
  static constexpr T BASE = static_cast<T>(0);
  static __device__ T afterProcess(T a, unsigned chunk) { return a; }
};

template <class T>
struct CUDAReduceMeanHelper {
  static __device__ T process(T a, T b) { return a + b; }
  static constexpr T BASE = static_cast<T>(0);
  static __device__ T afterProcess(T a, unsigned chunk) { return a / static_cast<T>(chunk); }
};

template <class T>
struct CUDAReduceProdHelper {
  static __device__ T process(T a, T b) { return a * b; }
  static constexpr T BASE = static_cast<T>(1);
  static __device__ T afterProcess(T a, unsigned chunk) { return a; }
};

template <class T>
struct CUDAReduceMaxHelper {
  static __device__ T process(T a, T b) { return a > b ? a : b; }
  static constexpr T BASE = std::numeric_limits<T>::lowest();
  static __device__ T afterProcess(T a, unsigned chunk) { return a; }
};

template <class T>
struct CUDAReduceMinHelper {
  static __device__ T process(T a, T b) { return a < b ? a : b; }
  static constexpr T BASE = std::numeric_limits<T>::max();
  static __device__ T afterProcess(T a, unsigned chunk) { return a; }
};

template <class T>
struct CUDAReduceAnyHelper {
  static __device__ T process(T a, T b) { return (a || b) ? static_cast<T>(true) : static_cast<T>(false); }
  static constexpr T BASE = static_cast<T>(false);
  static __device__ T afterProcess(T a, unsigned chunk) { return a; }
};

template <class T, unsigned THREAD_COUNT, bool NEED_CHECK, class ReduceHelper>
static __global__ void _cuda_reduce(const T* a, unsigned chunk, T* b) {
  static_assert(THREAD_COUNT >= 1 && THREAD_COUNT <= 1024 && (THREAD_COUNT % 32 == 0 || THREAD_COUNT < 32), "THREAD_COUNT error");
  __shared__ T sdata[32];
  a += blockIdx.x * chunk;
  unsigned int tid = threadIdx.x;
  T result;
  if (NEED_CHECK)
    result = tid < chunk ? a[tid] : ReduceHelper::BASE;
  else
    result = a[tid];
  for (unsigned i = tid + blockDim.x; i < chunk; i += blockDim.x)
    result = ReduceHelper::process(result, a[i]);
  if (THREAD_COUNT == 1) {
    b[blockIdx.x] = ReduceHelper::afterProcess(result, chunk);
    return;
  }
  if (THREAD_COUNT >= 32) {
    result = ReduceHelper::process(result, __shfl_down_sync(0xffffffff, result, 16));
    result = ReduceHelper::process(result, __shfl_down_sync(0xffffffff, result, 8));
    result = ReduceHelper::process(result, __shfl_down_sync(0xffffffff, result, 4));
    result = ReduceHelper::process(result, __shfl_down_sync(0xffffffff, result, 2));
    result = ReduceHelper::process(result, __shfl_down_sync(0xffffffff, result, 1));
    if (tid % 32 == 0)
      sdata[tid / 32] = result;
    __syncthreads();
  }
  if (THREAD_COUNT == 32) {
    if (tid == 0) b[blockIdx.x] = ReduceHelper::afterProcess(result, chunk);
    return;
  }
  constexpr unsigned L2_COUNT = THREAD_COUNT > 32 ? THREAD_COUNT / 32 : THREAD_COUNT;
  if (tid < 32) {
    if (THREAD_COUNT > 32)
      result = tid < L2_COUNT ? sdata[tid] : ReduceHelper::BASE;

    if (L2_COUNT > 16) result = ReduceHelper::process(result, __shfl_down_sync(0xffffffff, result, 16));
    if (L2_COUNT > 8) result = ReduceHelper::process(result, __shfl_down_sync(0xffffffff, result, 8));
    if (L2_COUNT > 4) result = ReduceHelper::process(result, __shfl_down_sync(0xffffffff, result, 4));
    if (L2_COUNT > 2) result = ReduceHelper::process(result, __shfl_down_sync(0xffffffff, result, 2));
    if (L2_COUNT > 1) result = ReduceHelper::process(result, __shfl_down_sync(0xffffffff, result, 1));
  }

  if (tid == 0) b[blockIdx.x] = ReduceHelper::afterProcess(result, chunk);
}

#define REDUCE_CASE(NUM, HELPER)\
  case NUM:\
    while (offset < batchSize) {\
      _cuda_reduce<T, NUM < 32 ? 32 : NUM, NUM < 32, HELPER<T>><<<blockCount, NUM < 32 ? 32 : NUM, 0, session->getImpl()->cudaStream()>>>(\
        a.data() + reduceSize * offset, reduceSize, result.data() + offset\
      );\
      offset += blockCount;\
    }\
    break

#define REDUCE_CASE_2(NUM1, NUM2, HELPER)\
  REDUCE_CASE(NUM1, HELPER);\
  REDUCE_CASE(NUM2, HELPER)

#define REDUCE_CASE_4(NUM1, NUM2, NUM3, NUM4, HELPER)\
  REDUCE_CASE_2(NUM1, NUM2, HELPER);\
  REDUCE_CASE_2(NUM3, NUM4, HELPER)

#define REDUCE_CASE_8(NUM1, NUM2, NUM3, NUM4, NUM5, NUM6, NUM7, NUM8, HELPER)\
  REDUCE_CASE_4(NUM1, NUM2, NUM3, NUM4, HELPER);\
  REDUCE_CASE_4(NUM5, NUM6, NUM7, NUM8, HELPER)

#define REDUCE_SWITCH(THREADCNT, HELPER)\
  unsigned offset = 0;\
  switch (THREADCNT) {\
    REDUCE_CASE_8(1, 2, 3, 4, 5, 6, 7, 8, HELPER);\
    REDUCE_CASE_8(9, 10, 11, 12, 13, 14, 15, 16, HELPER);\
    REDUCE_CASE_8(17, 18, 19, 20, 21, 22, 23, 24, HELPER);\
    REDUCE_CASE_8(25, 26, 27, 28, 29, 30, 31, 32, HELPER);\
    REDUCE_CASE_4(32 * 2, 32 * 3, 32 * 4, 32 * 5, HELPER);\
    REDUCE_CASE_2(32 * 6, 32 * 7, HELPER);\
    REDUCE_CASE(32 * 8, HELPER);\
    REDUCE_CASE_8(32 * 9, 32 * 10, 32 * 11, 32 * 12, 32 * 13, 32 * 14, 32 * 15, 32 * 16, HELPER);\
    REDUCE_CASE_8(32 * 17, 32 * 18, 32 * 19, 32 * 20, 32 * 21, 32 * 22, 32 * 23, 32 * 24, HELPER);\
    REDUCE_CASE_8(32 * 25, 32 * 26, 32 * 27, 32 * 28, 32 * 29, 32 * 30, 32 * 31, 32 * 32, HELPER);\
  }

template <class T>
CUDAReduceInterface<T>::CUDAReduceInterface(const CUDADeviceProperty& property)
  : _property(property) {
}

template <class T>
CUDAReduceInterface<T>::~CUDAReduceInterface() {
}

template <class T>
Variable<T> CUDAReduceInterface<T>::reduceSum(const Tensor<T>& a, size_t keep) {
  std::vector<unsigned> s = a.shape().toVector();
  for (size_t i = keep; i < s.size(); ++i)
    s[i] = 1;
  Variable<T> result(s);
  unsigned reduceSize = 1;
  for (size_t i = keep; i < a.shape().size(); ++i) {
    reduceSize *= a.shape(i);
  }
  unsigned batchSize = a.size() / reduceSize;

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());

  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(batchSize, (reduceSize + 1) / 2);
  threadCount = std::max(1lu, threadCount);
  threadCount = threadCount < 32 ? threadCount : (threadCount + 31) / 32 * 32;
  threadCount = std::min(threadCount, 1024lu);

  REDUCE_SWITCH(threadCount, CUDAReduceSumHelper);

  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T>
Variable<T> CUDAReduceInterface<T>::reduceMean(const Tensor<T>& a, size_t keep) {
  std::vector<unsigned> s = a.shape().toVector();
  for (size_t i = keep; i < s.size(); ++i)
    s[i] = 1;
  Variable<T> result(s);
  unsigned reduceSize = 1;
  for (size_t i = keep; i < a.shape().size(); ++i) {
    reduceSize *= a.shape(i);
  }
  unsigned batchSize = a.size() / reduceSize;

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());

  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(batchSize, (reduceSize + 1) / 2);
  threadCount = std::max(1lu, threadCount);
  threadCount = threadCount < 32 ? threadCount : (threadCount + 31) / 32 * 32;
  threadCount = std::min(threadCount, 1024lu);

  REDUCE_SWITCH(threadCount, CUDAReduceMeanHelper);

  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T>
Variable<T> CUDAReduceInterface<T>::reduceProd(const Tensor<T>& a, size_t keep) {
  std::vector<unsigned> s = a.shape().toVector();
  for (size_t i = keep; i < s.size(); ++i)
    s[i] = 1;
  Variable<T> result(s);
  unsigned reduceSize = 1;
  for (size_t i = keep; i < a.shape().size(); ++i) {
    reduceSize *= a.shape(i);
  }
  unsigned batchSize = a.size() / reduceSize;

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());

  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(batchSize, (reduceSize + 1) / 2);
  threadCount = std::max(1lu, threadCount);
  threadCount = threadCount < 32 ? threadCount : (threadCount + 31) / 32 * 32;
  threadCount = std::min(threadCount, 1024lu);

  REDUCE_SWITCH(threadCount, CUDAReduceProdHelper);

  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T>
Variable<T> CUDAReduceInterface<T>::reduceMax(const Tensor<T>& a, size_t keep) {
  std::vector<unsigned> s = a.shape().toVector();
  for (size_t i = keep; i < s.size(); ++i)
    s[i] = 1;
  Variable<T> result(s);
  unsigned reduceSize = 1;
  for (size_t i = keep; i < a.shape().size(); ++i) {
    reduceSize *= a.shape(i);
  }
  unsigned batchSize = a.size() / reduceSize;

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());

  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(batchSize, (reduceSize + 1) / 2);
  threadCount = std::max(1lu, threadCount);
  threadCount = threadCount < 32 ? threadCount : (threadCount + 31) / 32 * 32;
  threadCount = std::min(threadCount, 1024lu);

  REDUCE_SWITCH(threadCount, CUDAReduceMaxHelper);

  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T>
Variable<T> CUDAReduceInterface<T>::reduceMin(const Tensor<T>& a, size_t keep) {
  std::vector<unsigned> s = a.shape().toVector();
  for (size_t i = keep; i < s.size(); ++i)
    s[i] = 1;
  Variable<T> result(s);
  unsigned reduceSize = 1;
  for (size_t i = keep; i < a.shape().size(); ++i) {
    reduceSize *= a.shape(i);
  }
  unsigned batchSize = a.size() / reduceSize;

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());

  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(batchSize, (reduceSize + 1) / 2);
  threadCount = std::max(1lu, threadCount);
  threadCount = threadCount < 32 ? threadCount : (threadCount + 31) / 32 * 32;
  threadCount = std::min(threadCount, 1024lu);

  REDUCE_SWITCH(threadCount, CUDAReduceMinHelper);

  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T>
Variable<T> CUDAReduceInterface<T>::reduceAny(const Tensor<T>& a, size_t keep) {
  std::vector<unsigned> s = a.shape().toVector();
  for (size_t i = keep; i < s.size(); ++i)
    s[i] = 1;
  Variable<T> result(s);
  unsigned reduceSize = 1;
  for (size_t i = keep; i < a.shape().size(); ++i) {
    reduceSize *= a.shape(i);
  }
  unsigned batchSize = a.size() / reduceSize;

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());

  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(batchSize, (reduceSize + 1) / 2);
  threadCount = std::max(1lu, threadCount);
  threadCount = threadCount < 32 ? threadCount : (threadCount + 31) / 32 * 32;
  threadCount = std::min(threadCount, 1024lu);

  REDUCE_SWITCH(threadCount, CUDAReduceAnyHelper);

  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

#define DEFINE_FUNC(type) template class CUDAReduceInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
