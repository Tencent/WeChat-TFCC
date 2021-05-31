

#include "tfcc_cudaarithmeticinterface.h"

#include "exceptions/tfcc_cudaruntimeerror.h"
#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_cudasession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "interfaces/tfcc_cudabroadcasthelper.h"

namespace tfcc {

// Basic op (+ - * /)
struct _AddOp {
  template <class T>
  static inline __device__ T process(T a, T b) { return a + b; }
};

struct _SubOp {
  template <class T>
  static inline __device__ T process(T a, T b) { return a - b; }
};

struct _MulOp {
  template <class T>
  static inline __device__ T process(T a, T b) { return a * b; }
};

struct _DivOp {
  template <class T>
  static inline __device__ T process(T a, T b) { return a / b; }
};

template <class T>
CUDAArithmeticInterface<T>::CUDAArithmeticInterface(const CUDADeviceProperty& property)
  : _property(property) {
}

template <class T>
CUDAArithmeticInterface<T>::~CUDAArithmeticInterface() {
}

template <class T>
Variable<T> CUDAArithmeticInterface<T>::add(const Tensor<T>& a, const Tensor<T>& b) {
  return _process_can_broadcast_op<T, _AddOp>(a, b, _property);
}

template <class T>
Variable<T> CUDAArithmeticInterface<T>::sub(const Tensor<T>& a, const Tensor<T>& b) {
  return _process_can_broadcast_op<T, _SubOp>(a, b, _property);
}

template <class T>
Variable<T> CUDAArithmeticInterface<T>::mul(const Tensor<T>& a, const Tensor<T>& b) {
  return _process_can_broadcast_op<T, _MulOp>(a, b, _property);
}

template <class T>
Variable<T> CUDAArithmeticInterface<T>::div(const Tensor<T>& a, const Tensor<T>& b) {
  return _process_can_broadcast_op<T, _DivOp>(a, b, _property);
}

#define DEFINE_FUNC(type) template class CUDAArithmeticInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
