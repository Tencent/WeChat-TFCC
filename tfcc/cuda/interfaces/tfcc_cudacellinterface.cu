

#include "tfcc_cudacellinterface.h"

#include <type_traits>

#include "exceptions/tfcc_cudaruntimeerror.h"
#include "exceptions/tfcc_cudnnruntimeerror.h"
#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_cudasession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"

namespace tfcc {

// cuda functionsz
template <class T>
static inline T __device__ sigmoid(const T value) {
  return static_cast<T>(1) / (static_cast<T>(1) + exp(-value));
}

template <class T>
static void __global__ _cuda_process_lstm_cell(
    unsigned batch,
    unsigned units,
    const T* stateC,
    const T* inputValue,
    const T* stateHValue,
    const T* bias,
    T forget,
    T* result,
    T* resultState) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  const unsigned total = batch * units;
  for (unsigned i = tid; i < total; i += skip) {
    unsigned row = i / units;
    unsigned col = i % units;
    const T* currentInputValue = inputValue + row * units * 4;
    const T* currentStateHValue = stateHValue + row * units * 4;

    T xi = currentInputValue[col] + currentStateHValue[col] + bias[col];
    T xc = currentInputValue[col + units] + currentStateHValue[col + units] + bias[col + units];
    T xf = currentInputValue[col + units * 2] + currentStateHValue[col + units * 2] + bias[col + units * 2];
    T xo = currentInputValue[col + units * 3] + currentStateHValue[col + units * 3] + bias[col + units * 3];

    xi = sigmoid(xi);
    xc = tanh(xc);
    xf = sigmoid(xf + forget);
    xo = sigmoid(xo);
    T cs = xc * xi + stateC[i] * xf;
    T rs = tanh(cs) * xo;
    result[i] = rs;
    resultState[i] = cs;
  }
}

template <class T>
static void __global__ _cuda_process_gru_cell_gates(
    unsigned batch,
    unsigned units,
    unsigned inputSize,
    const T* state,
    const T* inputs,
    const T* value,
    const T* bias,
    T* result) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  const unsigned total = batch * units;
  for (unsigned i = tid; i < total; i += skip) {
    unsigned row = i / units;
    unsigned col = i % units;
    T r = sigmoid(value[row * units * 2 + col] + bias[col]);
    T s = r * state[i];
    result[row * (inputSize + units) + col + inputSize] = s;
  }

  const unsigned inputTotal = batch * inputSize;
  for (unsigned i = tid; i < inputTotal; i += skip) {
    unsigned row = i / inputSize;
    unsigned col = i % inputSize;
    result[row * (inputSize + units) + col] = inputs[i];
  }
}

template <class T>
static void __global__ _cuda_process_gru_cell_candidate(
    unsigned batch,
    unsigned units,
    const T* state,
    const T* value,
    const T* bias,
    const T* cValue,
    const T* cBias,
    T* result) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  const unsigned total = batch * units;
  for (unsigned i = tid; i < total; i += skip) {
    unsigned row = i / units;
    unsigned col = i % units;
    T r = sigmoid(value[row * units * 2 + col] + bias[col]);
    T u = sigmoid(value[row * units * 2 + col + units] + bias[col + units]);
    T c = tanh(cValue[i] + cBias[col]);
    result[i] = u * state[i] + (static_cast<T>(1) - u) * c;
  }
}

template <class T>
static void __global__ _cuda_process_pytorch_gru_cell(
    unsigned batch,
    unsigned units,
    const T* state,
    const T* inputValue,
    const T* stateValue,
    const T* gateBias,
    const T* candidateIBias,
    const T* candidateHBias,
    T* result) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  const unsigned total = batch * units;
  for (unsigned i = tid; i < total; i += skip) {
    unsigned row = i / units;
    unsigned col = i % units;
    const T* currentInputValue = inputValue + row * units * 3;
    const T* currentStateValue = stateValue + row * units * 3;
    T r = currentInputValue[col] + currentStateValue[col] + gateBias[col];
    r = sigmoid(r);
    T z = currentInputValue[col + units] + currentStateValue[col + units] + gateBias[col + units];
    z = sigmoid(z);
    T ni = currentInputValue[col + units * 2] + candidateIBias[col];
    T nh = currentStateValue[col + units * 2] + candidateHBias[col];
    T n = tanh(ni + r * nh);
    result[i] = n + z * (state[i] - n);
  }
}

// helper functions
template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, std::tuple<Variable<T>, Variable<T>>>::type _process_lstm_cell_helper(
    const Tensor<T>& stateC,
    const Tensor<T>& inputValue,
    const Tensor<T>& stateHValue,
    const Tensor<T>& bias,
    T forget,
    size_t blockCount,
    size_t threadCount) {
  unsigned batch = stateC.shape(0);
  unsigned units = stateC.shape(1);
  Variable<T> result(stateC.shape());
  Variable<T> resultState(stateC.shape());
  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_process_lstm_cell<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      batch,
      units,
      stateC.data(),
      inputValue.data(),
      stateHValue.data(),
      bias.data(),
      forget,
      result.data(),
      resultState.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return std::make_tuple(std::move(result), std::move(resultState));
}

template <class T, class ST>
static inline std::tuple<Variable<T>, Variable<T>> _process_lstm_cell_helper(
    const Tensor<T>& stateC,
    const Tensor<T>& inputValue,
    const Tensor<T>& stateHValue,
    const Tensor<T>& bias,
    T forget,
    ST blockCount,
    ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type _process_gru_cell_gates_helper(
    const Tensor<T>& state,
    const Tensor<T>& inputs,
    const Tensor<T>& value,
    const Tensor<T>& bias,
    size_t blockCount,
    size_t threadCount) {
  unsigned batch = state.shape(0);
  unsigned units = state.shape(1);
  unsigned inputSize = inputs.shape(1);
  Variable<T> result({batch, inputSize + units});

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_process_gru_cell_gates<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      batch,
      units,
      inputSize,
      state.data(),
      inputs.data(),
      value.data(),
      bias.data(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _process_gru_cell_gates_helper(
    const Tensor<T>& state,
    const Tensor<T>& inputs,
    const Tensor<T>& value,
    const Tensor<T>& bias,
    ST blockCount,
    ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type _process_gru_cell_candidate_helper(
    const Tensor<T>& state,
    const Tensor<T>& value,
    const Tensor<T>& bias,
    const Tensor<T>& cValue,
    const Tensor<T>& cBias,
    size_t blockCount,
    size_t threadCount) {
  unsigned batch = state.shape(0);
  unsigned units = state.shape(1);
  Variable<T> result({batch, units});

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_process_gru_cell_candidate<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      batch,
      units,
      state.data(),
      value.data(),
      bias.data(),
      cValue.data(),
      cBias.data(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _process_gru_cell_candidate_helper(
    const Tensor<T>& state,
    const Tensor<T>& value,
    const Tensor<T>& bias,
    const Tensor<T>& cValue,
    const Tensor<T>& cBias,
    ST blockCount,
    ST threadCount) {
  throw NotImplementedError();
}

template <class T>
static inline typename std::enable_if<std::is_same<float, T>::value || std::is_same<double, T>::value, Variable<T>>::type _process_pytorch_gru_cell_helper(
    const Tensor<T>& state,
    const Tensor<T>& inputValue,
    const Tensor<T>& stateValue,
    const Tensor<T>& gateBias,
    const Tensor<T>& candidateIBias,
    const Tensor<T>& candidateHBias,
    size_t blockCount,
    size_t threadCount) {
  unsigned batch = state.shape(0);
  unsigned units = state.shape(1);
  Variable<T> result(state.shape());

  tfcc::CUDASession* session = static_cast<tfcc::CUDASession*>(Session::getThreadDefault());
  _cuda_process_pytorch_gru_cell<<<blockCount, threadCount, 0, session->getImpl()->cudaStream()>>>(
      batch,
      units,
      state.data(),
      inputValue.data(),
      stateValue.data(),
      gateBias.data(),
      candidateIBias.data(),
      candidateHBias.data(),
      result.data());
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  return result;
}

template <class T, class ST>
static inline Variable<T> _process_pytorch_gru_cell_helper(
    const Tensor<T>& state,
    const Tensor<T>& inputValue,
    const Tensor<T>& stateValue,
    const Tensor<T>& gateBias,
    const Tensor<T>& candidateIBias,
    const Tensor<T>& candidateHBias,
    ST blockCount,
    ST threadCount) {
  throw NotImplementedError();
}

//class function
template <class T>
CUDACellInterface<T>::CUDACellInterface(const CUDADeviceProperty& property)
  : _property(property) {
}

template <class T>
CUDACellInterface<T>::~CUDACellInterface() {
}

template <class T>
std::tuple<Variable<T>, Variable<T>> CUDACellInterface<T>::processLSTMCell(
    const Tensor<T>& stateC,
    const Tensor<T>& inputValue,
    const Tensor<T>& stateHValue,
    const Tensor<T>& bias,
    T forget) {
  if (stateC.shape().size() != 2)
    throw InvalidArgumentError("invalid stateC");
  unsigned batch = stateC.shape(0);
  unsigned units = stateC.shape(1);

  if (inputValue.shape().size() != 2 || inputValue.shape(0) != batch || inputValue.shape(1) != units * 4)
    throw InvalidArgumentError("invalid inputValue");
  if (stateHValue.shape().size() != 2 || stateHValue.shape(0) != batch || stateHValue.shape(1) != units * 4)
    throw InvalidArgumentError("invalid stateHValue");
  if (bias.shape().size() != 1 || bias.shape(0) != units * 4)
    throw InvalidArgumentError("invalid bias");

  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(batch * units);
  return _process_lstm_cell_helper(stateC, inputValue, stateHValue, bias, forget, blockCount, threadCount);
}

template <class T>
Variable<T> CUDACellInterface<T>::processGRUCellGates(
    const Tensor<T>& state,
    const Tensor<T>& inputs,
    const Tensor<T>& value,
    const Tensor<T>& bias) {
  if (state.shape().size() != 2)
    throw InvalidArgumentError("invalid state");
  unsigned batch = state.shape(0);
  unsigned units = state.shape(1);
  if (inputs.shape().size() != 2 || inputs.shape(0) != batch)
    throw InvalidArgumentError("invalid inputs");
  unsigned inputSize = inputs.shape(1);
  if (value.shape().size() != 2 || value.shape(0) != batch || value.shape(1) != units * 2)
    throw InvalidArgumentError("invalid value");
  if (bias.shape().size() != 1 || bias.shape(0) != units * 2)
    throw InvalidArgumentError("invalid bias");

  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(batch * units);
  return _process_gru_cell_gates_helper(state, inputs, value, bias, blockCount, threadCount);
}

template <class T>
Variable<T> CUDACellInterface<T>::processGRUCellCandidate(
    const Tensor<T>& state,
    const Tensor<T>& value,
    const Tensor<T>& bias,
    const Tensor<T>& cValue,
    const Tensor<T>& cBias) {
  if (state.shape().size() != 2)
    throw InvalidArgumentError("invalid state");
  unsigned batch = state.shape(0);
  unsigned units = state.shape(1);
  if (value.shape().size() != 2 || value.shape(0) != batch || value.shape(1) != units * 2)
    throw InvalidArgumentError("invalid value");
  if (bias.shape().size() != 1 || bias.shape(0) != units * 2)
    throw InvalidArgumentError("invalid bias");
  if (cValue.shape().size() != 2 || cValue.shape(0) != batch || cValue.shape(1) != units)
    throw InvalidArgumentError("invalid cValue");
  if (cBias.shape().size() != 1 || cBias.shape(0) != units)
    throw InvalidArgumentError("invalid cBias");

  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(batch * units);
  return _process_gru_cell_candidate_helper(state, value, bias, cValue, cBias, blockCount, threadCount);
}

template <class T>
Variable<T> CUDACellInterface<T>::processPyTorchGRUCell(
    const Tensor<T>& state,
    const Tensor<T>& inputValue,
    const Tensor<T>& stateValue,
    const Tensor<T>& gateBias,
    const Tensor<T>& candidateIBias,
    const Tensor<T>& candidateHBias) {
  if (state.shape().size() != 2)
    throw InvalidArgumentError("invalid state");

  unsigned batch = state.shape(0);
  unsigned units = state.shape(1);

  if (inputValue.shape().size() != 2 || inputValue.shape(0) != batch || inputValue.shape(1) != units * 3)
    throw InvalidArgumentError("invalid inputValue");
  if (stateValue.shape().size() != 2 || stateValue.shape(0) != batch || stateValue.shape(1) != units * 3)
    throw InvalidArgumentError("invalid stateValue");
  if (gateBias.shape().size() != 1 || gateBias.shape(0) != units * 2)
    throw InvalidArgumentError("invalid gateBias");
  if (candidateIBias.shape().size() != 1 || candidateIBias.shape(0) != units)
    throw InvalidArgumentError("invalid candidateIBias");
  if (candidateHBias.shape().size() != 1 || candidateHBias.shape(0) != units)
    throw InvalidArgumentError("invalid candidateHBias");

  Variable<T> result(state.shape());

  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = _property.getSuitableKernelSize(batch * units);
  return _process_pytorch_gru_cell_helper(state, inputValue, stateValue, gateBias, candidateIBias, candidateHBias, blockCount, threadCount);
}

#define DEFINE_FUNC(type) template class CUDACellInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
