// Copyright 2021 Wechat Group, Tencent
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <omp.h>
#include <cmath>
#include <cstring>

namespace tfcc {

template <class T>
class _MKLCellKernel {
  static inline T sigmoid(T v);

 public:
  static inline void processLSTMCell(
      unsigned batch, unsigned units, const T* stateC, const T* inputValue, const T* stateHValue,
      const T* bias, T forget, T* result, T* resultState);

  static inline void processGRUCellGates(
      unsigned batch, unsigned units, unsigned inputSize, const T* state, const T* inputs,
      const T* value, const T* bias, T* result);

  static inline void processGRUCellCandidate(
      unsigned batch, unsigned units, const T* state, const T* value, const T* bias,
      const T* cValue, const T* cBias, T* result);

  static inline void processPyTorchGRUCell(
      unsigned batch, unsigned units, const T* state, const T* inputValue, const T* stateValue,
      const T* gateBias, const T* candidateIBias, const T* candidateHBias, T* result);
};

template <class T>
inline T _MKLCellKernel<T>::sigmoid(T v) {
  return static_cast<T>(1 / (1 + std::exp(0 - v)));
}

template <class T>
inline void _MKLCellKernel<T>::processLSTMCell(
    unsigned batch, unsigned units, const T* stateC, const T* inputValue, const T* stateHValue,
    const T* bias, T forget, T* result, T* resultState) {
#pragma omp parallel for
  for (unsigned i = 0; i < units; ++i) {
    const T* currentStateC = stateC + i;
    const T* currentInputValue = inputValue + i;
    const T* currentStateHValue = stateHValue + i;
    T* currentResult = result + i;
    T* currentResultState = resultState + i;
    T oib = bias[i];
    T ocb = bias[i + units];
    T ofb = bias[i + units * 2];
    T oob = bias[i + units * 3];
    for (unsigned b = 0; b < batch; ++b) {
      T xi = currentInputValue[0] + currentStateHValue[0] + oib;
      T xc = currentInputValue[units] + currentStateHValue[units] + ocb;
      T xf = currentInputValue[units * 2] + currentStateHValue[units * 2] + ofb;
      T xo = currentInputValue[units * 3] + currentStateHValue[units * 3] + oob;

      xi = sigmoid(xi);
      xc = static_cast<T>(std::tanh(xc));
      xf = sigmoid(xf + forget);
      xo = sigmoid(xo);
      T cs = xc * xi + currentStateC[0] * xf;
      T rs = tanh(cs) * xo;
      currentResult[0] = rs;
      currentResultState[0] = cs;

      currentStateC += units;
      currentInputValue += units * 4;
      currentStateHValue += units * 4;
      currentResult += units;
      currentResultState += units;
    }
  }
}

template <class T>
inline void _MKLCellKernel<T>::processGRUCellGates(
    unsigned batch, unsigned units, unsigned inputSize, const T* state, const T* inputs,
    const T* value, const T* bias, T* result) {
#pragma omp parallel for
  for (unsigned i = 0; i < units; ++i) {
    const T* currentState = state + i;
    const T* currentValue = value + i;
    T* currentResult = result + i;
    T ob = bias[i];
    for (unsigned b = 0; b < batch; ++b) {
      T r = sigmoid(currentValue[0] + ob);
      T s = r * currentState[0];
      currentResult[inputSize] = s;

      currentState += units;
      currentValue += 2 * units;
      currentResult += inputSize + units;
    }
  }

  unsigned maxThread = omp_get_max_threads();
  if (maxThread <= batch) {
#pragma omp parallel for
    for (unsigned i = 0; i < batch; ++i) {
      memcpy(result + (inputSize + units) * i, inputs + inputSize * i, inputSize * sizeof(T));
    }
  } else {
#pragma omp parallel
    {
      unsigned threadCount = omp_get_num_threads();
      unsigned num = (inputSize + threadCount - 1) / threadCount;
      num = std::max(1u, num);
      unsigned threadId = omp_get_thread_num();
      unsigned start = threadId * num;
      start = std::min(start, inputSize);
      unsigned end = start + num;
      end = std::min(end, inputSize);
      for (unsigned i = 0; i < batch; ++i) {
        memcpy(
            result + (inputSize + units) * i + start, inputs + inputSize * i + start,
            (end - start) * sizeof(T));
      }
    }
  }
}

template <class T>
inline void _MKLCellKernel<T>::processGRUCellCandidate(
    unsigned batch, unsigned units, const T* state, const T* value, const T* bias, const T* cValue,
    const T* cBias, T* result) {
#pragma omp parallel for
  for (unsigned i = 0; i < units; ++i) {
    const T* currentState = state + i;
    const T* currentValue = value + i + units;
    const T* currentCValue = cValue + i;
    T* currentResult = result + i;
    T ob = bias[i + units];
    T ocb = cBias[i];
    for (unsigned b = 0; b < batch; ++b) {
      T u = sigmoid(currentValue[0] + ob);
      T c = static_cast<T>(tanh(currentCValue[0] + ocb));
      currentResult[0] = u * currentState[0] + (static_cast<T>(1) - u) * c;

      currentState += units;
      currentValue += 2 * units;
      currentCValue += units;
      currentResult += units;
    }
  }
}

template <class T>
inline void _MKLCellKernel<T>::processPyTorchGRUCell(
    unsigned batch, unsigned units, const T* state, const T* inputValue, const T* stateValue,
    const T* gateBias, const T* candidateIBias, const T* candidateHBias, T* result) {
#pragma omp parallel for
  for (unsigned i = 0; i < units; ++i) {
    const T* currentState = state + i;
    const T* currentInputValue = inputValue + i;
    const T* currentStateValue = stateValue + i;
    T* currentResult = result + i;
    T ogr = gateBias[i];
    T ogz = gateBias[i + units];
    T oci = candidateIBias[i];
    T och = candidateHBias[i];
    for (unsigned b = 0; b < batch; ++b) {
      T r = currentInputValue[0] + currentStateValue[0] + ogr;
      r = sigmoid(r);
      T z = currentInputValue[units] + currentStateValue[units] + ogz;
      z = sigmoid(z);
      T ni = currentInputValue[2 * units] + oci;
      T nh = currentStateValue[2 * units] + och;
      T n = static_cast<T>(std::tanh(ni + r * nh));
      currentResult[0] = n + z * (currentState[0] - n);

      currentState += units;
      currentInputValue += 3 * units;
      currentStateValue += 3 * units;
      currentResult += units;
    }
  }
}

}  // namespace tfcc
