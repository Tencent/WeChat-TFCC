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
#include <algorithm>
#include <cstring>

namespace tfcc {

template <class T>
static inline T sigmoid(T v, T one, T zero) {
  return one / (one + exp(zero - v));
}

template <class VCL>
static inline void _processLSTMCell(
    unsigned batch, unsigned units, const typename VCL::BaseType* stateC,
    const typename VCL::BaseType* inputValue, const typename VCL::BaseType* stateHValue,
    const typename VCL::BaseType* bias, typename VCL::BaseType forget,
    typename VCL::BaseType* result, typename VCL::BaseType* resultState) {
  typedef typename VCL::BaseType BaseType;
  unsigned pUnits = (units / VCL::size()) * VCL::size();
  VCL one(1.f);
  VCL zero(0.f);
  VCL vf(forget);
#pragma omp parallel for
  for (unsigned i = 0; i < pUnits; i += VCL::size()) {
    const BaseType* currentStateC = stateC + i;
    const BaseType* currentInputValue = inputValue + i;
    const BaseType* currentStateHValue = stateHValue + i;
    BaseType* currentResult = result + i;
    BaseType* currentResultState = resultState + i;
    VCL oib(bias + i);
    VCL ocb(bias + i + units);
    VCL ofb(bias + i + units * 2);
    VCL oob(bias + i + units * 3);
    for (unsigned b = 0; b < batch; ++b) {
      VCL xi = VCL(currentInputValue) + VCL(currentStateHValue) + oib;
      VCL xc = VCL(currentInputValue + units) + VCL(currentStateHValue + units) + ocb;
      VCL xf = VCL(currentInputValue + units * 2) + VCL(currentStateHValue + units * 2) + ofb;
      VCL xo = VCL(currentInputValue + units * 3) + VCL(currentStateHValue + units * 3) + oob;

      xi = sigmoid(xi, one, zero);
      xc = tanh(xc);
      xf = sigmoid(xf + vf, one, zero);
      xo = sigmoid(xo, one, zero);
      VCL cs = xc * xi + VCL(currentStateC) * xf;
      VCL rs = tanh(cs) * xo;
      rs.store(currentResult);
      cs.store(currentResultState);

      currentStateC += units;
      currentInputValue += units * 4;
      currentStateHValue += units * 4;
      currentResult += units;
      currentResultState += units;
    }
  }

  if (pUnits < units) {
    const BaseType* currentStateC = stateC + pUnits;
    const BaseType* currentInputValue = inputValue + pUnits;
    const BaseType* currentStateHValue = stateHValue + pUnits;
    BaseType* currentResult = result + pUnits;
    BaseType* currentResultState = resultState + pUnits;
    VCL oib(bias + pUnits);
    VCL ocb(bias + pUnits + units);
    VCL ofb(bias + pUnits + units * 2);
    VCL oob(bias + pUnits + units * 3);
    for (unsigned b = 0; b < batch; ++b) {
      VCL xi = VCL(currentInputValue) + VCL(currentStateHValue) + oib;
      VCL xc = VCL(currentInputValue + units) + VCL(currentStateHValue + units) + ocb;
      VCL xf = VCL(currentInputValue + units * 2) + VCL(currentStateHValue + units * 2) + ofb;
      VCL xo = VCL(currentInputValue + units * 3) + VCL(currentStateHValue + units * 3) + oob;

      xi = sigmoid(xi, one, zero);
      xc = tanh(xc);
      xf = sigmoid(xf + vf, one, zero);
      xo = sigmoid(xo, one, zero);
      VCL cs = xc * xi + VCL(currentStateC) * xf;
      VCL rs = tanh(cs) * xo;
      BaseType rsTmp[VCL::size()];
      BaseType csTmp[VCL::size()];
      rs.store(rsTmp);
      cs.store(csTmp);
      for (unsigned i = 0; i < units - pUnits; ++i) {
        currentResult[i] = rsTmp[i];
        currentResultState[i] = csTmp[i];
      }

      currentStateC += units;
      currentInputValue += units * 4;
      currentStateHValue += units * 4;
      currentResult += units;
      currentResultState += units;
    }
  }
}

template <class VCL>
static inline void _processGRUCellGates(
    unsigned batch, unsigned units, unsigned inputSize, const typename VCL::BaseType* state,
    const typename VCL::BaseType* inputs, const typename VCL::BaseType* value,
    const typename VCL::BaseType* bias, typename VCL::BaseType* result) {
  typedef typename VCL::BaseType BaseType;
  unsigned pUnits = (units / VCL::size()) * VCL::size();
  VCL one(1.f);
  VCL zero(0.f);
#pragma omp parallel for
  for (unsigned i = 0; i < pUnits; i += VCL::size()) {
    const BaseType* currentState = state + i;
    const BaseType* currentValue = value + i;
    BaseType* currentResult = result + i;
    VCL ob(bias + i);
    for (unsigned b = 0; b < batch; ++b) {
      VCL r = sigmoid(VCL(currentValue) + ob, one, zero);
      VCL s = r * VCL(currentState);
      s.store(currentResult + inputSize);

      currentState += units;
      currentValue += 2 * units;
      currentResult += inputSize + units;
    }
  }
  if (pUnits < units) {
    const BaseType* currentState = state + pUnits;
    const BaseType* currentValue = value + pUnits;
    BaseType* currentResult = result + pUnits;
    VCL ob(bias + pUnits);
    for (unsigned b = 0; b < batch; ++b) {
      VCL r = sigmoid(VCL(currentValue) + ob, one, zero);
      VCL s = r * VCL(currentState);
      BaseType rsTmp[VCL::size()];
      s.store(rsTmp);
      for (unsigned i = 0; i < units - pUnits; ++i) currentResult[i + inputSize] = rsTmp[i];

      currentState += units;
      currentValue += 2 * units;
      currentResult += inputSize + units;
    }
  }

  unsigned maxThread = omp_get_max_threads();
  if (maxThread <= batch) {
#pragma omp parallel for
    for (unsigned i = 0; i < batch; ++i) {
      memcpy(
          result + (inputSize + units) * i, inputs + inputSize * i, inputSize * sizeof(BaseType));
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
            (end - start) * sizeof(BaseType));
      }
    }
  }
}

template <class VCL>
static inline void _processGRUCellCandidate(
    unsigned batch, unsigned units, const typename VCL::BaseType* state,
    const typename VCL::BaseType* value, const typename VCL::BaseType* bias,
    const typename VCL::BaseType* cValue, const typename VCL::BaseType* cBias,
    typename VCL::BaseType* result) {
  typedef typename VCL::BaseType BaseType;
  unsigned pUnits = (units / VCL::size()) * VCL::size();
  VCL one(1.f);
  VCL zero(0.f);
#pragma omp parallel for
  for (unsigned i = 0; i < pUnits; i += VCL::size()) {
    const BaseType* currentState = state + i;
    const BaseType* currentValue = value + i + units;
    const BaseType* currentCValue = cValue + i;
    BaseType* currentResult = result + i;
    VCL ob(bias + i + units);
    VCL ocb(cBias + i);
    for (unsigned b = 0; b < batch; ++b) {
      VCL u = sigmoid(VCL(currentValue) + ob, one, zero);
      VCL c = tanh(VCL(currentCValue) + ocb);
      VCL r = u * VCL(currentState) + (one - u) * c;
      r.store(currentResult);

      currentState += units;
      currentValue += 2 * units;
      currentCValue += units;
      currentResult += units;
    }
  }

  if (pUnits < units) {
    const BaseType* currentState = state + pUnits;
    const BaseType* currentValue = value + pUnits + units;
    const BaseType* currentCValue = cValue + pUnits;
    BaseType* currentResult = result + pUnits;
    VCL ob(bias + pUnits + units);
    VCL ocb(cBias + pUnits);
    for (unsigned b = 0; b < batch; ++b) {
      VCL u = sigmoid(VCL(currentValue) + ob, one, zero);
      VCL c = tanh(VCL(currentCValue) + ocb);
      VCL r = u * VCL(currentState) + (one - u) * c;
      BaseType rsTmp[VCL::size()];
      r.store(rsTmp);
      for (unsigned i = 0; i < units - pUnits; ++i) currentResult[i] = rsTmp[i];

      currentState += units;
      currentValue += 2 * units;
      currentCValue += units;
      currentResult += units;
    }
  }
}

template <class VCL>
static inline void _processPyTorchGRUCell(
    unsigned batch, unsigned units, const typename VCL::BaseType* state,
    const typename VCL::BaseType* inputValue, const typename VCL::BaseType* stateValue,
    const typename VCL::BaseType* gateBias, const typename VCL::BaseType* candidateIBias,
    const typename VCL::BaseType* candidateHBias, typename VCL::BaseType* result) {
  typedef typename VCL::BaseType BaseType;
  unsigned pUnits = (units / VCL::size()) * VCL::size();
  VCL one(1.f);
  VCL zero(0.f);
#pragma omp parallel for
  for (unsigned i = 0; i < pUnits; i += VCL::size()) {
    const BaseType* currentState = state + i;
    const BaseType* currentInputValue = inputValue + i;
    const BaseType* currentStateValue = stateValue + i;
    BaseType* currentResult = result + i;
    VCL ogr(gateBias + i);
    VCL ogz(gateBias + i + units);
    VCL oci(candidateIBias + i);
    VCL och(candidateHBias + i);
    for (unsigned b = 0; b < batch; ++b) {
      VCL r = VCL(currentInputValue) + VCL(currentStateValue) + ogr;
      r = sigmoid(r, one, zero);
      VCL z = VCL(currentInputValue + units) + VCL(currentStateValue + units) + ogz;
      z = sigmoid(z, one, zero);
      VCL ni = VCL(currentInputValue + 2 * units) + oci;
      VCL nh = VCL(currentStateValue + 2 * units) + och;
      VCL n = tanh(ni + r * nh);
      VCL rs = n + z * (VCL(currentState) - n);
      rs.store(currentResult);

      currentState += units;
      currentInputValue += 3 * units;
      currentStateValue += 3 * units;
      currentResult += units;
    }
  }

  if (pUnits < units) {
    const BaseType* currentState = state + pUnits;
    const BaseType* currentInputValue = inputValue + pUnits;
    const BaseType* currentStateValue = stateValue + pUnits;
    BaseType* currentResult = result + pUnits;
    VCL ogr(gateBias + pUnits);
    VCL ogz(gateBias + pUnits + units);
    VCL oci(candidateIBias + pUnits);
    VCL och(candidateHBias + pUnits);
    for (unsigned b = 0; b < batch; ++b) {
      VCL r = VCL(currentInputValue) + VCL(currentStateValue) + ogr;
      r = sigmoid(r, one, zero);
      VCL z = VCL(currentInputValue + units) + VCL(currentStateValue + units) + ogz;
      z = sigmoid(z, one, zero);
      VCL ni = VCL(currentInputValue + 2 * units) + oci;
      VCL nh = VCL(currentStateValue + 2 * units) + och;
      VCL n = tanh(ni + r * nh);
      VCL rs = n + z * (VCL(currentState) - n);
      BaseType rsTmp[VCL::size()];
      rs.store(rsTmp);
      for (unsigned i = 0; i < units - pUnits; ++i) currentResult[i] = rsTmp[i];

      currentState += units;
      currentInputValue += 3 * units;
      currentStateValue += 3 * units;
      currentResult += units;
    }
  }
}

}  // namespace tfcc
