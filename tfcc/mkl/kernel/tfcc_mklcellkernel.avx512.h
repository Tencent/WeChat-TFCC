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

namespace tfcc {

template <class T>
class _MKLCellKernelAVX512 {
 public:
};

#ifdef TFCC_USE_AVX512

template <>
class _MKLCellKernelAVX512<float> {
 public:
  static void processLSTMCell(
      unsigned batch, unsigned units, const float* stateC, const float* inputValue,
      const float* stateHValue, const float* bias, float forget, float* result, float* resultState);

  static void processGRUCellGates(
      unsigned batch, unsigned units, unsigned inputSize, const float* state, const float* inputs,
      const float* value, const float* bias, float* result);

  static void processGRUCellCandidate(
      unsigned batch, unsigned units, const float* state, const float* value, const float* bias,
      const float* cValue, const float* cBias, float* result);

  static void processPyTorchGRUCell(
      unsigned batch, unsigned units, const float* state, const float* inputValue,
      const float* stateValue, const float* gateBias, const float* candidateIBias,
      const float* candidateHBias, float* result);
};

template <>
class _MKLCellKernelAVX512<double> {
 public:
  static void processLSTMCell(
      unsigned batch, unsigned units, const double* stateC, const double* inputValue,
      const double* stateHValue, const double* bias, double forget, double* result,
      double* resultState);

  static void processGRUCellGates(
      unsigned batch, unsigned units, unsigned inputSize, const double* state, const double* inputs,
      const double* value, const double* bias, double* result);

  static void processGRUCellCandidate(
      unsigned batch, unsigned units, const double* state, const double* value, const double* bias,
      const double* cValue, const double* cBias, double* result);

  static void processPyTorchGRUCell(
      unsigned batch, unsigned units, const double* state, const double* inputValue,
      const double* stateValue, const double* gateBias, const double* candidateIBias,
      const double* candidateHBias, double* result);
};

#endif

}  // namespace tfcc
