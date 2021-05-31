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

#include "tfcc_mklcellkernel.avx512.h"

#include "vcl/tfcc_mklvcl16f.hpp"
#include "vcl/tfcc_mklvcl8d.hpp"

#include "tfcc_mklcellkernel.avximpl.hpp"

namespace tfcc {

void _MKLCellKernelAVX512<float>::processLSTMCell(
    unsigned batch, unsigned units, const float* stateC, const float* inputValue,
    const float* stateHValue, const float* bias, float forget, float* result, float* resultState) {
  _processLSTMCell<MKLVcl16f>(
      batch, units, stateC, inputValue, stateHValue, bias, forget, result, resultState);
}

void _MKLCellKernelAVX512<float>::processGRUCellGates(
    unsigned batch, unsigned units, unsigned inputSize, const float* state, const float* inputs,
    const float* value, const float* bias, float* result) {
  _processGRUCellGates<MKLVcl16f>(batch, units, inputSize, state, inputs, value, bias, result);
}

void _MKLCellKernelAVX512<float>::processGRUCellCandidate(
    unsigned batch, unsigned units, const float* state, const float* value, const float* bias,
    const float* cValue, const float* cBias, float* result) {
  _processGRUCellCandidate<MKLVcl16f>(batch, units, state, value, bias, cValue, cBias, result);
}

void _MKLCellKernelAVX512<float>::processPyTorchGRUCell(
    unsigned batch, unsigned units, const float* state, const float* inputValue,
    const float* stateValue, const float* gateBias, const float* candidateIBias,
    const float* candidateHBias, float* result) {
  _processPyTorchGRUCell<MKLVcl16f>(
      batch, units, state, inputValue, stateValue, gateBias, candidateIBias, candidateHBias,
      result);
}

// double
void _MKLCellKernelAVX512<double>::processLSTMCell(
    unsigned batch, unsigned units, const double* stateC, const double* inputValue,
    const double* stateHValue, const double* bias, double forget, double* result,
    double* resultState) {
  _processLSTMCell<MKLVcl8d>(
      batch, units, stateC, inputValue, stateHValue, bias, forget, result, resultState);
}

void _MKLCellKernelAVX512<double>::processGRUCellGates(
    unsigned batch, unsigned units, unsigned inputSize, const double* state, const double* inputs,
    const double* value, const double* bias, double* result) {
  _processGRUCellGates<MKLVcl8d>(batch, units, inputSize, state, inputs, value, bias, result);
}

void _MKLCellKernelAVX512<double>::processGRUCellCandidate(
    unsigned batch, unsigned units, const double* state, const double* value, const double* bias,
    const double* cValue, const double* cBias, double* result) {
  _processGRUCellCandidate<MKLVcl8d>(batch, units, state, value, bias, cValue, cBias, result);
}

void _MKLCellKernelAVX512<double>::processPyTorchGRUCell(
    unsigned batch, unsigned units, const double* state, const double* inputValue,
    const double* stateValue, const double* gateBias, const double* candidateIBias,
    const double* candidateHBias, double* result) {
  _processPyTorchGRUCell<MKLVcl8d>(
      batch, units, state, inputValue, stateValue, gateBias, candidateIBias, candidateHBias,
      result);
}

}  // namespace tfcc
