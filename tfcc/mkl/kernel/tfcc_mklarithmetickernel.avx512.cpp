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

#include "tfcc_mklarithmetickernel.avx512.h"

#include "framework/tfcc_types.h"
#include "vcl/tfcc_mklvcl16f.hpp"
#include "vcl/tfcc_mklvcl8d.hpp"

namespace tfcc {

// float
void _MKLArithmeticKernelAVX512<float>::batchAdd(
    const float* a, const float* b, float* c, unsigned total) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    MKLVcl16f va, vb;
    va.load(a + i);
    vb.load(b + i);
    MKLVcl16f vc = va + vb;
    vc.store(c + i);
  }
}

void _MKLArithmeticKernelAVX512<float>::batchSub(
    const float* a, const float* b, float* c, unsigned total) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    MKLVcl16f va, vb;
    va.load(a + i);
    vb.load(b + i);
    MKLVcl16f vc = va - vb;
    vc.store(c + i);
  }
}

void _MKLArithmeticKernelAVX512<float>::batchMul(
    const float* a, const float* b, float* c, unsigned total) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    MKLVcl16f va, vb;
    va.load(a + i);
    vb.load(b + i);
    MKLVcl16f vc = va * vb;
    vc.store(c + i);
  }
}

void _MKLArithmeticKernelAVX512<float>::batchDiv(
    const float* a, const float* b, float* c, unsigned total) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    MKLVcl16f va, vb;
    va.load(a + i);
    vb.load(b + i);
    MKLVcl16f vc = va / vb;
    vc.store(c + i);
  }
}

// double
void _MKLArithmeticKernelAVX512<double>::batchAdd(
    const double* a, const double* b, double* c, unsigned total) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    MKLVcl8d va, vb;
    va.load(a + i);
    vb.load(b + i);
    MKLVcl8d vc = va + vb;
    vc.store(c + i);
  }
}

void _MKLArithmeticKernelAVX512<double>::batchSub(
    const double* a, const double* b, double* c, unsigned total) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    MKLVcl8d va, vb;
    va.load(a + i);
    vb.load(b + i);
    MKLVcl8d vc = va - vb;
    vc.store(c + i);
  }
}

void _MKLArithmeticKernelAVX512<double>::batchMul(
    const double* a, const double* b, double* c, unsigned total) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    MKLVcl8d va, vb;
    va.load(a + i);
    vb.load(b + i);
    MKLVcl8d vc = va * vb;
    vc.store(c + i);
  }
}

void _MKLArithmeticKernelAVX512<double>::batchDiv(
    const double* a, const double* b, double* c, unsigned total) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    MKLVcl8d va, vb;
    va.load(a + i);
    vb.load(b + i);
    MKLVcl8d vc = va / vb;
    vc.store(c + i);
  }
}

#define DEFINE_FUNC(type) template class _MKLArithmeticKernelAVX512<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
