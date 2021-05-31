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

#include "tfcc_mkltransformationkernel.avx512.h"

#include "framework/tfcc_types.h"
#include "vcl/tfcc_mklvcl16f.hpp"
#include "vcl/tfcc_mklvcl8d.hpp"

namespace tfcc {

// float
void _MKLTransformationKernelAVX512<float>::transform(
    const float* a, unsigned total, float alpha, float beta, float* b) {
  MKLVcl16f vAlpha(alpha);
  MKLVcl16f vBeta(beta);
#pragma omp parallel for firstprivate(vAlpha, vBeta)
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    MKLVcl16f va;
    va.load(a + i);
    MKLVcl16f vb = va * vAlpha + vBeta;
    vb.store(b + i);
  }
}

void _MKLTransformationKernelAVX512<float>::transform2(
    const float* a, unsigned total, float alpha, float beta, float* b) {
  MKLVcl16f vAlpha(alpha);
  MKLVcl16f vBeta(beta);
#pragma omp parallel for firstprivate(vAlpha, vBeta)
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    MKLVcl16f va;
    va.load(a + i);
    MKLVcl16f vb = va / vAlpha + vBeta;
    vb.store(b + i);
  }
}

void _MKLTransformationKernelAVX512<float>::transform3(
    const float* a, unsigned total, float alpha, float beta, float* b) {
  MKLVcl16f vAlpha(alpha);
  MKLVcl16f vBeta(beta);
#pragma omp parallel for firstprivate(vAlpha, vBeta)
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    MKLVcl16f va;
    va.load(a + i);
    MKLVcl16f vb = vAlpha / va + vBeta;
    vb.store(b + i);
  }
}

void _MKLTransformationKernelAVX512<float>::transform4(
    const float* a, unsigned total, float alpha, float beta, float* b) {
  MKLVcl16f vAlpha(alpha);
  MKLVcl16f vBeta(beta);
#pragma omp parallel for firstprivate(vAlpha, vBeta)
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    MKLVcl16f va;
    va.load(a + i);
    MKLVcl16f vb = vBeta - va * vAlpha;
    vb.store(b + i);
  }
}

void _MKLTransformationKernelAVX512<float>::transform5(
    const float* a, unsigned total, float alpha, float beta, float* b) {
  MKLVcl16f vAlpha(alpha);
  MKLVcl16f vBeta(beta);
#pragma omp parallel for firstprivate(vAlpha, vBeta)
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    MKLVcl16f va;
    va.load(a + i);
    MKLVcl16f vb = vBeta - va / vAlpha;
    vb.store(b + i);
  }
}

void _MKLTransformationKernelAVX512<float>::transform6(
    const float* a, unsigned total, float alpha, float beta, float* b) {
  MKLVcl16f vAlpha(alpha);
  MKLVcl16f vBeta(beta);
#pragma omp parallel for firstprivate(vAlpha, vBeta)
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    MKLVcl16f va;
    va.load(a + i);
    MKLVcl16f vb = vBeta - vAlpha / va;
    vb.store(b + i);
  }
}

// double
void _MKLTransformationKernelAVX512<double>::transform(
    const double* a, unsigned total, double alpha, double beta, double* b) {
  MKLVcl8d vAlpha(alpha);
  MKLVcl8d vBeta(beta);
#pragma omp parallel for firstprivate(vAlpha, vBeta)
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    MKLVcl8d va;
    va.load(a + i);
    MKLVcl8d vb = va * vAlpha + vBeta;
    vb.store(b + i);
  }
}

void _MKLTransformationKernelAVX512<double>::transform2(
    const double* a, unsigned total, double alpha, double beta, double* b) {
  MKLVcl8d vAlpha(alpha);
  MKLVcl8d vBeta(beta);
#pragma omp parallel for firstprivate(vAlpha, vBeta)
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    MKLVcl8d va;
    va.load(a + i);
    MKLVcl8d vb = va / vAlpha + vBeta;
    vb.store(b + i);
  }
}

void _MKLTransformationKernelAVX512<double>::transform3(
    const double* a, unsigned total, double alpha, double beta, double* b) {
  MKLVcl8d vAlpha(alpha);
  MKLVcl8d vBeta(beta);
#pragma omp parallel for firstprivate(vAlpha, vBeta)
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    MKLVcl8d va;
    va.load(a + i);
    MKLVcl8d vb = vAlpha / va + vBeta;
    vb.store(b + i);
  }
}

void _MKLTransformationKernelAVX512<double>::transform4(
    const double* a, unsigned total, double alpha, double beta, double* b) {
  MKLVcl8d vAlpha(alpha);
  MKLVcl8d vBeta(beta);
#pragma omp parallel for firstprivate(vAlpha, vBeta)
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    MKLVcl8d va;
    va.load(a + i);
    MKLVcl8d vb = vBeta - va * vAlpha;
    vb.store(b + i);
  }
}

void _MKLTransformationKernelAVX512<double>::transform5(
    const double* a, unsigned total, double alpha, double beta, double* b) {
  MKLVcl8d vAlpha(alpha);
  MKLVcl8d vBeta(beta);
#pragma omp parallel for firstprivate(vAlpha, vBeta)
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    MKLVcl8d va;
    va.load(a + i);
    MKLVcl8d vb = vBeta - va / vAlpha;
    vb.store(b + i);
  }
}

void _MKLTransformationKernelAVX512<double>::transform6(
    const double* a, unsigned total, double alpha, double beta, double* b) {
  MKLVcl8d vAlpha(alpha);
  MKLVcl8d vBeta(beta);
#pragma omp parallel for firstprivate(vAlpha, vBeta)
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    MKLVcl8d va;
    va.load(a + i);
    MKLVcl8d vb = vBeta - vAlpha / va;
    vb.store(b + i);
  }
}

#define DEFINE_FUNC(type) template class _MKLTransformationKernelAVX512<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
