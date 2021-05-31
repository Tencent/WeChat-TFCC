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

#include "tfcc_mklactivationkernel.avx512.h"

#include <algorithm>

#define GCC_VERSION ((__GNUC__)*10000 + (__GNUC_MINOR__)*100 + (__GNUC_PATCHLEVEL__))
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) && GCC_VERSION == 70500
#  include <instrset.h>
__m512i _mm512_set_epi16(
    short e31, short e30, short e29, short e28, short e27, short e26, short e25, short e24,
    short e23, short e22, short e21, short e20, short e19, short e18, short e17, short e16,
    short e15, short e14, short e13, short e12, short e11, short e10, short e9, short e8, short e7,
    short e6, short e5, short e4, short e3, short e2, short e1, short e0);
#endif
#undef GCC_VERSION
#define MAX_VECTOR_SIZE 512
#define VCL_NAMESPACE vcl
#include "vectorclass.h"
#ifdef TFCC_USE_SVML
namespace vcl {
vcl::Vec8f cexp(vcl::Vec8f const& x);
vcl::Vec4d cexp(vcl::Vec4d const& x);
}  // namespace vcl
#  include "vectormath_lib.h"
#else
#  include "vectormath_exp.h"
#  include "vectormath_hyp.h"
#  include "vectormath_trig.h"
#endif

#include "framework/tfcc_types.h"
#include "vcl/tfcc_mklvcl16f.hpp"
#include "vcl/tfcc_mklvcl8d.hpp"

namespace tfcc {

// float
void _MKLActivationKernelAVX512<float>::sigmoid(const float* a, unsigned total, float* b) {
  MKLVcl16f one(1.f);
  MKLVcl16f zero(0.f);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    MKLVcl16f v;
    v.load(a + i);
    v = one / (one + exp(zero - v));
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<float>::tanh(const float* a, unsigned total, float* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    vcl::Vec16f v;
    v.load(a + i);
    v = vcl::tanh(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<float>::relu(const float* a, unsigned total, float* b) {
  MKLVcl16f zero(0.f);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    MKLVcl16f v;
    v.load(a + i);
    v = max(v, zero);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<float>::leakyRelu(
    const float* a, unsigned total, float alpha, float* b) {
  MKLVcl16f m(alpha);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    MKLVcl16f v;
    v.load(a + i);
    v = max(v, v * m);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<float>::log(const float* a, unsigned total, float* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    MKLVcl16f v;
    v.load(a + i);
    v = ::tfcc::log(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<float>::rsqrt(const float* a, unsigned total, float* b) {
  vcl::Vec16f one(1.0f);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    vcl::Vec16f v;
    v.load(a + i);
    v = one / vcl::sqrt(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<float>::softmax(
    const float* a, unsigned s1, unsigned s2, unsigned s3, float* b) {
  float maxValue = std::numeric_limits<float>::lowest();
  unsigned total = s1 * s2 * s3;
#pragma omp parallel for reduction(max : maxValue)
  for (unsigned i = 0; i < total; ++i) {
    maxValue = std::max(maxValue, a[i]);
  }
  MKLVcl16f vMaxValue(maxValue);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    MKLVcl16f v;
    v.load(a + i);
    v = exp(v - vMaxValue);
    v.store(b + i);
  }

  unsigned batch = s1 * s3;
#pragma omp parallel for
  for (unsigned i = 0; i < batch; ++i) {
    unsigned x = i / s3;
    unsigned z = i % s3;
    float sumValue = static_cast<float>(0);
    for (unsigned y = 0; y < s2; ++y) {
      sumValue += b[z + y * s3 + x * s3 * s2];
    }
    for (unsigned y = 0; y < s2; ++y) {
      b[z + y * s3 + x * s3 * s2] /= sumValue;
    }
  }
}

void _MKLActivationKernelAVX512<float>::softmaxV2(
    const float* a, unsigned s1, unsigned s2, float* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < s1; ++i) {
    float maxValue = std::numeric_limits<float>::lowest();
    const float* ra = a + i * s2;
    float* rb = b + i * s2;
    for (unsigned y = 0; y < s2; ++y) {
      maxValue = std::max(maxValue, ra[y]);
    }
    MKLVcl16f vMaxValue(maxValue);
    unsigned y;
    for (y = 0; y + (64 / sizeof(float)) < s2; y += (64 / sizeof(float))) {
      MKLVcl16f v;
      v.load(ra + y);
      v = exp(v - vMaxValue);
      v.store(rb + y);
    }

    if (y < s2) {
      float tmp[16];
      for (unsigned i = 0; y + i < s2; ++i) {
        tmp[i] = ra[y + i];
      }
      MKLVcl16f v;
      v.load(tmp);
      v = exp(v - vMaxValue);
      v.store(tmp);
      for (unsigned i = 0; y + i < s2; ++i) {
        rb[y + i] = tmp[i];
      }
    }

    MKLVcl16f vSumValue(0.f);
    for (y = 0; y + (64 / sizeof(float)) < s2; y += (64 / sizeof(float))) {
      MKLVcl16f tmp;
      tmp.load(rb + y);
      vSumValue += tmp;
    }
    float sumValue = 0.f;
    for (; y < s2; ++y) {
      sumValue += rb[y];
    }
    float vTmpSumValue[64 / sizeof(float)];
    vSumValue.store(vTmpSumValue);
    for (size_t i = 0; i < (64 / sizeof(float)); ++i) {
      sumValue += vTmpSumValue[i];
    }

    vSumValue = MKLVcl16f(sumValue);
    for (y = 0; y + (64 / sizeof(float)) < s2; y += (64 / sizeof(float))) {
      MKLVcl16f tmp;
      tmp.load(rb + y);
      tmp /= vSumValue;
      tmp.store(rb + y);
    }
    for (; y < s2; ++y) {
      rb[y] /= sumValue;
    }
  }
}

void _MKLActivationKernelAVX512<float>::sin(const float* a, unsigned total, float* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    vcl::Vec16f v;
    v.load(a + i);
    v = vcl::sin(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<float>::cos(const float* a, unsigned total, float* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    vcl::Vec16f v;
    v.load(a + i);
    v = vcl::cos(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<float>::pow(
    const float* a, unsigned total, float exponent, float* b) {
  vcl::Vec16f e(exponent);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    vcl::Vec16f v;
    v.load(a + i);
    v = vcl::pow(v, e);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<float>::gelu(const float* a, unsigned total, float* b) {
  MKLVcl16f vc1(0.7978845608028654);
  MKLVcl16f vc2(0.044715);
  MKLVcl16f one(1.0);
  MKLVcl16f half(0.5);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    MKLVcl16f v;
    v.load(a + i);
    MKLVcl16f tmp = mul_add(vc2, v * v * v, v) * vc1;
    MKLVcl16f result = v * half * (one + tfcc::tanh(tmp));
    result.store(b + i);
  }
}

void _MKLActivationKernelAVX512<float>::asin(const float* a, unsigned total, float* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    vcl::Vec16f v;
    v.load(a + i);
    v = vcl::asin(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<float>::asinh(const float* a, unsigned total, float* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    vcl::Vec16f v;
    v.load(a + i);
    v = vcl::asinh(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<float>::acos(const float* a, unsigned total, float* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    vcl::Vec16f v;
    v.load(a + i);
    v = vcl::acos(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<float>::acosh(const float* a, unsigned total, float* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    vcl::Vec16f v;
    v.load(a + i);
    v = vcl::acosh(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<float>::atan(const float* a, unsigned total, float* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    vcl::Vec16f v;
    v.load(a + i);
    v = vcl::atan(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<float>::atanh(const float* a, unsigned total, float* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    vcl::Vec16f v;
    v.load(a + i);
    v = vcl::atanh(v);
    v.store(b + i);
  }
}

#ifdef TFCC_USE_SVML
void _MKLActivationKernelAVX512<float>::erf(const float* a, unsigned total, float* b) {
#  pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    vcl::Vec16f v;
    v.load(a + i);
    v = vcl::erf(v);
    v.store(b + i);
  }
}
#endif

// double
void _MKLActivationKernelAVX512<double>::sigmoid(const double* a, unsigned total, double* b) {
  MKLVcl8d one(1.);
  MKLVcl8d zero(0.);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    MKLVcl8d v;
    v.load(a + i);
    v = one / (one + exp(zero - v));
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<double>::tanh(const double* a, unsigned total, double* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    vcl::Vec8d v;
    v.load(a + i);
    v = vcl::tanh(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<double>::relu(const double* a, unsigned total, double* b) {
  MKLVcl8d zero(0.);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    MKLVcl8d v;
    v.load(a + i);
    v = max(v, zero);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<double>::leakyRelu(
    const double* a, unsigned total, double alpha, double* b) {
  MKLVcl8d m(alpha);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    MKLVcl8d v;
    v.load(a + i);
    v = max(v, v * m);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<double>::log(const double* a, unsigned total, double* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    MKLVcl8d v;
    v.load(a + i);
    v = ::tfcc::log(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<double>::rsqrt(const double* a, unsigned total, double* b) {
  vcl::Vec8d one(1.0f);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    vcl::Vec8d v;
    v.load(a + i);
    v = one / vcl::sqrt(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<double>::softmax(
    const double* a, unsigned s1, unsigned s2, unsigned s3, double* b) {
  double maxValue = std::numeric_limits<double>::lowest();
  unsigned total = s1 * s2 * s3;
#pragma omp parallel for reduction(max : maxValue)
  for (unsigned i = 0; i < total; ++i) {
    maxValue = std::max(maxValue, a[i]);
  }
  MKLVcl8d vMaxValue(maxValue);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    MKLVcl8d v;
    v.load(a + i);
    v = exp(v - vMaxValue);
    v.store(b + i);
  }

  unsigned batch = s1 * s3;
#pragma omp parallel for
  for (unsigned i = 0; i < batch; ++i) {
    unsigned x = i / s3;
    unsigned z = i % s3;
    double sumValue = static_cast<double>(0);
    for (unsigned y = 0; y < s2; ++y) {
      sumValue += b[z + y * s3 + x * s3 * s2];
    }
    for (unsigned y = 0; y < s2; ++y) {
      b[z + y * s3 + x * s3 * s2] /= sumValue;
    }
  }
}

void _MKLActivationKernelAVX512<double>::softmaxV2(
    const double* a, unsigned s1, unsigned s2, double* b) {
  double maxValue = std::numeric_limits<double>::lowest();
  unsigned total = s1 * s2;
#pragma omp parallel for reduction(max : maxValue)
  for (unsigned i = 0; i < total; ++i) {
    maxValue = std::max(maxValue, a[i]);
  }
  MKLVcl8d vMaxValue(maxValue);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    MKLVcl8d v;
    v.load(a + i);
    v = exp(v - vMaxValue);
    v.store(b + i);
  }

#pragma omp parallel for
  for (unsigned i = 0; i < s1; ++i) {
    double sumValue = static_cast<double>(0);
    for (unsigned y = 0; y < s2; ++y) {
      sumValue += b[y + i * s2];
    }
    for (unsigned y = 0; y < s2; ++y) {
      b[y + i * s2] /= sumValue;
    }
  }

#pragma omp parallel for
  for (unsigned i = 0; i < s1; ++i) {
    MKLVcl8d vSumValue(0.f);
    unsigned y;
    for (y = 0; y + (64 / sizeof(double)) < s2; y += (64 / sizeof(double))) {
      MKLVcl8d tmp;
      tmp.load(b + i * s2 + y);
      vSumValue += tmp;
    }
    double sumValue = 0.f;
    for (; y < s2; ++y) {
      sumValue += b[i * s2 + y];
    }
    double vTmpSumValue[64 / sizeof(double)];
    vSumValue.store(vTmpSumValue);
    for (size_t i = 0; i < (64 / sizeof(double)); ++i) {
      sumValue += vTmpSumValue[i];
    }

    vSumValue = MKLVcl8d(sumValue);
    for (y = 0; y + (64 / sizeof(double)) < s2; y += (64 / sizeof(double))) {
      MKLVcl8d tmp;
      tmp.load(b + y + i * s2);
      tmp /= vSumValue;
      tmp.store(b + y + i * s2);
    }
    for (; y < s2; ++y) {
      b[y + i * s2] /= sumValue;
    }
  }
}

void _MKLActivationKernelAVX512<double>::sin(const double* a, unsigned total, double* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    vcl::Vec8d v;
    v.load(a + i);
    v = vcl::sin(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<double>::cos(const double* a, unsigned total, double* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    vcl::Vec8d v;
    v.load(a + i);
    v = vcl::cos(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<double>::pow(
    const double* a, unsigned total, double exponent, double* b) {
  vcl::Vec8d e(exponent);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    vcl::Vec8d v;
    v.load(a + i);
    v = vcl::pow(v, e);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<double>::gelu(const double* a, unsigned total, double* b) {
  MKLVcl8d vc1(0.7978845608028654);
  MKLVcl8d vc2(0.044715);
  MKLVcl8d one(1.0);
  MKLVcl8d half(0.5);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    MKLVcl8d v;
    v.load(a + i);
    MKLVcl8d tmp = mul_add(vc2, v * v * v, v) * vc1;
    MKLVcl8d result = v * half * (one + tfcc::tanh(tmp));
    result.store(b + i);
  }
}

void _MKLActivationKernelAVX512<double>::asin(const double* a, unsigned total, double* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    vcl::Vec8d v;
    v.load(a + i);
    v = vcl::asin(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<double>::asinh(const double* a, unsigned total, double* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    vcl::Vec8d v;
    v.load(a + i);
    v = vcl::asinh(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<double>::acos(const double* a, unsigned total, double* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    vcl::Vec8d v;
    v.load(a + i);
    v = vcl::acos(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<double>::acosh(const double* a, unsigned total, double* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    vcl::Vec8d v;
    v.load(a + i);
    v = vcl::acosh(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<double>::atan(const double* a, unsigned total, double* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    vcl::Vec8d v;
    v.load(a + i);
    v = vcl::atan(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX512<double>::atanh(const double* a, unsigned total, double* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    vcl::Vec8d v;
    v.load(a + i);
    v = vcl::atanh(v);
    v.store(b + i);
  }
}

#ifdef TFCC_USE_SVML
void _MKLActivationKernelAVX512<double>::erf(const double* a, unsigned total, double* b) {
#  pragma omp parallel for
  for (unsigned i = 0; i < total; i += (64 / sizeof(double))) {
    vcl::Vec8d v;
    v.load(a + i);
    v = vcl::erf(v);
    v.store(b + i);
  }
}
#endif

#define DEFINE_FUNC(type) template class _MKLActivationKernelAVX512<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
