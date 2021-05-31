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

#include "tfcc_mklactivationkernel.avx256.h"

#include <algorithm>

#define MAX_VECTOR_SIZE 256
#define VCL_NAMESPACE vcl
#include "vectorclass.h"
#ifdef TFCC_USE_SVML
#  include "vectormath_lib.h"
#else
#  include "vectormath_exp.h"
#  include "vectormath_hyp.h"
#  include "vectormath_trig.h"
#endif

#include "framework/tfcc_types.h"
#include "vcl/tfcc_mklvcl4d.hpp"
#include "vcl/tfcc_mklvcl8f.hpp"

namespace tfcc {

// float
void _MKLActivationKernelAVX256<float>::sigmoid(const float* a, unsigned total, float* b) {
  MKLVcl8f one(1.f);
  MKLVcl8f zero(0.f);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(float))) {
    MKLVcl8f v;
    v.load(a + i);
    v = one / (one + exp(zero - v));
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<float>::tanh(const float* a, unsigned total, float* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(float))) {
    vcl::Vec8f v;
    v.load(a + i);
    v = vcl::tanh(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<float>::relu(const float* a, unsigned total, float* b) {
  MKLVcl8f zero(0.f);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(float))) {
    MKLVcl8f v;
    v.load(a + i);
    v = max(v, zero);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<float>::leakyRelu(
    const float* a, unsigned total, float alpha, float* b) {
  MKLVcl8f m(alpha);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(float))) {
    MKLVcl8f v;
    v.load(a + i);
    v = max(v, v * m);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<float>::log(const float* a, unsigned total, float* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(float))) {
    MKLVcl8f v;
    v.load(a + i);
    v = ::tfcc::log(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<float>::rsqrt(const float* a, unsigned total, float* b) {
  vcl::Vec8f one(1.0f);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(float))) {
    vcl::Vec8f v;
    v.load(a + i);
    v = one / vcl::sqrt(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<float>::softmax(
    const float* a, unsigned s1, unsigned s2, unsigned s3, float* b) {
  float maxValue = std::numeric_limits<float>::lowest();
  unsigned total = s1 * s2 * s3;
#pragma omp parallel for reduction(max : maxValue)
  for (unsigned i = 0; i < total; ++i) {
    maxValue = std::max(maxValue, a[i]);
  }
  MKLVcl8f vMaxValue(maxValue);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(float))) {
    MKLVcl8f v;
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

void _MKLActivationKernelAVX256<float>::softmaxV2(
    const float* a, unsigned s1, unsigned s2, float* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < s1; ++i) {
    float maxValue = std::numeric_limits<float>::lowest();
    const float* ra = a + i * s2;
    float* rb = b + i * s2;
    for (unsigned y = 0; y < s2; ++y) {
      maxValue = std::max(maxValue, ra[y]);
    }
    MKLVcl8f vMaxValue(maxValue);
    unsigned y;
    for (y = 0; y + (32 / sizeof(float)) < s2; y += (32 / sizeof(float))) {
      MKLVcl8f v;
      v.load(ra + y);
      v = exp(v - vMaxValue);
      v.store(rb + y);
    }

    if (y < s2) {
      float tmp[8];
      for (unsigned i = 0; y + i < s2; ++i) {
        tmp[i] = ra[y + i];
      }
      MKLVcl8f v;
      v.load(tmp);
      v = exp(v - vMaxValue);
      v.store(tmp);
      for (unsigned i = 0; y + i < s2; ++i) {
        rb[y + i] = tmp[i];
      }
    }

    MKLVcl8f vSumValue(0.f);
    for (y = 0; y + (32 / sizeof(float)) < s2; y += (32 / sizeof(float))) {
      MKLVcl8f tmp;
      tmp.load(rb + y);
      vSumValue += tmp;
    }
    float sumValue = 0.f;
    for (; y < s2; ++y) {
      sumValue += rb[y];
    }
    float vTmpSumValue[32 / sizeof(float)];
    vSumValue.store(vTmpSumValue);
    for (size_t i = 0; i < (32 / sizeof(float)); ++i) {
      sumValue += vTmpSumValue[i];
    }

    vSumValue = MKLVcl8f(sumValue);
    for (y = 0; y + (32 / sizeof(float)) < s2; y += (32 / sizeof(float))) {
      MKLVcl8f tmp;
      tmp.load(rb + y);
      tmp /= vSumValue;
      tmp.store(rb + y);
    }
    for (; y < s2; ++y) {
      rb[y] /= sumValue;
    }
  }
}

void _MKLActivationKernelAVX256<float>::sin(const float* a, unsigned total, float* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(float))) {
    vcl::Vec8f v;
    v.load(a + i);
    v = vcl::sin(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<float>::cos(const float* a, unsigned total, float* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(float))) {
    vcl::Vec8f v;
    v.load(a + i);
    v = vcl::cos(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<float>::pow(
    const float* a, unsigned total, float exponent, float* b) {
  vcl::Vec8f e(exponent);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(float))) {
    vcl::Vec8f v;
    v.load(a + i);
    v = vcl::pow(v, e);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<float>::powV2(
    const float* a, const float* exponent, float* b, unsigned total) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(float))) {
    vcl::Vec8f v;
    v.load(a + i);
    vcl::Vec8f e;
    e.load(exponent + i);
    v = vcl::pow(v, e);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<float>::powV3(
    const float* exponent, unsigned total, float a, float* b) {
  vcl::Vec8f x(a);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(float))) {
    vcl::Vec8f e;
    e.load(exponent + i);
    vcl::Vec8f v = vcl::pow(x, e);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<float>::gelu(const float* a, unsigned total, float* b) {
  MKLVcl8f vc1(0.7978845608028654);
  MKLVcl8f vc2(0.044715);
  MKLVcl8f one(1.0);
  MKLVcl8f half(0.5);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(float))) {
    MKLVcl8f v;
    v.load(a + i);
    MKLVcl8f tmp = mul_add(vc2, v * v * v, v) * vc1;
    MKLVcl8f result = v * half * (one + tfcc::tanh(tmp));
    result.store(b + i);
  }
}

void _MKLActivationKernelAVX256<float>::asin(const float* a, unsigned total, float* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(float))) {
    vcl::Vec8f v;
    v.load(a + i);
    v = vcl::asin(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<float>::asinh(const float* a, unsigned total, float* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(float))) {
    vcl::Vec8f v;
    v.load(a + i);
    v = vcl::asinh(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<float>::acos(const float* a, unsigned total, float* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(float))) {
    vcl::Vec8f v;
    v.load(a + i);
    v = vcl::acos(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<float>::acosh(const float* a, unsigned total, float* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(float))) {
    vcl::Vec8f v;
    v.load(a + i);
    v = vcl::acosh(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<float>::atan(const float* a, unsigned total, float* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(float))) {
    vcl::Vec8f v;
    v.load(a + i);
    v = vcl::atan(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<float>::atanh(const float* a, unsigned total, float* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(float))) {
    vcl::Vec8f v;
    v.load(a + i);
    v = vcl::atanh(v);
    v.store(b + i);
  }
}

#ifdef TFCC_USE_SVML
void _MKLActivationKernelAVX256<float>::erf(const float* a, unsigned total, float* b) {
#  pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(float))) {
    vcl::Vec8f v;
    v.load(a + i);
    v = vcl::erf(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<float>::geluAccurate(const float* a, unsigned total, float* b) {
  vcl::Vec8f c(1.4142135623730951f);
#  pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(float))) {
    vcl::Vec8f v;
    v.load(a + i);
    v = 0.5 * v * (1 + vcl::erf(v / c));
    v.store(b + i);
  }
}
#endif

// double
void _MKLActivationKernelAVX256<double>::sigmoid(const double* a, unsigned total, double* b) {
  MKLVcl4d one(1.);
  MKLVcl4d zero(0.);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(double))) {
    MKLVcl4d v;
    v.load(a + i);
    v = one / (one + exp(zero - v));
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<double>::tanh(const double* a, unsigned total, double* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(double))) {
    vcl::Vec4d v;
    v.load(a + i);
    v = vcl::tanh(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<double>::relu(const double* a, unsigned total, double* b) {
  MKLVcl4d zero(0.);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(double))) {
    MKLVcl4d v;
    v.load(a + i);
    v = max(v, zero);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<double>::leakyRelu(
    const double* a, unsigned total, double alpha, double* b) {
  MKLVcl4d m(alpha);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(double))) {
    MKLVcl4d v;
    v.load(a + i);
    v = max(v, v * m);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<double>::log(const double* a, unsigned total, double* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(double))) {
    MKLVcl4d v;
    v.load(a + i);
    v = ::tfcc::log(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<double>::rsqrt(const double* a, unsigned total, double* b) {
  vcl::Vec4d one(1.0);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(double))) {
    vcl::Vec4d v;
    v.load(a + i);
    v = one / vcl::sqrt(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<double>::softmax(
    const double* a, unsigned s1, unsigned s2, unsigned s3, double* b) {
  double maxValue = std::numeric_limits<double>::lowest();
  unsigned total = s1 * s2 * s3;
#pragma omp parallel for reduction(max : maxValue)
  for (unsigned i = 0; i < total; ++i) {
    maxValue = std::max(maxValue, a[i]);
  }
  MKLVcl4d vMaxValue(maxValue);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(double))) {
    MKLVcl4d v;
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

void _MKLActivationKernelAVX256<double>::softmaxV2(
    const double* a, unsigned s1, unsigned s2, double* b) {
  double maxValue = std::numeric_limits<double>::lowest();
  unsigned total = s1 * s2;
#pragma omp parallel for reduction(max : maxValue)
  for (unsigned i = 0; i < total; ++i) {
    maxValue = std::max(maxValue, a[i]);
  }
  MKLVcl4d vMaxValue(maxValue);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(double))) {
    MKLVcl4d v;
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
    MKLVcl4d vSumValue(0.);
    unsigned y;
    for (y = 0; y + (32 / sizeof(double)) < s2; y += (32 / sizeof(double))) {
      MKLVcl4d tmp;
      tmp.load(b + i * s2 + y);
      vSumValue += tmp;
    }
    double sumValue = 0.;
    for (; y < s2; ++y) {
      sumValue += b[i * s2 + y];
    }
    double vTmpSumValue[32 / sizeof(double)];
    vSumValue.store(vTmpSumValue);
    for (size_t i = 0; i < (32 / sizeof(double)); ++i) {
      sumValue += vTmpSumValue[i];
    }

    vSumValue = MKLVcl4d(sumValue);
    for (y = 0; y + (32 / sizeof(double)) < s2; y += (32 / sizeof(double))) {
      MKLVcl4d tmp;
      tmp.load(b + y + i * s2);
      tmp /= vSumValue;
      tmp.store(b + y + i * s2);
    }
    for (; y < s2; ++y) {
      b[y + i * s2] /= sumValue;
    }
  }
}

void _MKLActivationKernelAVX256<double>::sin(const double* a, unsigned total, double* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(double))) {
    vcl::Vec4d v;
    v.load(a + i);
    v = vcl::sin(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<double>::cos(const double* a, unsigned total, double* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(double))) {
    vcl::Vec4d v;
    v.load(a + i);
    v = vcl::cos(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<double>::pow(
    const double* a, unsigned total, double exponent, double* b) {
  vcl::Vec4d e(exponent);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(double))) {
    vcl::Vec4d v;
    v.load(a + i);
    v = vcl::pow(v, e);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<double>::powV2(
    const double* a, const double* exponent, double* b, unsigned total) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(double))) {
    vcl::Vec4d v;
    v.load(a + i);
    vcl::Vec4d e;
    e.load(exponent + i);
    v = vcl::pow(v, e);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<double>::powV3(
    const double* exponent, unsigned total, double a, double* b) {
  vcl::Vec4d x(a);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(double))) {
    vcl::Vec4d e;
    e.load(exponent + i);
    vcl::Vec4d v = vcl::pow(v, e);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<double>::gelu(const double* a, unsigned total, double* b) {
  MKLVcl4d vc1(0.7978845608028654);
  MKLVcl4d vc2(0.044715);
  MKLVcl4d one(1.0);
  MKLVcl4d half(0.5);
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(double))) {
    MKLVcl4d v;
    v.load(a + i);
    MKLVcl4d tmp = mul_add(vc2, v * v * v, v) * vc1;
    MKLVcl4d result = v * half * (one + tfcc::tanh(tmp));
    result.store(b + i);
  }
}

void _MKLActivationKernelAVX256<double>::asin(const double* a, unsigned total, double* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(double))) {
    vcl::Vec4d v;
    v.load(a + i);
    v = vcl::asin(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<double>::asinh(const double* a, unsigned total, double* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(double))) {
    vcl::Vec4d v;
    v.load(a + i);
    v = vcl::asinh(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<double>::acos(const double* a, unsigned total, double* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(double))) {
    vcl::Vec4d v;
    v.load(a + i);
    v = vcl::acos(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<double>::acosh(const double* a, unsigned total, double* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(double))) {
    vcl::Vec4d v;
    v.load(a + i);
    v = vcl::acosh(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<double>::atan(const double* a, unsigned total, double* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(double))) {
    vcl::Vec4d v;
    v.load(a + i);
    v = vcl::atan(v);
    v.store(b + i);
  }
}

void _MKLActivationKernelAVX256<double>::atanh(const double* a, unsigned total, double* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(double))) {
    vcl::Vec4d v;
    v.load(a + i);
    v = vcl::atanh(v);
    v.store(b + i);
  }
}

#ifdef TFCC_USE_SVML
void _MKLActivationKernelAVX256<double>::erf(const double* a, unsigned total, double* b) {
#  pragma omp parallel for
  for (unsigned i = 0; i < total; i += (32 / sizeof(double))) {
    vcl::Vec4d v;
    v.load(a + i);
    v = vcl::erf(v);
    v.store(b + i);
  }
}
#endif

#define DEFINE_FUNC(type) template class _MKLActivationKernelAVX256<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
