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

#include "tfcc_mklquantizationkernel.avx256.h"

#include <limits>

#include "framework/tfcc_types.h"
#include "vcl/tfcc_mklvcl256.hpp"
#include "vcl/tfcc_mklvcl4d.hpp"
#include "vcl/tfcc_mklvcl4q.hpp"
#include "vcl/tfcc_mklvcl8f.hpp"

namespace tfcc {

template <class T>
static inline void _mkl_avx256_save_vq(
    MKLVcl4q vq1, MKLVcl4q vq2, MKLVcl4q vq3, MKLVcl4q vq4, T* result) {
  int64_t tmp[4];
  vq1.store(tmp);
  result[0] = static_cast<T>(tmp[0]);
  result[1] = static_cast<T>(tmp[1]);
  result[2] = static_cast<T>(tmp[2]);
  result[3] = static_cast<T>(tmp[3]);

  vq2.store(tmp);
  result[4] = static_cast<T>(tmp[0]);
  result[5] = static_cast<T>(tmp[1]);
  result[6] = static_cast<T>(tmp[2]);
  result[7] = static_cast<T>(tmp[3]);

  vq3.store(tmp);
  result[8] = static_cast<T>(tmp[0]);
  result[9] = static_cast<T>(tmp[1]);
  result[10] = static_cast<T>(tmp[2]);
  result[11] = static_cast<T>(tmp[3]);

  vq4.store(tmp);
  result[12] = static_cast<T>(tmp[0]);
  result[13] = static_cast<T>(tmp[1]);
  result[14] = static_cast<T>(tmp[2]);
  result[15] = static_cast<T>(tmp[3]);
}

template <class T>
void _MKLQuantizationKernelAVX256<T>::quantize(
    const float* a, unsigned total, double scale, int64_t offset, T* result) {
  MKLVcl4d vScale(scale);
  MKLVcl4q vQMin(static_cast<int64_t>(std::numeric_limits<T>::lowest()));
  MKLVcl4q vQMax(static_cast<int64_t>(std::numeric_limits<T>::max()));
  MKLVcl4q vOffset(offset);
#pragma omp parallel for firstprivate(vScale, vOffset, vQMin, vQMax)
  for (unsigned i = 0; i < total; i += (64 / sizeof(float))) {
    MKLVcl8f va;
    va.load(a + i);
    MKLVcl4d va1 = extend_low(va);
    MKLVcl4d va2 = extend_high(va);

    MKLVcl4q vq1 = round_to_int64(va1 * vScale) - vOffset;
    vq1 += vQMin;
    vq1 = max(vq1, vQMin);
    vq1 = min(vq1, vQMax);

    MKLVcl4q vq2 = round_to_int64(va2 * vScale) - vOffset;
    vq2 += vQMin;
    vq2 = max(vq2, vQMin);
    vq2 = min(vq2, vQMax);

    va.load(a + i + 8);
    va1 = extend_low(va);
    va2 = extend_high(va);

    MKLVcl4q vq3 = round_to_int64(va1 * vScale) - vOffset;
    vq3 += vQMin;
    vq3 = max(vq3, vQMin);
    vq3 = min(vq3, vQMax);

    MKLVcl4q vq4 = round_to_int64(va2 * vScale) - vOffset;
    vq4 += vQMin;
    vq4 = max(vq4, vQMin);
    vq4 = min(vq4, vQMax);

    _mkl_avx256_save_vq(vq1, vq2, vq3, vq4, result + i);
  }
}

template <class T>
void _MKLQuantizationKernelAVX256<T>::dequantize(
    const T* a, unsigned total, double scale, double minRounded, float* result) {
  MKLVcl4d vDMin(static_cast<double>(std::numeric_limits<T>::lowest()));
  MKLVcl4d vScale(scale);
  MKLVcl4d vMinRounded(minRounded);
#pragma omp parallel for firstprivate(vDMin, vScale, vMinRounded)
  for (unsigned i = 0; i < total; i += (32 / sizeof(float))) {
    MKLVcl4d va1(
        static_cast<double>(a[i]), static_cast<double>(a[i + 1]), static_cast<double>(a[i + 2]),
        static_cast<double>(a[i + 3]));
    MKLVcl4d vr1 = (va1 - vDMin) * vScale + vMinRounded;

    MKLVcl4d va2(
        static_cast<double>(a[i + 4]), static_cast<double>(a[i + 5]), static_cast<double>(a[i + 6]),
        static_cast<double>(a[i + 7]));
    MKLVcl4d vr2 = (va2 - vDMin) * vScale + vMinRounded;
    MKLVcl8f vf = compress(vr1, vr2);
    vf.store(result + i);
  }
}

template <class TI, class TO>
void _MKLRequantizationKernelAVX256<TI, TO>::requantize(
    const TI* a, unsigned total, double inputScale, double inputMinRounded, double outputScale,
    int64_t outputOffset, TO* result) {
  MKLVcl4q vQMin(static_cast<int64_t>(std::numeric_limits<TO>::lowest()));
  MKLVcl4q vQMax(static_cast<int64_t>(std::numeric_limits<TO>::max()));
#pragma omp parallel for firstprivate(vQMin, vQMax)
  for (unsigned i = 0; i < total; i += (32 / sizeof(double))) {
    MKLVcl4d vOffset(
        static_cast<double>(a[i]), static_cast<double>(a[i + 1]), static_cast<double>(a[i + 2]),
        static_cast<double>(a[i + 3]));
    vOffset -= MKLVcl4d(static_cast<double>(std::numeric_limits<TI>::lowest()));
    MKLVcl4d vReal = MKLVcl4d(inputMinRounded) + (vOffset * MKLVcl4d(inputScale));

    MKLVcl4q vQuantized = round_to_int64(vReal * MKLVcl4d(outputScale)) - MKLVcl4q(outputOffset);
    vQuantized += MKLVcl4q(static_cast<int64_t>(std::numeric_limits<TO>::lowest()));
    vQuantized = max(vQuantized, vQMin);
    vQuantized = min(vQuantized, vQMax);
    int64_t vTmpQuantized[4];
    vQuantized.store(vTmpQuantized);
    result[i + 0] = static_cast<TO>(vTmpQuantized[0]);
    result[i + 1] = static_cast<TO>(vTmpQuantized[1]);
    result[i + 2] = static_cast<TO>(vTmpQuantized[2]);
    result[i + 3] = static_cast<TO>(vTmpQuantized[3]);
  }
}

#define DEFINE_FUNC_QUANTIZATION(type) template class _MKLQuantizationKernelAVX256<type>;

TFCC_FOR_QUANTIZATION_TYPES(DEFINE_FUNC_QUANTIZATION);

#define DEFINE_FUNC_REQUANTIZATION(type)                         \
  template class _MKLRequantizationKernelAVX256<type, int8_t>;   \
  template class _MKLRequantizationKernelAVX256<type, uint8_t>;  \
  template class _MKLRequantizationKernelAVX256<type, int16_t>;  \
  template class _MKLRequantizationKernelAVX256<type, uint16_t>; \
  template class _MKLRequantizationKernelAVX256<type, int32_t>;  \
  template class _MKLRequantizationKernelAVX256<type, uint32_t>;

TFCC_FOR_QUANTIZATION_TYPES(DEFINE_FUNC_REQUANTIZATION);

}  // namespace tfcc
