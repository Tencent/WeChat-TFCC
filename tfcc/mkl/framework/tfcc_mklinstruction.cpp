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

#include "tfcc_mklinstruction.h"

#if defined(__GNUC__) || defined(__clang__)

#  include <cpuid.h>

namespace tfcc {

static inline void _get_cpu_id(
    uint32_t cmd, uint32_t& eax, uint32_t& ebx, uint32_t& ecx, uint32_t& edx) {
  __cpuid_count(cmd, 0, eax, ebx, ecx, edx);
}

}  // namespace tfcc

#elif defined(_MSC_VER)

#  include <immintrin.h>
#  include <intrin.h>

namespace tfcc {

static inline void _get_cpu_id(
    uint32_t cmd, uint32_t& eax, uint32_t& ebx, uint32_t& ecx, uint32_t& edx) {
  int data[4];
  __cpuid(data, cmd);
  eax = static_cast<uint32_t>(data[0]);
  ebx = static_cast<uint32_t>(data[1]);
  ecx = static_cast<uint32_t>(data[2]);
  edx = static_cast<uint32_t>(data[3]);
}

}  // namespace tfcc

#endif

namespace tfcc {

uint64_t get_cpu_instruction_set() {
  constexpr uint32_t MASK_AVX2 = 0x1u << 5;
  constexpr uint32_t MASK_AVX512F = 0x1u << 16;
  constexpr uint32_t MASK_AVX512DQ = 0x1u << 17;
  constexpr uint32_t MASK_AVX512CD = 0x1u << 28;
  constexpr uint32_t MASK_AVX512BW = 0x1u << 30;
  constexpr uint32_t MASK_AVX512VL = 0x1u << 31;
  constexpr uint32_t MASK_AVX512 =
      MASK_AVX512F | MASK_AVX512DQ | MASK_AVX512CD | MASK_AVX512BW | MASK_AVX512VL;
  constexpr uint32_t MASK_VNNI = 0x1u << 11;

  unsigned eax, ebx, ecx, edx;
  _get_cpu_id(7, eax, ebx, ecx, edx);
  uint64_t instruction = 0;
  if ((ebx & MASK_AVX2) == MASK_AVX2) {
    instruction |= MKLInstruction::AVX256;
  }
  if ((ebx & MASK_AVX512) == MASK_AVX512) {
    instruction |= MKLInstruction::AVX512;
  }
  if ((ecx & MASK_VNNI) == MASK_VNNI) {
    instruction |= MKLInstruction::VNNI;
  }
  return instruction;
}

}  // namespace tfcc
