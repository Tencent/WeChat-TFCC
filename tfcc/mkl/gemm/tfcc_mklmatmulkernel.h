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

#include <immintrin.h>
#include <cstdint>

namespace tfcc {

void matmul_m24_n4_k2(
    const uint8_t* a, const uint8_t* b, int32_t* dst, uint64_t stride, uint64_t batch) {
  asm volatile(
      "vpxor ymm4, ymm4, ymm4\n\t"
      "vpxor ymm5, ymm5, ymm5\n\t"
      "vpxor ymm6, ymm6, ymm6\n\t"
      "vpxor ymm7, ymm7, ymm7\n\t"
      "vpxor ymm8, ymm8, ymm8\n\t"
      "vpxor ymm9, ymm9, ymm9\n\t"
      "vpxor ymm10, ymm10, ymm10\n\t"
      "vpxor ymm11, ymm11, ymm11\n\t"
      "vpxor ymm12, ymm12, ymm12\n\t"
      "vpxor ymm13, ymm13, ymm13\n\t"
      "vpxor ymm14, ymm14, ymm14\n\t"
      "vpxor ymm15, ymm15, ymm15\n\t"
      "lea r14, [%[dst_ptr] + %[stride] * 2]\n\t"

      // loop
      "_matmul_m24_n4_k2_loop_start_ua_ub:\n\t"

      "vpmovzxbw ymm1, [%[b_ptr]]\n\t"
      "vpermq ymm1, ymm1, 0x44\n\t"
      // a_ptr section 0
      "vpmovzxbw ymm0, [%[a_ptr]]\n\t"
      "vpshufd ymm2, ymm1, 0x00\n\t"
      "vpshufd ymm3, ymm1, 0x55\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm4, ymm2, ymm4\n\t"
      "vpaddd ymm5, ymm3, ymm5\n\t"
      "vpshufd ymm2, ymm1, 0xaa\n\t"
      "vpshufd ymm3, ymm1, 0xff\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm6, ymm2, ymm6\n\t"
      "vpaddd ymm7, ymm3, ymm7\n\t"

      // a_ptr section 1
      "vpmovzxbw ymm0, [%[a_ptr] + 0x10]\n\t"
      "vpshufd ymm2, ymm1, 0x00\n\t"
      "vpshufd ymm3, ymm1, 0x55\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm8, ymm2, ymm8\n\t"
      "vpaddd ymm9, ymm3, ymm9\n\t"
      "vpshufd ymm2, ymm1, 0xaa\n\t"
      "vpshufd ymm3, ymm1, 0xff\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm10, ymm2, ymm10\n\t"
      "vpaddd ymm11, ymm3, ymm11\n\t"

      // a_ptr section 2
      "vpmovzxbw ymm0, [%[a_ptr] + 0x20]\n\t"
      "vpshufd ymm2, ymm1, 0x00\n\t"
      "vpshufd ymm3, ymm1, 0x55\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm12, ymm2, ymm12\n\t"
      "vpaddd ymm13, ymm3, ymm13\n\t"
      "vpshufd ymm2, ymm1, 0xaa\n\t"
      "vpshufd ymm3, ymm1, 0xff\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm14, ymm2, ymm14\n\t"
      "vpaddd ymm15, ymm3, ymm15\n\t"

      "add %[a_ptr], 0x30 \n\t"
      "add %[b_ptr], 0x08 \n\t"

      "dec %[batch]\n\t"
      "jnz _matmul_m24_n4_k2_loop_start_ua_ub\n\t"

      // add dst
      "vpaddd ymm4, ymm4, [%[dst_ptr]]\n\t"
      "vpaddd ymm8, ymm8, [%[dst_ptr] + 0x20]\n\t"
      "vpaddd ymm12, ymm12, [%[dst_ptr] + 0x40]\n\t"

      "vpaddd ymm5, ymm5, [%[dst_ptr] + %[stride] * 1]\n\t"
      "vpaddd ymm9, ymm9, [%[dst_ptr] + %[stride] * 1 + 0x20]\n\t"
      "vpaddd ymm13, ymm13, [%[dst_ptr] + %[stride] * 1 + 0x40]\n\t"

      "vpaddd ymm6, ymm6, [%[dst_ptr] + %[stride] * 2]\n\t"
      "vpaddd ymm10, ymm10, [%[dst_ptr] + %[stride] * 2 + 0x20]\n\t"
      "vpaddd ymm14, ymm14, [%[dst_ptr] + %[stride] * 2 + 0x40]\n\t"

      "vpaddd ymm7, ymm7, [r14 + %[stride] * 1]\n\t"
      "vpaddd ymm11, ymm11, [r14 + %[stride] * 1 + 0x20]\n\t"
      "vpaddd ymm15, ymm15, [r14 + %[stride] * 1 + 0x40]\n\t"

      // save to dst
      "vmovdqu [%[dst_ptr]], ymm4\n\t"
      "vmovdqu [%[dst_ptr] + 0x20], ymm8\n\t"
      "vmovdqu [%[dst_ptr] + 0x40], ymm12\n\t"

      "vmovdqu [%[dst_ptr] + %[stride] * 1], ymm5\n\t"
      "vmovdqu [%[dst_ptr] + %[stride] * 1 + 0x20], ymm9\n\t"
      "vmovdqu [%[dst_ptr] + %[stride] * 1 + 0x40], ymm13\n\t"

      "vmovdqu [%[dst_ptr] + %[stride] * 2], ymm6\n\t"
      "vmovdqu [%[dst_ptr] + %[stride] * 2 + 0x20], ymm10\n\t"
      "vmovdqu [%[dst_ptr] + %[stride] * 2 + 0x40], ymm14\n\t"

      "vmovdqu [r14 + %[stride] * 1], ymm7\n\t"
      "vmovdqu [r14 + %[stride] * 1 + 0x20], ymm11\n\t"
      "vmovdqu [r14 + %[stride] * 1 + 0x40], ymm15\n\t"
      :
      : [ a_ptr ] "r"(a), [ b_ptr ] "r"(b), [ dst_ptr ] "r"(dst),
        [ stride ] "r"(stride * sizeof(int32_t)), [ batch ] "r"(batch)
      : "memory", "cc", "r14",
#if __GNUC__ >= 5
        "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
        "ymm11", "ymm12", "ymm13", "ymm14", "ymm15"
#else
        "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10",
        "xmm11", "xmm12", "xmm13", "xmm14", "xmm15"
#endif
  );
}

void matmul_m24_n4_k2(
    const int8_t* a, const uint8_t* b, int32_t* dst, uint64_t stride, uint64_t batch) {
  asm volatile(
      "vpxor ymm4, ymm4, ymm4\n\t"
      "vpxor ymm5, ymm5, ymm5\n\t"
      "vpxor ymm6, ymm6, ymm6\n\t"
      "vpxor ymm7, ymm7, ymm7\n\t"
      "vpxor ymm8, ymm8, ymm8\n\t"
      "vpxor ymm9, ymm9, ymm9\n\t"
      "vpxor ymm10, ymm10, ymm10\n\t"
      "vpxor ymm11, ymm11, ymm11\n\t"
      "vpxor ymm12, ymm12, ymm12\n\t"
      "vpxor ymm13, ymm13, ymm13\n\t"
      "vpxor ymm14, ymm14, ymm14\n\t"
      "vpxor ymm15, ymm15, ymm15\n\t"
      "lea r14, [%[dst_ptr] + %[stride] * 2]\n\t"

      // loop
      "_matmul_m24_n4_k2_loop_start_ia_ub:\n\t"

      "vpmovzxbw ymm1, [%[b_ptr]]\n\t"
      "vpermq ymm1, ymm1, 0x44\n\t"
      // a_ptr section 0
      "vpmovsxbw ymm0, [%[a_ptr]]\n\t"
      "vpshufd ymm2, ymm1, 0x00\n\t"
      "vpshufd ymm3, ymm1, 0x55\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm4, ymm2, ymm4\n\t"
      "vpaddd ymm5, ymm3, ymm5\n\t"
      "vpshufd ymm2, ymm1, 0xaa\n\t"
      "vpshufd ymm3, ymm1, 0xff\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm6, ymm2, ymm6\n\t"
      "vpaddd ymm7, ymm3, ymm7\n\t"

      // a_ptr section 1
      "vpmovsxbw ymm0, [%[a_ptr] + 0x10]\n\t"
      "vpshufd ymm2, ymm1, 0x00\n\t"
      "vpshufd ymm3, ymm1, 0x55\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm8, ymm2, ymm8\n\t"
      "vpaddd ymm9, ymm3, ymm9\n\t"
      "vpshufd ymm2, ymm1, 0xaa\n\t"
      "vpshufd ymm3, ymm1, 0xff\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm10, ymm2, ymm10\n\t"
      "vpaddd ymm11, ymm3, ymm11\n\t"

      // a_ptr section 2
      "vpmovsxbw ymm0, [%[a_ptr] + 0x20]\n\t"
      "vpshufd ymm2, ymm1, 0x00\n\t"
      "vpshufd ymm3, ymm1, 0x55\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm12, ymm2, ymm12\n\t"
      "vpaddd ymm13, ymm3, ymm13\n\t"
      "vpshufd ymm2, ymm1, 0xaa\n\t"
      "vpshufd ymm3, ymm1, 0xff\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm14, ymm2, ymm14\n\t"
      "vpaddd ymm15, ymm3, ymm15\n\t"

      "add %[a_ptr], 0x30 \n\t"
      "add %[b_ptr], 0x08 \n\t"

      "dec %[batch]\n\t"
      "jnz _matmul_m24_n4_k2_loop_start_ia_ub\n\t"

      // add dst
      "vpaddd ymm4, ymm4, [%[dst_ptr]]\n\t"
      "vpaddd ymm8, ymm8, [%[dst_ptr] + 0x20]\n\t"
      "vpaddd ymm12, ymm12, [%[dst_ptr] + 0x40]\n\t"

      "vpaddd ymm5, ymm5, [%[dst_ptr] + %[stride] * 1]\n\t"
      "vpaddd ymm9, ymm9, [%[dst_ptr] + %[stride] * 1 + 0x20]\n\t"
      "vpaddd ymm13, ymm13, [%[dst_ptr] + %[stride] * 1 + 0x40]\n\t"

      "vpaddd ymm6, ymm6, [%[dst_ptr] + %[stride] * 2]\n\t"
      "vpaddd ymm10, ymm10, [%[dst_ptr] + %[stride] * 2 + 0x20]\n\t"
      "vpaddd ymm14, ymm14, [%[dst_ptr] + %[stride] * 2 + 0x40]\n\t"

      "vpaddd ymm7, ymm7, [r14 + %[stride] * 1]\n\t"
      "vpaddd ymm11, ymm11, [r14 + %[stride] * 1 + 0x20]\n\t"
      "vpaddd ymm15, ymm15, [r14 + %[stride] * 1 + 0x40]\n\t"

      // save to dst
      "vmovdqu [%[dst_ptr]], ymm4\n\t"
      "vmovdqu [%[dst_ptr] + 0x20], ymm8\n\t"
      "vmovdqu [%[dst_ptr] + 0x40], ymm12\n\t"

      "vmovdqu [%[dst_ptr] + %[stride] * 1], ymm5\n\t"
      "vmovdqu [%[dst_ptr] + %[stride] * 1 + 0x20], ymm9\n\t"
      "vmovdqu [%[dst_ptr] + %[stride] * 1 + 0x40], ymm13\n\t"

      "vmovdqu [%[dst_ptr] + %[stride] * 2], ymm6\n\t"
      "vmovdqu [%[dst_ptr] + %[stride] * 2 + 0x20], ymm10\n\t"
      "vmovdqu [%[dst_ptr] + %[stride] * 2 + 0x40], ymm14\n\t"

      "vmovdqu [r14 + %[stride] * 1], ymm7\n\t"
      "vmovdqu [r14 + %[stride] * 1 + 0x20], ymm11\n\t"
      "vmovdqu [r14 + %[stride] * 1 + 0x40], ymm15\n\t"
      :
      : [ a_ptr ] "r"(a), [ b_ptr ] "r"(b), [ dst_ptr ] "r"(dst),
        [ stride ] "r"(stride * sizeof(int32_t)), [ batch ] "r"(batch)
      : "memory", "cc", "r14",
#if __GNUC__ >= 5
        "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
        "ymm11", "ymm12", "ymm13", "ymm14", "ymm15"
#else
        "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10",
        "xmm11", "xmm12", "xmm13", "xmm14", "xmm15"
#endif
  );
}

void matmul_m24_n4_k2(
    const uint8_t* a, const int8_t* b, int32_t* dst, uint64_t stride, uint64_t batch) {
  asm volatile(
      "vpxor ymm4, ymm4, ymm4\n\t"
      "vpxor ymm5, ymm5, ymm5\n\t"
      "vpxor ymm6, ymm6, ymm6\n\t"
      "vpxor ymm7, ymm7, ymm7\n\t"
      "vpxor ymm8, ymm8, ymm8\n\t"
      "vpxor ymm9, ymm9, ymm9\n\t"
      "vpxor ymm10, ymm10, ymm10\n\t"
      "vpxor ymm11, ymm11, ymm11\n\t"
      "vpxor ymm12, ymm12, ymm12\n\t"
      "vpxor ymm13, ymm13, ymm13\n\t"
      "vpxor ymm14, ymm14, ymm14\n\t"
      "vpxor ymm15, ymm15, ymm15\n\t"
      "lea r14, [%[dst_ptr] + %[stride] * 2]\n\t"

      // loop
      "_matmul_m24_n4_k2_loop_start_ua_ib:\n\t"

      "vpmovsxbw ymm1, [%[b_ptr]]\n\t"
      "vpermq ymm1, ymm1, 0x44\n\t"
      // a_ptr section 0
      "vpmovzxbw ymm0, [%[a_ptr]]\n\t"
      "vpshufd ymm2, ymm1, 0x00\n\t"
      "vpshufd ymm3, ymm1, 0x55\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm4, ymm2, ymm4\n\t"
      "vpaddd ymm5, ymm3, ymm5\n\t"
      "vpshufd ymm2, ymm1, 0xaa\n\t"
      "vpshufd ymm3, ymm1, 0xff\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm6, ymm2, ymm6\n\t"
      "vpaddd ymm7, ymm3, ymm7\n\t"

      // a_ptr section 1
      "vpmovzxbw ymm0, [%[a_ptr] + 0x10]\n\t"
      "vpshufd ymm2, ymm1, 0x00\n\t"
      "vpshufd ymm3, ymm1, 0x55\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm8, ymm2, ymm8\n\t"
      "vpaddd ymm9, ymm3, ymm9\n\t"
      "vpshufd ymm2, ymm1, 0xaa\n\t"
      "vpshufd ymm3, ymm1, 0xff\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm10, ymm2, ymm10\n\t"
      "vpaddd ymm11, ymm3, ymm11\n\t"

      // a_ptr section 2
      "vpmovzxbw ymm0, [%[a_ptr] + 0x20]\n\t"
      "vpshufd ymm2, ymm1, 0x00\n\t"
      "vpshufd ymm3, ymm1, 0x55\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm12, ymm2, ymm12\n\t"
      "vpaddd ymm13, ymm3, ymm13\n\t"
      "vpshufd ymm2, ymm1, 0xaa\n\t"
      "vpshufd ymm3, ymm1, 0xff\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm14, ymm2, ymm14\n\t"
      "vpaddd ymm15, ymm3, ymm15\n\t"

      "add %[a_ptr], 0x30 \n\t"
      "add %[b_ptr], 0x08 \n\t"

      "dec %[batch]\n\t"
      "jnz _matmul_m24_n4_k2_loop_start_ua_ib\n\t"

      // add dst
      "vpaddd ymm4, ymm4, [%[dst_ptr]]\n\t"
      "vpaddd ymm8, ymm8, [%[dst_ptr] + 0x20]\n\t"
      "vpaddd ymm12, ymm12, [%[dst_ptr] + 0x40]\n\t"

      "vpaddd ymm5, ymm5, [%[dst_ptr] + %[stride] * 1]\n\t"
      "vpaddd ymm9, ymm9, [%[dst_ptr] + %[stride] * 1 + 0x20]\n\t"
      "vpaddd ymm13, ymm13, [%[dst_ptr] + %[stride] * 1 + 0x40]\n\t"

      "vpaddd ymm6, ymm6, [%[dst_ptr] + %[stride] * 2]\n\t"
      "vpaddd ymm10, ymm10, [%[dst_ptr] + %[stride] * 2 + 0x20]\n\t"
      "vpaddd ymm14, ymm14, [%[dst_ptr] + %[stride] * 2 + 0x40]\n\t"

      "vpaddd ymm7, ymm7, [r14 + %[stride] * 1]\n\t"
      "vpaddd ymm11, ymm11, [r14 + %[stride] * 1 + 0x20]\n\t"
      "vpaddd ymm15, ymm15, [r14 + %[stride] * 1 + 0x40]\n\t"

      // save to dst
      "vmovdqu [%[dst_ptr]], ymm4\n\t"
      "vmovdqu [%[dst_ptr] + 0x20], ymm8\n\t"
      "vmovdqu [%[dst_ptr] + 0x40], ymm12\n\t"

      "vmovdqu [%[dst_ptr] + %[stride] * 1], ymm5\n\t"
      "vmovdqu [%[dst_ptr] + %[stride] * 1 + 0x20], ymm9\n\t"
      "vmovdqu [%[dst_ptr] + %[stride] * 1 + 0x40], ymm13\n\t"

      "vmovdqu [%[dst_ptr] + %[stride] * 2], ymm6\n\t"
      "vmovdqu [%[dst_ptr] + %[stride] * 2 + 0x20], ymm10\n\t"
      "vmovdqu [%[dst_ptr] + %[stride] * 2 + 0x40], ymm14\n\t"

      "vmovdqu [r14 + %[stride] * 1], ymm7\n\t"
      "vmovdqu [r14 + %[stride] * 1 + 0x20], ymm11\n\t"
      "vmovdqu [r14 + %[stride] * 1 + 0x40], ymm15\n\t"
      :
      : [ a_ptr ] "r"(a), [ b_ptr ] "r"(b), [ dst_ptr ] "r"(dst),
        [ stride ] "r"(stride * sizeof(int32_t)), [ batch ] "r"(batch)
      : "memory", "cc", "r14",
#if __GNUC__ >= 5
        "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
        "ymm11", "ymm12", "ymm13", "ymm14", "ymm15"
#else
        "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10",
        "xmm11", "xmm12", "xmm13", "xmm14", "xmm15"
#endif
  );
}

void matmul_m24_n4_k2(
    const int8_t* a, const int8_t* b, int32_t* dst, uint64_t stride, uint64_t batch) {
  asm volatile(
      "vpxor ymm4, ymm4, ymm4\n\t"
      "vpxor ymm5, ymm5, ymm5\n\t"
      "vpxor ymm6, ymm6, ymm6\n\t"
      "vpxor ymm7, ymm7, ymm7\n\t"
      "vpxor ymm8, ymm8, ymm8\n\t"
      "vpxor ymm9, ymm9, ymm9\n\t"
      "vpxor ymm10, ymm10, ymm10\n\t"
      "vpxor ymm11, ymm11, ymm11\n\t"
      "vpxor ymm12, ymm12, ymm12\n\t"
      "vpxor ymm13, ymm13, ymm13\n\t"
      "vpxor ymm14, ymm14, ymm14\n\t"
      "vpxor ymm15, ymm15, ymm15\n\t"
      "lea r14, [%[dst_ptr] + %[stride] * 2]\n\t"

      // loop
      "_matmul_m24_n4_k2_loop_start_ia_ib:\n\t"

      "vpmovsxbw ymm1, [%[b_ptr]]\n\t"
      "vpermq ymm1, ymm1, 0x44\n\t"
      // a_ptr section 0
      "vpmovsxbw ymm0, [%[a_ptr]]\n\t"
      "vpshufd ymm2, ymm1, 0x00\n\t"
      "vpshufd ymm3, ymm1, 0x55\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm4, ymm2, ymm4\n\t"
      "vpaddd ymm5, ymm3, ymm5\n\t"
      "vpshufd ymm2, ymm1, 0xaa\n\t"
      "vpshufd ymm3, ymm1, 0xff\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm6, ymm2, ymm6\n\t"
      "vpaddd ymm7, ymm3, ymm7\n\t"

      // a_ptr section 1
      "vpmovsxbw ymm0, [%[a_ptr] + 0x10]\n\t"
      "vpshufd ymm2, ymm1, 0x00\n\t"
      "vpshufd ymm3, ymm1, 0x55\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm8, ymm2, ymm8\n\t"
      "vpaddd ymm9, ymm3, ymm9\n\t"
      "vpshufd ymm2, ymm1, 0xaa\n\t"
      "vpshufd ymm3, ymm1, 0xff\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm10, ymm2, ymm10\n\t"
      "vpaddd ymm11, ymm3, ymm11\n\t"

      // a_ptr section 2
      "vpmovsxbw ymm0, [%[a_ptr] + 0x20]\n\t"
      "vpshufd ymm2, ymm1, 0x00\n\t"
      "vpshufd ymm3, ymm1, 0x55\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm12, ymm2, ymm12\n\t"
      "vpaddd ymm13, ymm3, ymm13\n\t"
      "vpshufd ymm2, ymm1, 0xaa\n\t"
      "vpshufd ymm3, ymm1, 0xff\n\t"
      "vpmaddwd ymm2, ymm0, ymm2\n\t"
      "vpmaddwd ymm3, ymm0, ymm3\n\t"
      "vpaddd ymm14, ymm2, ymm14\n\t"
      "vpaddd ymm15, ymm3, ymm15\n\t"

      "add %[a_ptr], 0x30 \n\t"
      "add %[b_ptr], 0x08 \n\t"

      "dec %[batch]\n\t"
      "jnz _matmul_m24_n4_k2_loop_start_ia_ib\n\t"

      // add dst
      "vpaddd ymm4, ymm4, [%[dst_ptr]]\n\t"
      "vpaddd ymm8, ymm8, [%[dst_ptr] + 0x20]\n\t"
      "vpaddd ymm12, ymm12, [%[dst_ptr] + 0x40]\n\t"

      "vpaddd ymm5, ymm5, [%[dst_ptr] + %[stride] * 1]\n\t"
      "vpaddd ymm9, ymm9, [%[dst_ptr] + %[stride] * 1 + 0x20]\n\t"
      "vpaddd ymm13, ymm13, [%[dst_ptr] + %[stride] * 1 + 0x40]\n\t"

      "vpaddd ymm6, ymm6, [%[dst_ptr] + %[stride] * 2]\n\t"
      "vpaddd ymm10, ymm10, [%[dst_ptr] + %[stride] * 2 + 0x20]\n\t"
      "vpaddd ymm14, ymm14, [%[dst_ptr] + %[stride] * 2 + 0x40]\n\t"

      "vpaddd ymm7, ymm7, [r14 + %[stride] * 1]\n\t"
      "vpaddd ymm11, ymm11, [r14 + %[stride] * 1 + 0x20]\n\t"
      "vpaddd ymm15, ymm15, [r14 + %[stride] * 1 + 0x40]\n\t"

      // save to dst
      "vmovdqu [%[dst_ptr]], ymm4\n\t"
      "vmovdqu [%[dst_ptr] + 0x20], ymm8\n\t"
      "vmovdqu [%[dst_ptr] + 0x40], ymm12\n\t"

      "vmovdqu [%[dst_ptr] + %[stride] * 1], ymm5\n\t"
      "vmovdqu [%[dst_ptr] + %[stride] * 1 + 0x20], ymm9\n\t"
      "vmovdqu [%[dst_ptr] + %[stride] * 1 + 0x40], ymm13\n\t"

      "vmovdqu [%[dst_ptr] + %[stride] * 2], ymm6\n\t"
      "vmovdqu [%[dst_ptr] + %[stride] * 2 + 0x20], ymm10\n\t"
      "vmovdqu [%[dst_ptr] + %[stride] * 2 + 0x40], ymm14\n\t"

      "vmovdqu [r14 + %[stride] * 1], ymm7\n\t"
      "vmovdqu [r14 + %[stride] * 1 + 0x20], ymm11\n\t"
      "vmovdqu [r14 + %[stride] * 1 + 0x40], ymm15\n\t"
      :
      : [ a_ptr ] "r"(a), [ b_ptr ] "r"(b), [ dst_ptr ] "r"(dst),
        [ stride ] "r"(stride * sizeof(int32_t)), [ batch ] "r"(batch)
      : "memory", "cc", "r14",
#if __GNUC__ >= 5
        "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
        "ymm11", "ymm12", "ymm13", "ymm14", "ymm15"
#else
        "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10",
        "xmm11", "xmm12", "xmm13", "xmm14", "xmm15"
#endif
  );
}

}  // namespace tfcc
