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

#include "jitutils.h"

#include "3rd/xbyak/xbyak.h"

#include "tfcc_runtime/exceptions/runtimeerror.h"
#include "tfcc_runtime/framework/graph.h"

namespace tfcc {
namespace runtime {
namespace jit {

void call_function_inner(Xbyak::CodeGenerator& jit, uintptr_t func, size_t& pos, size_t& fpos) {
  jit.mov(jit.rax, func);
  jit.call(jit.rax);
}

static inline bool is_in_int32(uint64_t x) {
  return ~uint64_t(0x7fffffffu) <= x || x <= 0x7fffffffu;
}

void set_argument_inner(Xbyak::CodeGenerator& jit, size_t& pos, uint64_t value) {
  if (pos == 0) {
    jit.mov(jit.rdi, value);
  } else if (pos == 1) {
    jit.mov(jit.rsi, value);
  } else if (pos == 2) {
    jit.mov(jit.rdx, value);
  } else if (pos == 3) {
    jit.mov(jit.rcx, value);
  } else if (pos == 4) {
    jit.mov(jit.r8, value);
  } else if (pos == 5) {
    jit.mov(jit.r9, value);
  } else {
    if (!is_in_int32(value)) {
      jit.mov(jit.rax, value);
      jit.mov(jit.qword[jit.rsp + (pos - 6) * 8], jit.rax);
    } else {
      jit.mov(jit.qword[jit.rsp + (pos - 6) * 8], value);
    }
  }
  pos += 1;
}

void set_argument_inner(Xbyak::CodeGenerator& jit, size_t& pos, float value) {
  union ValueWrapper {
    uint32_t uint32Value;
    float value;
  };

  constexpr size_t TMP_POS = 15 * 8;
  static_assert(
      sizeof(ValueWrapper) == sizeof(float) && sizeof(uint32_t) == sizeof(float), "unknow error");
  static_assert(TMP_POS < Graph::RESERVED_STACK_SIZE && TMP_POS % 8 == 0, "stack error");

  ValueWrapper wrapper;
  wrapper.uint32Value = 0;
  wrapper.value = value;
  if (!is_in_int32(wrapper.uint32Value)) {
    jit.mov(jit.eax, wrapper.uint32Value);
    jit.mov(jit.qword[jit.rsp + TMP_POS], jit.eax);
  } else {
    jit.mov(jit.qword[jit.rsp + TMP_POS], wrapper.uint32Value);
  }
  if (pos == 0) {
    jit.movss(jit.xmm0, jit.dword[jit.rsp + TMP_POS]);
  } else if (pos == 1) {
    jit.movss(jit.xmm1, jit.dword[jit.rsp + TMP_POS]);
  } else if (pos == 2) {
    jit.movss(jit.xmm2, jit.dword[jit.rsp + TMP_POS]);
  } else if (pos == 3) {
    jit.movss(jit.xmm3, jit.dword[jit.rsp + TMP_POS]);
  } else if (pos == 4) {
    jit.movss(jit.xmm4, jit.dword[jit.rsp + TMP_POS]);
  } else if (pos == 5) {
    jit.movss(jit.xmm5, jit.dword[jit.rsp + TMP_POS]);
  } else if (pos == 6) {
    jit.movss(jit.xmm6, jit.dword[jit.rsp + TMP_POS]);
  } else if (pos == 7) {
    jit.movss(jit.xmm7, jit.dword[jit.rsp + TMP_POS]);
  } else {
    throw RuntimeError("Unknow error");
  }
  pos += 1;
}

void set_argument_inner(Xbyak::CodeGenerator& jit, size_t& pos, double value) {
  union ValueWrapper {
    uint64_t uint64Value;
    double value;
  };

  constexpr size_t TMP_POS = 15 * 8;
  static_assert(
      sizeof(ValueWrapper) == sizeof(double) && sizeof(uint64_t) == sizeof(double), "unknow error");
  static_assert(TMP_POS < Graph::RESERVED_STACK_SIZE && TMP_POS % 8 == 0, "stack error");

  ValueWrapper wrapper;
  wrapper.uint64Value = 0;
  wrapper.value = value;
  if (!is_in_int32(wrapper.uint64Value)) {
    jit.mov(jit.rax, wrapper.uint64Value);
    jit.mov(jit.qword[jit.rsp + TMP_POS], jit.rax);
  } else {
    jit.mov(jit.qword[jit.rsp + TMP_POS], wrapper.uint64Value);
  }
  if (pos == 0) {
    jit.movsd(jit.xmm0, jit.qword[jit.rsp + TMP_POS]);
  } else if (pos == 1) {
    jit.movsd(jit.xmm1, jit.qword[jit.rsp + TMP_POS]);
  } else if (pos == 2) {
    jit.movsd(jit.xmm2, jit.qword[jit.rsp + TMP_POS]);
  } else if (pos == 3) {
    jit.movsd(jit.xmm3, jit.qword[jit.rsp + TMP_POS]);
  } else if (pos == 4) {
    jit.movsd(jit.xmm4, jit.qword[jit.rsp + TMP_POS]);
  } else if (pos == 5) {
    jit.movsd(jit.xmm5, jit.qword[jit.rsp + TMP_POS]);
  } else if (pos == 6) {
    jit.movsd(jit.xmm6, jit.qword[jit.rsp + TMP_POS]);
  } else if (pos == 7) {
    jit.movsd(jit.xmm7, jit.qword[jit.rsp + TMP_POS]);
  } else {
    throw RuntimeError("Unknow error");
  }
  pos += 1;
}

void mov_rax_to_rdi(Xbyak::CodeGenerator& jit) { jit.mov(jit.rdi, jit.rax); }

}  // namespace jit
}  // namespace runtime
}  // namespace tfcc
