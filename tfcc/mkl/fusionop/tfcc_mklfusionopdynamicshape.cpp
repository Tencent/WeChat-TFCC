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

#include "tfcc_mklfusionopdynamicshape.h"

#include <map>
#include <set>

#include "tfcc_mklfusionop.h"

namespace tfcc {
namespace fusionop {

static const int kFusionOpDynamicShapeMask[56] = {
    -1, -1, -1, -1, -1, -1, -1, 0,  -1, -1, -1, -1, -1, -1, 0,  0,  -1, -1, -1,
    -1, -1, 0,  0,  0,  -1, -1, -1, -1, 0,  0,  0,  0,  -1, -1, -1, 0,  0,  0,
    0,  0,  -1, -1, 0,  0,  0,  0,  0,  0,  -1, 0,  0,  0,  0,  0,  0,  0,
};

FusionOpDynamicShape::FusionOpDynamicShape(
    const std::vector<OperationType>& opTypes, const std::vector<std::vector<bool>>& broadcastMarks)
    : _jit(4096 * 1024, Xbyak::AutoGrow), _broadcastMarks(broadcastMarks) {
  // rdi rsi rdx rcx r8 r9

  // 1. get param count.
  // 2. save rbx, rbp, r12-r15 if need.
  // 3. mov inputs to r8-r15
  // 4. mov result to rax
  // 5. clear rsi

  // 6. create loop asm.

  // 逆波兰解析
  // {
  // if current op type is input, r = manager.getInputRegister
  // else operation = get_operation(opType); r = operation->process(xxx);
  // manager.nextPosition();
  // }

  // 8. save result to dst
  // 9. end loop asm
  // 10. recover rbx, rbp, r12-r15 if need.
  // 11. ret

  // r8, r9, r10, r11
  // rbx, rbp, r12, r13, r14, r15
  std::vector<Xbyak::Reg> generalReg = {_jit.r8,  _jit.r9,  _jit.r10, _jit.r11, _jit.r12,
                                        _jit.r13, _jit.r14, _jit.r15, _jit.rbx, _jit.rbp};
  std::vector<Xbyak::Ymm> ymmVec = {
      _jit.ymm0, _jit.ymm1, _jit.ymm2,  _jit.ymm3,  _jit.ymm4,  _jit.ymm5,  _jit.ymm6,  _jit.ymm7,
      _jit.ymm8, _jit.ymm9, _jit.ymm10, _jit.ymm11, _jit.ymm12, _jit.ymm13, _jit.ymm14, _jit.ymm15};
  _manager.reset(new RegisterManager(opTypes.size(), generalReg, ymmVec, _jit));

  auto positionTuple = get_position_map(opTypes);
  std::map<OperationType, std::set<size_t>>& paramPositionMap = std::get<0>(positionTuple);
  std::map<size_t, std::set<size_t>>& opPositionsMap = std::get<1>(positionTuple);
  size_t paramSize = paramPositionMap.size();

  int index = 0;
  std::map<int, std::shared_ptr<GeneralRegister>> indexRegMap;
  std::map<OperationType, std::shared_ptr<GeneralRegister>> regMap;
  std::map<OperationType, bool> broadcastMarksMap;
  for (auto& it : paramPositionMap) {
    auto p = _manager->getGeneralRegister(0);
    _jit.mov(p->reg(), _jit.qword[_jit.rdi + 8 * index]);
    indexRegMap[index] = p;
    regMap[it.first] = p;
    broadcastMarksMap[it.first] = broadcastMarks[index].back();
    ++index;
  }

  _jit.mov(_jit.rax, _jit.rsi);

  uint64_t dims = broadcastMarks[0].size();
  std::vector<std::pair<std::shared_ptr<GeneralRegister>, std::shared_ptr<GeneralRegister>>>
      dimsOffsetMap;
  std::vector<Xbyak::Label> loops;
  if (dims >= 2) {
    for (uint64_t i = 0; i <= dims - 2; ++i) {
      auto d = _manager->getGeneralRegister(64);
      auto c = _manager->getGeneralRegister(64);
      _jit.mov(d->reg(), _jit.qword[_jit.rdx + 8 * i]);
      dimsOffsetMap.push_back({d, c});
    }
    _jit.mov(_jit.rdx, _jit.qword[_jit.rdx + 8 * (dims - 1)]);

    for (uint64_t i = 0; i <= dims - 2; ++i) {
      _jit.and_(dimsOffsetMap[i].second->reg(), 0x0);
      loops.push_back(Xbyak::Label());
      _jit.L(loops.back());
    }
  } else {
    _jit.mov(_jit.rdx, _jit.qword[_jit.rdx]);
  }

  _jit.xor_(_jit.rsi, _jit.rsi);
  Xbyak::Label loopLabel;
  Xbyak::Label nomaskLabel;
  _jit.L(loopLabel);

  std::shared_ptr<Register> result =
      do_rpn(opTypes, paramPositionMap, opPositionsMap, regMap, broadcastMarksMap, *_manager);

  _jit.vmovups(_jit.ptr[_jit.rax + _jit.rsi * sizeof(float)], result->reg());
  _jit.add(_jit.rsi, 0x8);
  _jit.cmp(_jit.rdx, _jit.rsi);
  _jit.ja(loopLabel, _jit.T_NEAR);
  _jit.je(nomaskLabel, _jit.T_NEAR);

  _jit.mov(_jit.rdi, _jit.rsi);
  _jit.sub(_jit.rdi, _jit.rdx);
  _jit.shl(_jit.rdi, 5);
  _jit.add(_jit.rdi, reinterpret_cast<uintptr_t>(kFusionOpDynamicShapeMask) - 32);
  auto tempReg = _manager->getTempRegister({_manager->getPosition()}, 128);
  _jit.vmovups(tempReg->reg(), _jit.ptr[_jit.rdi]);
  _jit.mov(_jit.rdi, _jit.rsi);
  _jit.sub(_jit.rdi, 0x8);
  _jit.vmaskmovps(_jit.ptr[_jit.rax + _jit.rdi * sizeof(float)], tempReg->reg(), result->reg());

  _jit.L(nomaskLabel);
  if (dims >= 2) {
    _jit.lea(_jit.rax, _jit.ptr[_jit.rax + _jit.rsi * sizeof(float)]);
    for (int i = dims - 2; i >= 0; --i) {
      for (size_t j = 0; j < paramSize; ++j) {
        _jit.mov(_jit.rdi, _jit.qword[_jit.rcx + 8 * (i + j * dims)]);
        _jit.lea(indexRegMap[j]->reg(), _jit.ptr[indexRegMap[j]->reg() + _jit.rdi * sizeof(float)]);
      }
      _jit.add(dimsOffsetMap[i].second->reg(), 0x1);
      _jit.cmp(dimsOffsetMap[i].first->reg(), dimsOffsetMap[i].second->reg());
      _jit.ja(loops[i], _jit.T_NEAR);
    }
  }

  _manager->freeStack();

  _jit.ret();
  _jit.readyRE();
  _func = _jit.getCode<void (*)(const float* const*, float*, const uint64_t*, const int64_t*)>();
}

void FusionOpDynamicShape::process(
    const float* const* inputs, float* result, const std::vector<unsigned>& resultShape) const {
  assert(resultShape.size() == _broadcastMarks[0].size());
  std::vector<std::vector<int64_t>> inputsSkipSize =
      get_inputs_skip_size(resultShape, _broadcastMarks);
  std::vector<uint64_t> shape(resultShape.size());
  std::vector<int64_t> skips(_broadcastMarks.size() * resultShape.size());
  for (size_t i = 0; i < resultShape.size(); ++i) {
    shape[i] = resultShape[i];
  }
  for (size_t i = 0; i < _broadcastMarks.size(); ++i) {
    for (size_t j = 0; j < resultShape.size(); ++j) {
      skips[j + i * resultShape.size()] = inputsSkipSize[i][j];
    }
  }
  _func(inputs, result, shape.data(), skips.data());
}

}  // namespace fusionop
}  // namespace tfcc
