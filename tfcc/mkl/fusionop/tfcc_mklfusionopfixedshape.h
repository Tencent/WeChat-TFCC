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

#include <memory>
#include <vector>
#include "operations/tfcc_mkloperationtype.h"
#include "tfcc_mklregistermanager.h"
#include "xbyak/xbyak.h"

namespace tfcc {
namespace fusionop {

class FusionOpFixedShape {
  Xbyak::CodeGenerator _jit;
  std::unique_ptr<RegisterManager> _manager;
  void (*_func)(const float* const*, float*);

 public:
  // 逆波兰
  FusionOpFixedShape(
      const std::vector<OperationType>& opTypes, const std::vector<unsigned>& resultShape,
      const std::vector<std::vector<bool> >& broadcastMarks);
  FusionOpFixedShape(const FusionOpFixedShape&) = delete;
  FusionOpFixedShape(FusionOpFixedShape&&) = delete;
  FusionOpFixedShape& operator=(const FusionOpFixedShape&) = delete;
  FusionOpFixedShape& operator=(FusionOpFixedShape&&) = delete;

  void process(const float* const* inputs, float* result);
};

}  // namespace fusionop
}  // namespace tfcc
