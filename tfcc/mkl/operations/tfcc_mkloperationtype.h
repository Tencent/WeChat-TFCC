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

namespace tfcc {
namespace fusionop {

enum OperationType {
  // operations
  ADD = 0,
  SUB = 1,
  MUL = 2,
  DIV = 3,
  ABS = 4,
  MIN = 5,
  MAX = 6,
  NEG = 7,
  SQRT = 8,
  RSQRT = 9,
  RELU = 10,
  TANH = 11,
  LEAKYRELU = 12,
  LOG = 13,
  SIGMOID = 14,
  SOFTPLUS = 15,
  RECIPROCAL = 16,
  CLIP = 17,
  // ERF = 18,
  // POW = 19,

  REPEATED = 1000000,

  // params
  PARAM_0 = 2000000,
  PARAM_1,
  PARAM_2,
  PARAM_3,
  PARAM_4,
  PARAM_5,
  PARAM_6,
  PARAM_7,
  PARAM_8,
  PARAM_9,
  PARAM_10,
  PARAM_11,
};

}  // namespace fusionop
}  // namespace tfcc
