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

#include "tfcc_mkloperation.h"
#include "fusionoperations/tfcc_mklfusion_abs.h"
#include "fusionoperations/tfcc_mklfusion_add.h"
#include "fusionoperations/tfcc_mklfusion_clip.h"
#include "fusionoperations/tfcc_mklfusion_div.h"
#include "fusionoperations/tfcc_mklfusion_leakyrelu.h"
#include "fusionoperations/tfcc_mklfusion_log.h"
#include "fusionoperations/tfcc_mklfusion_max.h"
#include "fusionoperations/tfcc_mklfusion_min.h"
#include "fusionoperations/tfcc_mklfusion_mul.h"
#include "fusionoperations/tfcc_mklfusion_neg.h"
#include "fusionoperations/tfcc_mklfusion_reciprocal.h"
#include "fusionoperations/tfcc_mklfusion_relu.h"
#include "fusionoperations/tfcc_mklfusion_rsqrt.h"
#include "fusionoperations/tfcc_mklfusion_sigmoid.h"
#include "fusionoperations/tfcc_mklfusion_softplus.h"
#include "fusionoperations/tfcc_mklfusion_sqrt.h"
#include "fusionoperations/tfcc_mklfusion_sub.h"
#include "fusionoperations/tfcc_mklfusion_tanh.h"
// #include "fusionoperations/tfcc_mklfusion_erf.h"
// #include "fusionoperations/tfcc_mklfusion_pow.h"

namespace tfcc {
namespace fusionop {

std::unique_ptr<Operation> get_operation(OperationType opType) {
  if (opType >= OperationType::REPEATED) {
    return nullptr;
  }
  switch (opType) {
    case OperationType::ADD:
      return std::unique_ptr<Operation>(new Add());
    case OperationType::SUB:
      return std::unique_ptr<Operation>(new Sub());
    case OperationType::MUL:
      return std::unique_ptr<Operation>(new Mul());
    case OperationType::DIV:
      return std::unique_ptr<Operation>(new Div());
    case OperationType::ABS:
      return std::unique_ptr<Operation>(new Abs());
    case OperationType::MIN:
      return std::unique_ptr<Operation>(new Min());
    case OperationType::MAX:
      return std::unique_ptr<Operation>(new Max());
    case OperationType::NEG:
      return std::unique_ptr<Operation>(new Neg());
    case OperationType::SQRT:
      return std::unique_ptr<Operation>(new Sqrt());
    case OperationType::RSQRT:
      return std::unique_ptr<Operation>(new Rsqrt());
    case OperationType::RELU:
      return std::unique_ptr<Operation>(new Relu());
    case OperationType::TANH:
      return std::unique_ptr<Operation>(new Tanh());
    case OperationType::LEAKYRELU:
      return std::unique_ptr<Operation>(new LeakyRelu());
    case OperationType::LOG:
      return std::unique_ptr<Operation>(new Log());
    case OperationType::SIGMOID:
      return std::unique_ptr<Operation>(new Sigmoid());
    case OperationType::SOFTPLUS:
      return std::unique_ptr<Operation>(new Softplus());
    case OperationType::RECIPROCAL:
      return std::unique_ptr<Operation>(new Reciprocal());
    case OperationType::CLIP:
      return std::unique_ptr<Operation>(new Clip());
    // case OperationType::ERF:
    //     return std::unique_ptr<Operation>(new Erf());
    // case OperationType::POW:
    //     return std::unique_ptr<Operation>(new POW());
    default:
      break;
  }
  return nullptr;
}

}  // namespace fusionop
}  // namespace tfcc
