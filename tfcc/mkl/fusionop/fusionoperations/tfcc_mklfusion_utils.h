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

#include "fusionop/tfcc_mklregister.h"
#include "fusionop/tfcc_mklregistermanager.h"

namespace tfcc {
namespace fusionop {

void abs(RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& a);
void is_finite(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& a);
void is_nan(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& a);
void select(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& s,
    std::shared_ptr<Register>& a, std::shared_ptr<Register>& b);
void mul_add(
    RegisterManager& manager, std::shared_ptr<Register>& a, std::shared_ptr<Register>& b,
    std::shared_ptr<Register>& c);
// the result must different from a, b and c.
void mul_add(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& a,
    std::shared_ptr<Register>& b, std::shared_ptr<Register>& c);
void nmul_add(
    RegisterManager& manager, std::shared_ptr<Register>& a, std::shared_ptr<Register>& b,
    std::shared_ptr<Register>& c);
// the result must different from a, b and c.
void nmul_add(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& a,
    std::shared_ptr<Register>& b, std::shared_ptr<Register>& c);
void polynomial_4(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& x,
    const float c0, const float c1, const float c2, const float c3, const float c4);
void polynomial_5(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& x,
    const float c0, const float c1, const float c2, const float c3, const float c4, const float c5);
void polynomial_8(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& x,
    const float c0, const float c1, const float c2, const float c3, const float c4, const float c5,
    const float c6, const float c7, const float c8);
void pow2in(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& n);
void sign_bit(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& a);
void sign_combine(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& a,
    std::shared_ptr<Register>& b);
void fraction_2(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& a);
void exponent_f(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& a);
void is_subnormal(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& a);
void is_inf(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& a);
void taylor_exp_f(
    RegisterManager& manager, std::shared_ptr<Register>& result,
    std::shared_ptr<Register>& initialX);
void taylor_erf_f(
    RegisterManager& manager, std::shared_ptr<Register>& result,
    std::shared_ptr<Register>& initialX);
void taylor_tanh_f(
    RegisterManager& manager, std::shared_ptr<Register>& result,
    std::shared_ptr<Register>& initialX);
void taylor_log_f(
    RegisterManager& manager, std::shared_ptr<Register>& result,
    std::shared_ptr<Register>& initialX);

}  // namespace fusionop
}  // namespace tfcc
