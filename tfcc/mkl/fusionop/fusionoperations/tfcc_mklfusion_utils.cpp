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

#include "tfcc_mklfusion_utils.h"

#include <set>

namespace tfcc {
namespace fusionop {

void abs(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& a) {
  auto mask = manager.getUint32ConstantRegister(0x7fffffff, 96);
  manager.jit().vandps(result->reg(), a->reg(), mask->reg());
}

void is_finite(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& a) {
  auto mask = manager.getUint32ConstantRegister(0xff000000, 96);
  auto minusone = manager.getUint32ConstantRegister(-1, 96);

  Xbyak::CodeGenerator& jit = manager.jit();
  jit.vpslld(result->reg(), a->reg(), 1);
  jit.vpand(result->reg(), result->reg(), mask->reg());
  jit.vpcmpeqd(result->reg(), result->reg(), mask->reg());
  jit.vpxor(result->reg(), result->reg(), minusone->reg());
}

void is_nan(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& a) {
  auto mask = manager.getUint32ConstantRegister(0xff000000, 96);
  auto zero = manager.getUint32ConstantRegister(0, 96);
  auto one = manager.getUint32ConstantRegister(1, 128);
  auto minusone = manager.getUint32ConstantRegister(-1, 96);

  Xbyak::CodeGenerator& jit = manager.jit();
  auto t2 = manager.getTempRegister({manager.getPosition()}, 0);

  jit.vpslld(t2->reg(), a->reg(), 1);
  jit.vpandn(result->reg(), mask->reg(), t2->reg());
  jit.vpcmpeqd(result->reg(), result->reg(), zero->reg());
  jit.vpxor(result->reg(), result->reg(), minusone->reg());

  jit.vpand(t2->reg(), t2->reg(), mask->reg());
  jit.vpcmpeqd(t2->reg(), t2->reg(), mask->reg());

  jit.vpand(result->reg(), result->reg(), t2->reg());
}

void select(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& s,
    std::shared_ptr<Register>& a, std::shared_ptr<Register>& b) {
  manager.jit().vblendvps(result->reg(), b->reg(), a->reg(), s->reg());
}

void mul_add(
    RegisterManager& manager, std::shared_ptr<Register>& a, std::shared_ptr<Register>& b,
    std::shared_ptr<Register>& c) {
  Xbyak::CodeGenerator& jit = manager.jit();
  jit.vfmadd213ps(a->reg(), b->reg(), c->reg());
}

void mul_add(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& a,
    std::shared_ptr<Register>& b, std::shared_ptr<Register>& c) {
  Xbyak::CodeGenerator& jit = manager.jit();
  jit.vmovaps(result->reg(), a->reg());
  jit.vfmadd213ps(result->reg(), b->reg(), c->reg());
}

void nmul_add(
    RegisterManager& manager, std::shared_ptr<Register>& a, std::shared_ptr<Register>& b,
    std::shared_ptr<Register>& c) {
  Xbyak::CodeGenerator& jit = manager.jit();
  jit.vfnmadd213ps(a->reg(), b->reg(), c->reg());
}

void nmul_add(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& a,
    std::shared_ptr<Register>& b, std::shared_ptr<Register>& c) {
  Xbyak::CodeGenerator& jit = manager.jit();
  jit.vmovaps(result->reg(), a->reg());
  jit.vfnmadd213ps(result->reg(), b->reg(), c->reg());
}

void polynomial_4(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& x,
    const float c0, const float c1, const float c2, const float c3, const float c4) {
  Xbyak::CodeGenerator& jit = manager.jit();
  std::set<size_t> positions({manager.getPosition()});

  auto x2 = manager.getTempRegister(positions, 0);
  auto x4 = manager.getTempRegister(positions, 0);
  jit.vmulps(x2->reg(), x->reg(), x->reg());
  jit.vmulps(x4->reg(), x2->reg(), x2->reg());
  auto r4 = manager.getFloatConstantRegister(c4, 128);
  jit.vmulps(x4->reg(), r4->reg(), x4->reg());

  auto r0 = manager.getFloatConstantRegister(c0, 128);
  auto r1 = manager.getFloatConstantRegister(c1, 128);
  auto r2 = manager.getFloatConstantRegister(c2, 128);
  auto r3 = manager.getFloatConstantRegister(c3, 128);
  auto temp = manager.getTempRegister(positions, 0);
  mul_add(manager, temp, r1, x, r0);
  jit.vaddps(x4->reg(), temp->reg(), x4->reg());
  temp.reset();
  mul_add(manager, result, r3, x, r2);
  mul_add(manager, result, x2, x4);
}

void polynomial_5(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& x,
    const float c0, const float c1, const float c2, const float c3, const float c4,
    const float c5) {
  std::set<size_t> positions({manager.getPosition()});
  Xbyak::CodeGenerator& jit = manager.jit();

  auto x2 = manager.getTempRegister(positions, 0);
  auto x4 = manager.getTempRegister(positions, 0);
  jit.vmulps(x2->reg(), x->reg(), x->reg());
  jit.vmulps(x4->reg(), x2->reg(), x2->reg());

  auto r0 = manager.getFloatConstantRegister(c0, 128);
  auto r1 = manager.getFloatConstantRegister(c1, 128);
  auto r2 = manager.getFloatConstantRegister(c2, 128);
  auto r3 = manager.getFloatConstantRegister(c3, 128);
  auto r4 = manager.getFloatConstantRegister(c4, 128);
  auto r5 = manager.getFloatConstantRegister(c5, 128);
  auto temp = manager.getTempRegister(positions, 0);
  auto temp2 = manager.getTempRegister(positions, 0);
  mul_add(manager, temp, r1, x, r0);
  mul_add(manager, temp2, r5, x, r4);
  mul_add(manager, temp2, x4, temp);
  x4.reset();
  temp.reset();
  mul_add(manager, result, r3, x, r2);
  mul_add(manager, result, x2, temp2);
}

void polynomial_8(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& x,
    const float c0, const float c1, const float c2, const float c3, const float c4, const float c5,
    const float c6, const float c7, const float c8) {
  std::set<size_t> positions({manager.getPosition()});
  Xbyak::CodeGenerator& jit = manager.jit();

  auto r4 = manager.getFloatConstantRegister(c4, 128);
  auto r5 = manager.getFloatConstantRegister(c5, 128);
  auto r6 = manager.getFloatConstantRegister(c6, 128);
  auto r7 = manager.getFloatConstantRegister(c7, 128);
  auto temp = manager.getTempRegister(positions, 0);
  auto result1 = manager.getTempRegister(positions, 0);
  mul_add(manager, temp, r5, x, r4);
  mul_add(manager, result1, r7, x, r6);
  auto x2 = manager.getTempRegister(positions, 0);
  jit.vmulps(x2->reg(), x->reg(), x->reg());
  mul_add(manager, result1, x2, temp);

  auto r0 = manager.getFloatConstantRegister(c0, 128);
  auto r1 = manager.getFloatConstantRegister(c1, 128);
  auto r2 = manager.getFloatConstantRegister(c2, 128);
  auto r3 = manager.getFloatConstantRegister(c3, 128);
  mul_add(manager, temp, r3, x, r2);
  auto temp2 = manager.getTempRegister(positions, 0);
  mul_add(manager, temp2, r1, x, r0);

  auto r8 = manager.getFloatConstantRegister(c8, 128);
  auto x4 = manager.getTempRegister(positions, 0);
  auto x8 = manager.getTempRegister(positions, 0);
  jit.vmulps(x4->reg(), x2->reg(), x2->reg());
  jit.vmulps(x8->reg(), x4->reg(), x4->reg());
  jit.vmulps(x8->reg(), r8->reg(), x8->reg());

  jit.vaddps(temp2->reg(), temp2->reg(), x8->reg());
  x8.reset();
  mul_add(manager, temp, x2, temp2);
  x2.reset();
  temp2.reset();
  mul_add(manager, result, result1, x4, temp);
}

void pow2in(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& n) {
  constexpr float pow2_23 = 8388608.0;
  constexpr float bias = 127.0;
  auto a = manager.getFloatConstantRegister(bias + pow2_23, 128);
  manager.jit().vaddps(result->reg(), n->reg(), a->reg());
  manager.jit().vpslld(result->reg(), result->reg(), 0x17);
}

void sign_bit(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& a) {
  Xbyak::CodeGenerator& jit = manager.jit();
  jit.vpsrad(result->reg(), a->reg(), 0x1f);
}

void sign_combine(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& a,
    std::shared_ptr<Register>& b) {
  auto mask = manager.getUint32ConstantRegister(0x80000000, 128);
  auto temp = manager.getTempRegister({manager.getPosition()}, 0);
  Xbyak::CodeGenerator& jit = manager.jit();
  jit.vandps(temp->reg(), b->reg(), mask->reg());
  jit.vxorps(result->reg(), a->reg(), temp->reg());
}

void fraction_2(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& a) {
  auto mask1 = manager.getUint32ConstantRegister(0x007fffff, 128);
  auto mask2 = manager.getUint32ConstantRegister(0x3f000000, 128);
  Xbyak::CodeGenerator& jit = manager.jit();
  jit.vpand(result->reg(), a->reg(), mask1->reg());
  jit.vpor(result->reg(), result->reg(), mask2->reg());
}

void exponent_f(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& a) {
  Xbyak::CodeGenerator& jit = manager.jit();
  auto pow2_23 = manager.getFloatConstantRegister(8388608.0f, 128);
  auto bias = manager.getFloatConstantRegister(8388608.0f + 127.f, 128);
  jit.vpsrld(result->reg(), a->reg(), 0x17);
  jit.vpor(result->reg(), result->reg(), pow2_23->reg());
  jit.vsubps(result->reg(), result->reg(), bias->reg());
}

void is_subnormal(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& a) {
  Xbyak::CodeGenerator& jit = manager.jit();
  auto mask = manager.getUint32ConstantRegister(0xff000000, 96);
  auto zero = manager.getUint32ConstantRegister(0, 96);
  jit.vpslld(result->reg(), a->reg(), 0x1);
  jit.vpand(result->reg(), result->reg(), mask->reg());
  auto temp = manager.getTempRegister({manager.getPosition()}, 0);
  jit.vpandn(temp->reg(), mask->reg(), result->reg());
  jit.vpcmpeqd(result->reg(), result->reg(), zero->reg());
  jit.vpcmpeqd(temp->reg(), temp->reg(), zero->reg());
  jit.vpandn(result->reg(), temp->reg(), result->reg());
}

void is_inf(
    RegisterManager& manager, std::shared_ptr<Register>& result, std::shared_ptr<Register>& a) {
  Xbyak::CodeGenerator& jit = manager.jit();
  auto mask = manager.getUint32ConstantRegister(0xff000000, 96);
  jit.vpslld(result->reg(), a->reg(), 0x1);
  jit.vpcmpeqd(result->reg(), result->reg(), mask->reg());
}

void taylor_exp_f(
    RegisterManager& manager, std::shared_ptr<Register>& result,
    std::shared_ptr<Register>& initialX) {
  constexpr float p0 = 1.f / 2.f;
  constexpr float p1 = 1.f / 6.f;
  constexpr float p2 = 1.f / 24.f;
  constexpr float p3 = 1.f / 120.f;
  constexpr float p4 = 1.f / 720.f;
  constexpr float p5 = 1.f / 5040.f;

  Xbyak::CodeGenerator& jit = manager.jit();
  std::set<size_t> positions({manager.getPosition()});
  Xbyak::Label outLabel;

  auto ln2d1 = manager.getFloatConstantRegister(0.693359375f, 128);
  auto ln2d2 = manager.getFloatConstantRegister(-2.12194440e-4f, 128);
  auto log2e = manager.getFloatConstantRegister(1.44269504088896340736f, 128);

  auto x = manager.getTempRegister(positions, 0);
  auto x2 = manager.getTempRegister(positions, 0);
  auto r = manager.getTempRegister(positions, 0);

  jit.vmovups(x->reg(), initialX->reg());
  jit.vmulps(r->reg(), x->reg(), log2e->reg());
  jit.vroundps(r->reg(), r->reg(), 8);

  auto temp = manager.getTempRegister(positions, 0);
  nmul_add(manager, temp, r, ln2d1, x);
  nmul_add(manager, x2, r, ln2d2, temp);

  jit.vmulps(temp->reg(), x2->reg(), x2->reg());
  polynomial_5(manager, result, x2, p0, p1, p2, p3, p4, p5);
  mul_add(manager, result, temp, x2);

  pow2in(manager, r, r);
  auto one = manager.getFloatConstantRegister(1.0f, 96);
  jit.vaddps(result->reg(), result->reg(), one->reg());
  jit.vmulps(result->reg(), result->reg(), r->reg());

  auto xMax = manager.getFloatConstantRegister(87.3f, 128);
  abs(manager, x2, x);
  jit.vcmpps(x2->reg(), x2->reg(), xMax->reg(), 1);
  is_finite(manager, temp, x);
  jit.vandps(x2->reg(), x2->reg(), temp->reg());

  auto minusone = manager.getFloatConstantRegister(-1.0f, 96);
  jit.vtestps(x2->reg(), minusone->reg());
  jit.jb(outLabel, jit.T_NEAR);

  auto zero = manager.getFloatConstantRegister(0.f, 96);
  auto vInfinite = manager.getFloatConstantRegister(std::numeric_limits<float>::infinity(), 128);
  sign_bit(manager, temp, x);
  select(manager, r, temp, zero, vInfinite);
  select(manager, temp, x2, result, r);
  x2.reset();
  is_nan(manager, r, x);
  select(manager, result, r, x, temp);
  jit.L(outLabel);
}

void taylor_erf_f(
    RegisterManager& manager, std::shared_ptr<Register>& result,
    std::shared_ptr<Register>& initialX) {}

void taylor_tanh_f(
    RegisterManager& manager, std::shared_ptr<Register>& result,
    std::shared_ptr<Register>& initialX) {
  constexpr float r0 = -3.33332819422e-1f;
  constexpr float r1 = 1.33314422036E-1f;
  constexpr float r2 = -5.37397155531E-2f;
  constexpr float r3 = 2.06390887954E-2f;
  constexpr float r4 = -5.70498872745E-3f;

  Xbyak::CodeGenerator& jit = manager.jit();
  Xbyak::Label label;
  Xbyak::Label andLabel;

  std::set<size_t> xpositions({manager.getPosition()});
  auto x = manager.getTempRegister(xpositions, 0);
  auto xSmall = manager.getTempRegister(xpositions, 0);
  auto cmpconstant = manager.getFloatConstantRegister(0.625f, 128);
  abs(manager, x, initialX);
  jit.vcmpleps(xSmall->reg(), x->reg(), cmpconstant->reg());

  jit.vtestps(xSmall->reg(), xSmall->reg());
  jit.je(andLabel, jit.T_NEAR);

  auto temp = manager.getTempRegister(xpositions, 0);
  jit.vmulps(temp->reg(), x->reg(), x->reg());
  polynomial_4(manager, result, temp, r0, r1, r2, r3, r4);
  jit.vmulps(temp->reg(), temp->reg(), x->reg());
  mul_add(manager, result, temp, x);

  jit.L(andLabel);
  auto minusone = manager.getFloatConstantRegister(-1.0f, 96);
  jit.vtestps(xSmall->reg(), minusone->reg());
  jit.jb(label, jit.T_NEAR);

  jit.vaddps(temp->reg(), x->reg(), x->reg());
  taylor_exp_f(manager, temp, temp);
  auto one = manager.getFloatConstantRegister(1.0f, 96);
  auto two = manager.getFloatConstantRegister(2.0f, 128);
  jit.vaddps(temp->reg(), temp->reg(), one->reg());
  jit.vdivps(temp->reg(), two->reg(), temp->reg());
  jit.vsubps(temp->reg(), one->reg(), temp->reg());

  jit.L(label);
  auto fffconstant = manager.getFloatConstantRegister(44.4f, 128);
  jit.vcmpltps(x->reg(), fffconstant->reg(), x->reg());

  select(manager, result, xSmall, result, temp);
  xSmall.reset();
  temp.reset();
  select(manager, result, x, one, result);
  x.reset();
  sign_combine(manager, result, result, initialX);
}

void taylor_log_f(
    RegisterManager& manager, std::shared_ptr<Register>& result,
    std::shared_ptr<Register>& initialX) {
  constexpr float p0log = 3.3333331174e-1f;
  constexpr float p1log = -2.4999993993e-1f;
  constexpr float p2log = 2.0000714765e-1f;
  constexpr float p3log = -1.6668057665e-1f;
  constexpr float p4log = 1.4249322787e-1f;
  constexpr float p5log = -1.2420140846e-1f;
  constexpr float p6log = 1.1676998740e-1f;
  constexpr float p7log = -1.1514610310e-1f;
  constexpr float p8log = 7.0376836292e-2f;

  Xbyak::CodeGenerator& jit = manager.jit();
  Xbyak::Label label;
  std::set<size_t> positions({manager.getPosition()});

  auto fe = manager.getTempRegister(positions, 0);
  auto x = manager.getTempRegister(positions, 0);
  exponent_f(manager, fe, initialX);
  fraction_2(manager, x, initialX);

  auto sqrt2 = manager.getFloatConstantRegister(1.41421356237309504880f * 0.5f, 128);
  auto minusone = manager.getUint32ConstantRegister(-1, 96);
  auto one = manager.getFloatConstantRegister(1.0f, 96);
  auto temp = manager.getTempRegister(positions, 0);
  auto blend = manager.getTempRegister(positions, 0);
  jit.vcmpps(blend->reg(), sqrt2->reg(), x->reg(), 1);
  jit.vpxor(temp->reg(), blend->reg(), minusone->reg());
  jit.vandps(temp->reg(), temp->reg(), x->reg());
  jit.vaddps(x->reg(), x->reg(), temp->reg());
  jit.vsubps(x->reg(), x->reg(), one->reg());
  jit.vandps(blend->reg(), blend->reg(), one->reg());
  jit.vaddps(fe->reg(), fe->reg(), blend->reg());
  blend.reset();

  polynomial_8(manager, temp, x, p0log, p1log, p2log, p3log, p4log, p5log, p6log, p7log, p8log);
  auto x2 = manager.getTempRegister(positions, 0);
  jit.vmulps(x2->reg(), x->reg(), x->reg());
  jit.vmulps(temp->reg(), temp->reg(), x2->reg());
  jit.vmulps(temp->reg(), temp->reg(), x->reg());

  auto half = manager.getFloatConstantRegister(0.5f, 128);
  auto ln2d2 = manager.getFloatConstantRegister(-2.12194440e-4f, 128);
  auto ln2d1 = manager.getFloatConstantRegister(0.693359375f, 128);
  auto res = manager.getTempRegister(positions, 0);
  mul_add(manager, res, fe, ln2d2, temp);
  temp.reset();
  nmul_add(manager, x2, half, x);
  x.reset();
  jit.vaddps(res->reg(), res->reg(), x2->reg());
  x2.reset();
  mul_add(manager, fe, ln2d1, res);
  res.reset();

  auto overflow = manager.getTempRegister(positions, 0);
  auto underflow = manager.getTempRegister(positions, 0);
  auto smallestNormal = manager.getFloatConstantRegister(1.17549435E-38f, 128);
  is_finite(manager, overflow, initialX);
  jit.vpxor(overflow->reg(), overflow->reg(), minusone->reg());
  jit.vcmpps(underflow->reg(), initialX->reg(), smallestNormal->reg(), 1);
  auto test = manager.getTempRegister(positions, 0);
  jit.vorps(test->reg(), overflow->reg(), underflow->reg());
  jit.vtestps(test->reg(), test->reg());
  jit.je(label, jit.T_NEAR);

  auto nan = manager.getUint32ConstantRegister(0x7fc00000 + 0x101, 128);
  auto nInf = manager.getFloatConstantRegister(-std::numeric_limits<float>::infinity(), 128);
  auto zero = manager.getFloatConstantRegister(0, 96);
  res = manager.getTempRegister(positions, 0);
  select(manager, res, underflow, nan, fe);
  jit.vcmpps(test->reg(), initialX->reg(), zero->reg(), 0);
  is_subnormal(manager, fe, initialX);
  jit.vorps(test->reg(), test->reg(), fe->reg());
  select(manager, res, test, nInf, res);
  select(manager, res, overflow, initialX, res);

  is_inf(manager, test, initialX);
  sign_bit(manager, fe, initialX);
  jit.vandps(fe->reg(), test->reg(), fe->reg());
  select(manager, fe, fe, nan, res);
  jit.L(label);
  jit.vmovaps(result->reg(), fe->reg());
}

}  // namespace fusionop
}  // namespace tfcc