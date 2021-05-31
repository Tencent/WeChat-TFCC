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

namespace tfcc {

template <class T, class C>
static inline T polynomial_2(const T& x, C c0, C c1, C c2) {
  auto x2 = x * x;
  return mul_add(x2, T(c2), mul_add(x, T(c1), T(c0)));
}

template <class T, class C>
static inline T polynomial_3(const T& x, C c0, C c1, C c2, C c3) {
  auto x2 = x * x;
  return mul_add(mul_add(T(c3), x, T(c2)), x2, mul_add(T(c1), x, T(c0)));
}

template <class T, class C>
static inline T polynomial_4(const T& x, C c0, C c1, C c2, C c3, C c4) {
  auto x2 = x * x;
  auto x4 = x2 * x2;
  return mul_add(mul_add(T(c3), x, T(c2)), x2, mul_add(T(c1), x, T(c0)) + T(c4) * x4);
}

template <class T, class C>
static inline T polynomial_5(const T& x, C c0, C c1, C c2, C c3, C c4, C c5) {
  auto x2 = x * x;
  auto x4 = x2 * x2;
  auto r1 = mul_add(mul_add(T(c5), x, T(c4)), x4, mul_add(T(c1), x, T(c0)));
  return mul_add(mul_add(T(c3), x, T(c2)), x2, r1);
}

template <class T, class C>
static inline T polynomial_5n(const T& x, C c0, C c1, C c2, C c3, C c4) {
  auto x2 = x * x;
  auto x4 = x2 * x2;
  auto r1 = mul_add(T(c4) + x, x4, mul_add(T(c1), x, T(c0)));
  return mul_add(mul_add(T(c3), x, T(c2)), x2, r1);
}

template <class T, class C>
static inline T polynomial_8(const T& x, C c0, C c1, C c2, C c3, C c4, C c5, C c6, C c7, C c8) {
  T x2 = x * x;
  T x4 = x2 * x2;
  T x8 = x4 * x4;
  auto r1 = mul_add(mul_add(T(c7), x, T(c6)), x2, mul_add(T(c5), x, T(c4)));
  auto r2 = mul_add(mul_add(T(c3), x, T(c2)), x2, mul_add(T(c1), x, T(c0)) + T(c8) * x8);
  return mul_add(r1, x4, r2);
}

template <class T, class C>
static inline T polynomial_13m(
    const T& x, C c2, C c3, C c4, C c5, C c6, C c7, C c8, C c9, C c10, C c11, C c12, C c13) {
  auto x2 = x * x;
  auto x4 = x2 * x2;
  auto x8 = x4 * x4;
  auto r1 = mul_add(
      mul_add(mul_add(T(c7), x, T(c6)), x2, mul_add(T(c5), x, T(c4))), x4,
      mul_add(mul_add(T(c3), x, T(c2)), x2, x));
  auto r2 = mul_add(
      mul_add(T(c13), x, T(c12)), x4,
      mul_add(mul_add(T(c11), x, T(c10)), x2, mul_add(T(c9), x, T(c8))));
  return mul_add(r2, x8, r1);
}

template <class T>
static inline T taylor_exp_d(const T& initialX) {
  constexpr double p2 = 1. / 2.;
  constexpr double p3 = 1. / 6.;
  constexpr double p4 = 1. / 24.;
  constexpr double p5 = 1. / 120.;
  constexpr double p6 = 1. / 720.;
  constexpr double p7 = 1. / 5040.;
  constexpr double p8 = 1. / 40320.;
  constexpr double p9 = 1. / 362880.;
  constexpr double p10 = 1. / 3628800.;
  constexpr double p11 = 1. / 39916800.;
  constexpr double p12 = 1. / 479001600.;
  constexpr double p13 = 1. / 6227020800.;

  constexpr double xMax = 708.39;
  constexpr double ln2d1 = 0.693145751953125;
  constexpr double ln2d2 = 1.42860682030941723212e-6;
  constexpr double log2e = 1.44269504088896340736;

  auto x = initialX;
  auto r = round(initialX * T(log2e));
  x = nmul_add(r, T(ln2d1), x);
  x = nmul_add(r, T(ln2d2), x);

  auto z = polynomial_13m(x, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);
  auto n2 = pow2in(r);
  z = (z + T(1.0)) * n2;

  auto inrange = abs(initialX) < T(xMax);
  inrange &= is_finite(initialX);

  if (horizontal_and(inrange)) return z;

  T vInfinite;
  vInfinite.setInfinite();
  r = select(sign_bit(initialX), T(0.), vInfinite);
  z = select(inrange, z, r);
  z = select(is_nan(initialX), initialX, z);
  return z;
}

template <class T>
static inline T taylor_exp_f(const T& initialX) {
  constexpr float p0 = 1.f / 2.f;
  constexpr float p1 = 1.f / 6.f;
  constexpr float p2 = 1.f / 24.f;
  constexpr float p3 = 1.f / 120.f;
  constexpr float p4 = 1.f / 720.f;
  constexpr float p5 = 1.f / 5040.f;

  constexpr float xMax = 87.3f;
  constexpr float ln2d1 = 0.693359375f;
  constexpr float ln2d2 = -2.12194440e-4f;
  constexpr float log2e = 1.44269504088896340736f;

  auto x = initialX;
  auto r = round(initialX * T(log2e));
  x = nmul_add(r, T(ln2d1), x);
  x = nmul_add(r, T(ln2d2), x);

  auto x2 = x * x;
  auto z = polynomial_5(x, p0, p1, p2, p3, p4, p5);
  z = mul_add(z, x2, x);

  auto n2 = pow2in(r);
  z = (z + T(1.0)) * n2;

  auto inrange = abs(initialX) < T(xMax);
  inrange &= is_finite(initialX);

  if (horizontal_and(inrange)) return z;

  T vInfinite;
  vInfinite.setInfinite();
  r = select(sign_bit(initialX), T(0.), vInfinite);
  z = select(inrange, z, r);
  z = select(is_nan(initialX), initialX, z);
  return z;
}

template <class T>
static inline T taylor_log_d(const T& initialX) {
  constexpr double ln2d1 = 0.693359375;
  constexpr double ln2d2 = -2.121944400546905827679e-4;
  constexpr double p0log = 7.70838733755885391666e0;
  constexpr double p1log = 1.79368678507819816313e1;
  constexpr double p2log = 1.44989225341610930846e1;
  constexpr double p3log = 4.70579119878881725854e0;
  constexpr double p4log = 4.97494994976747001425e-1;
  constexpr double p5log = 1.01875663804580931796e-4;
  constexpr double q0log = 2.31251620126765340583e1;
  constexpr double q1log = 7.11544750618563894466e1;
  constexpr double q2log = 8.29875266912776603211e1;
  constexpr double q3log = 4.52279145837532221105e1;
  constexpr double q4log = 1.12873587189167450590e1;
  constexpr double sqrt2 = 1.41421356237309504880;
  constexpr double smallestNormal = 2.2250738585072014e-308;

  auto x = fraction_2(initialX);
  auto fe = exponent_f(initialX);

  auto blend = x > T(sqrt2 * 0.5);
  x = x + ((!blend) & x);
  fe = fe + (blend & T(1.));
  x -= T(1.0);

  auto px = polynomial_5(x, p0log, p1log, p2log, p3log, p4log, p5log);
  auto x2 = x * x;
  px *= x * x2;
  auto qx = polynomial_5n(x, q0log, q1log, q2log, q3log, q4log);
  auto res = px / qx;

  res = mul_add(fe, T(ln2d2), res);
  res += nmul_add(x2, T(0.5), x);
  res = mul_add(fe, T(ln2d1), res);

  auto overflow = !is_finite(initialX);
  auto underflow = initialX < T(smallestNormal);

  if (!horizontal_or(overflow | underflow)) return res;
  T nan;
  nan.setNaN(0x101);
  T nInf;
  nInf.setNegativeInfinite();
  res = select(underflow, nan, res);
  res = select((initialX == T(0.)) | is_subnormal(initialX), nInf, res);
  res = select(overflow, initialX, res);
  res = select(is_inf(initialX) & sign_bit(initialX), nan, res);
  return res;
}

template <class T>
static inline T taylor_log_f(const T& initialX) {
  constexpr float ln2d1 = 0.693359375f;
  constexpr float ln2d2 = -2.12194440e-4f;
  constexpr float p0log = 3.3333331174e-1f;
  constexpr float p1log = -2.4999993993e-1f;
  constexpr float p2log = 2.0000714765e-1f;
  constexpr float p3log = -1.6668057665e-1f;
  constexpr float p4log = 1.4249322787e-1f;
  constexpr float p5log = -1.2420140846e-1f;
  constexpr float p6log = 1.1676998740e-1f;
  constexpr float p7log = -1.1514610310e-1f;
  constexpr float p8log = 7.0376836292e-2f;
  constexpr float sqrt2 = 1.41421356237309504880f;
  constexpr float smallestNormal = 1.17549435E-38f;

  auto x = fraction_2(initialX);
  auto fe = exponent_f(initialX);

  auto blend = x > T(sqrt2 * 0.5f);
  x = x + ((!blend) & x);
  fe = fe + (blend & T(1.));
  x -= T(1.0f);

  auto x2 = x * x;
  auto res = polynomial_8(x, p0log, p1log, p2log, p3log, p4log, p5log, p6log, p7log, p8log);
  res *= x2 * x;

  res = mul_add(fe, T(ln2d2), res);
  res += nmul_add(x2, T(0.5f), x);
  res = mul_add(fe, T(ln2d1), res);

  auto overflow = !is_finite(initialX);
  auto underflow = initialX < T(smallestNormal);

  if (!horizontal_or(overflow | underflow)) return res;
  T nan;
  nan.setNaN(0x101);
  T nInf;
  nInf.setNegativeInfinite();
  res = select(underflow, nan, res);
  res = select((initialX == T(0.f)) | is_subnormal(initialX), nInf, res);
  res = select(overflow, initialX, res);
  res = select(is_inf(initialX) & sign_bit(initialX), nan, res);
  return res;
}

template <class T>
static inline T taylor_tanh_d(const T& initialX) {
  constexpr double p0 = -1.61468768441708447952e3;
  constexpr double p1 = -9.92877231001918586564e1;
  constexpr double p2 = -9.64399179425052238628e-1;
  constexpr double q0 = 4.84406305325125486048e3;
  constexpr double q1 = 2.23548839060100448583e3;
  constexpr double q2 = 1.12811678491632931402e2;
  constexpr double q3 = 1.0;

  auto x = abs(initialX);
  auto xSmall = x <= T(0.625);

  T y1, y2;
  if (horizontal_or(xSmall)) {
    auto x2 = x * x;
    y1 = polynomial_2(x2, p0, p1, p2) / polynomial_3(x2, q0, q1, q2, q3);
    y1 = mul_add(y1, x2 * x, x);
  }
  if (!horizontal_and(xSmall)) {
    y2 = taylor_exp_d(x + x);
    y2 = T(1.0) - T(2.0) / (y2 + T(1.0));
  }
  auto xBig = x > T(350.);
  y1 = select(xSmall, y1, y2);
  y1 = select(xBig, T(1.0), y1);
  y1 = sign_combine(y1, initialX);
  return y1;
}

template <class T>
static inline T taylor_tanh_f(const T& initialX) {
  constexpr float r0 = -3.33332819422e-1f;
  constexpr float r1 = 1.33314422036E-1f;
  constexpr float r2 = -5.37397155531E-2f;
  constexpr float r3 = 2.06390887954E-2f;
  constexpr float r4 = -5.70498872745E-3f;

  auto x = abs(initialX);
  auto xSmall = x <= T(0.625f);

  T y1, y2;
  if (horizontal_or(xSmall)) {
    auto x2 = x * x;
    y1 = polynomial_4(x2, r0, r1, r2, r3, r4);
    y1 = mul_add(y1, x2 * x, x);
  }
  if (!horizontal_and(xSmall)) {
    y2 = taylor_exp_f(x + x);
    y2 = T(1.0f) - T(2.0f) / (y2 + T(1.0f));
  }
  auto xBig = x > T(44.4f);
  y1 = select(xSmall, y1, y2);
  y1 = select(xBig, T(1.0f), y1);
  y1 = sign_combine(y1, initialX);
  return y1;
}

}  // namespace tfcc
