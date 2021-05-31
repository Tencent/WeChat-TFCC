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

#include <vectorclass.h>

extern "C" {
extern __m512 __svml_expf16(__m512);
extern __m512d __svml_exp8(__m512d);
extern __m512 __svml_logf16(__m512);
extern __m512d __svml_log8(__m512d);
extern __m512 __svml_tanhf16(__m512);
extern __m512d __svml_tanh8(__m512d);
}

namespace VCL_NAMESPACE {

static inline Vec16f exp(Vec16f const& x) { return __svml_expf16(x); }  // exponential function
static inline Vec8d exp(Vec8d const& x) { return __svml_exp8(x); }      // exponential function

static inline Vec16f log(Vec16f const& x) { return __svml_logf16(x); }  // natural logarithm
static inline Vec8d log(Vec8d const& x) { return __svml_log8(x); }      // natural logarithm

static inline Vec16f tanh(Vec16f const& x) { return __svml_tanhf16(x); }  // hyperbolic tangent
static inline Vec8d tanh(Vec8d const& x) { return __svml_tanh8(x); }      // hyperbolic tangent

}  // namespace VCL_NAMESPACE
