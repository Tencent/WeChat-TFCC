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

#include <immintrin.h>
#include <algorithm>
#include <iostream>

#include "xbyak/xbyak.h"

#include "tfcc.h"
#include "tfcc_mkl.h"
#include "tfcc_mklfusionopdynamicshape.h"
#include "tfcc_mklfusionopfixedshape.h"
#include "tfcc_mkloperation.h"

using namespace tfcc;
using namespace tfcc::fusionop;

void add(const float* a, const float* b, float* c, unsigned len) {
  for (unsigned i = 0; i < len; ++i) {
    c[i] = a[i] + b[i];
  }
}

void sub(const float* a, const float* b, float* c, unsigned len) {
  for (unsigned i = 0; i < len; ++i) {
    c[i] = a[i] - b[i];
  }
}

void mul(const float* a, const float* b, float* c, unsigned len) {
  for (unsigned i = 0; i < len; ++i) {
    c[i] = a[i] * b[i];
  }
}

void div(const float* a, const float* b, float* c, unsigned len) {
  for (unsigned i = 0; i < len; ++i) {
    c[i] = a[i] / b[i];
  }
}

void relu(const float* a, float* b, unsigned len) {
  for (unsigned i = 0; i < len; ++i) {
    if (a[i] > 0.0) {
      b[i] = a[i];
    } else {
      b[i] = 0.0;
    }
  }
}
/*
void addAvx256(const float* a, const float* b, float* c, unsigned len)
{
    for (unsigned i = 0; i < len; i += 8) {
        __m256 a1 = _mm256_loadu_ps(&a[i]);
        __m256 b1 = _mm256_loadu_ps(&b[i]);
        __m256 c1 = _mm256_add_ps(a1, b1);
        _mm256_storeu_ps(&c[i], c1);
    }
}

void subAvx256(const float* a, const float* b, float* c, unsigned len)
{
    for (unsigned i = 0; i < len; i += 8) {
        __m256 a1 = _mm256_loadu_ps(&a[i]);
        __m256 b1 = _mm256_loadu_ps(&b[i]);
        __m256 c1 = _mm256_sub_ps(a1, b1);
        _mm256_storeu_ps(&c[i], c1);
    }
}

void mulAvx256(const float* a, const float* b, float* c, unsigned len)
{
    for (unsigned i = 0; i < len; i += 8) {
        __m256 a1 = _mm256_loadu_ps(&a[i]);
        __m256 b1 = _mm256_loadu_ps(&b[i]);
        __m256 c1 = _mm256_mul_ps(a1, b1);
        _mm256_storeu_ps(&c[i], c1);
    }
}

void divAvx256(const float* a, const float* b, float* c, unsigned len)
{
    for (unsigned i = 0; i < len; i += 8) {
        __m256 a1 = _mm256_loadu_ps(&a[i]);
        __m256 b1 = _mm256_loadu_ps(&b[i]);
        __m256 c1 = _mm256_div_ps(a1, b1);
        _mm256_storeu_ps(&c[i], c1);
    }
}

void reluAvx256(const float* a, float* b, unsigned len)
{
    __m256 zero = _mm256_set_ps(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
    for (unsigned i = 0; i < len; i += 8) {
        __m256 a1 = _mm256_loadu_ps(&a[i]);
        __m256 b1 = _mm256_max_ps(a1, zero);
        _mm256_storeu_ps(&b[i], b1);
    }
}
*/
int main(int argc, char** argv) {
  std::cout << "benchmark batchsize batchlen\n";
  unsigned batchsize = std::strtoul(argv[1], NULL, 0);
  unsigned benchLen = batchsize * std::strtoul(argv[2], NULL, 0);
  tfcc::initialize_mkl(1, 1);
  tfcc::Shape shape({benchLen});
  tfcc::Variable<float> a1 = tfcc::random::normal<float>(shape, 10, 1);
  tfcc::Variable<float> b1 = tfcc::random::normal<float>(shape, 10, 1);
  tfcc::Variable<float> c1 = tfcc::random::normal<float>(shape, 10, 1);
  tfcc::Variable<float> d1 = tfcc::random::normal<float>(shape, 10, 1);
  tfcc::Session::getThreadDefault()->sync();

  float* a = a1.data();
  float* b = b1.data();
  float* c = c1.data();
  float* d = d1.data();

  float** inputs = new float*[4];
  inputs[0] = a;
  inputs[1] = b;
  inputs[2] = c;
  inputs[3] = d;

  float* result = new float[benchLen];
  float* result1 = nullptr;
  float* result2 = nullptr;
  float* result3 = nullptr;
  float* result4 = nullptr;

  std::vector<OperationType> opTypes;
  opTypes.push_back(OperationType::PARAM_0);
  opTypes.push_back(OperationType::PARAM_1);
  opTypes.push_back(OperationType::ADD);
  opTypes.push_back(OperationType::PARAM_2);
  opTypes.push_back(OperationType::SUB);
  opTypes.push_back(OperationType::PARAM_3);
  opTypes.push_back(OperationType::MUL);
  opTypes.push_back(OperationType::PARAM_3);
  opTypes.push_back(OperationType::DIV);
  opTypes.push_back(OperationType::RELU);

  std::vector<unsigned> resultShape({benchLen});
  std::vector<std::vector<bool>> broadcastMarks{{false}, {false}, {false}, {false}};

  tfcc::Coster coster;
  FusionOpFixedShape opv1(opTypes, resultShape, broadcastMarks);
  opv1.process(inputs, result);
  std::cout << "v1 first cost: " << coster.lap().milliseconds() << std::endl;

  coster.reset();
  opv1.process(inputs, result);
  std::cout << "v1 second cost: " << coster.lap().milliseconds() << std::endl;

  coster.reset();
  FusionOpDynamicShape opv2(opTypes, broadcastMarks);
  opv2.process(inputs, result, resultShape);
  std::cout << "v2 first cost: " << coster.lap().milliseconds() << std::endl;

  coster.reset();
  opv2.process(inputs, result, resultShape);
  std::cout << "v2 second cost: " << coster.lap().milliseconds() << std::endl;

  coster.reset();
  unsigned oneLen = benchLen / batchsize;
  float** oneinputs = new float*[4];
  for (unsigned i = 0; i < batchsize; ++i) {
    unsigned step = i * oneLen;
    oneinputs[0] = a + step;
    oneinputs[1] = b + step;
    oneinputs[2] = c + step;
    oneinputs[3] = d + step;
    opv2.process(oneinputs, result + step, {oneLen});
  }
  delete[] oneinputs;
  std::cout << "v2 third cost: " << coster.lap().milliseconds() << std::endl;

  coster.reset();
  add(a, b, result, benchLen);
  sub(result, c, result, benchLen);
  mul(result, d, result, benchLen);
  div(result, d, result, benchLen);
  relu(result, result, benchLen);
  std::cout << "handwriting cost: " << coster.lap().milliseconds() << std::endl;

  coster.reset();
  result1 = new float[benchLen];
  result2 = new float[benchLen];
  result3 = new float[benchLen];
  result4 = new float[benchLen];
  add(a, b, result, benchLen);
  sub(result, c, result1, benchLen);
  mul(result1, d, result2, benchLen);
  div(result2, d, result3, benchLen);
  relu(result3, result4, benchLen);
  delete[] result1;
  delete[] result2;
  delete[] result3;
  delete[] result4;
  std::cout << "auto cost: " << coster.lap().milliseconds() << std::endl;
  /*
    coster.reset();
    result1 = new float[benchLen];
    result2 = new float[benchLen];
    result3 = new float[benchLen];
    result4 = new float[benchLen];
    addAvx256(a, b, result, benchLen);
    subAvx256(result, c, result1, benchLen);
    mulAvx256(result1, d, result2, benchLen);
    divAvx256(result2, d, result3, benchLen);
    reluAvx256(result3, result4, benchLen);
    delete []result1;
    delete []result2;
    delete []result3;
    delete []result4;
    std::cout << "avx256 cost: " << coster.lap().milliseconds() << std::endl;
*/
  coster.reset();
  tfcc::math::relu((a1 + b1 - c1) * d1 / d1);
  tfcc::Session::getThreadDefault()->sync();
  std::cout << "tfcc cost: " << coster.lap().milliseconds() << std::endl;
}
