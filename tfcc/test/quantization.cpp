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

#include <gtest/gtest.h>
#include <iostream>
#include <limits>

#include "tfcc.h"

#ifdef TFCC_WITH_MKL
#  include "operations/tfcc_mklquantization.h"
#endif

#include "environment.h"

class QuantizationTest : public testing::Test {
  static Environment _env;

 protected:
  static void SetUpTestCase() { _env.init(); }

  static void TearDownTestCase() { _env.release(); }

  static tfcc::DeviceType GetDeviceType() { return _env.getCurrentDeviceType(); }
};

Environment QuantizationTest::_env;

TEST_F(QuantizationTest, quantize) {
  if (QuantizationTest::GetDeviceType() == tfcc::DeviceType::CUDA) {
    return;
  }

  auto scope1 = tfcc::Scope::scope("quantization");
  auto scope2 = tfcc::Scope::scope("quantize");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& int8E = tfcc::Constant<int8_t>::getConstant("int8_a");
    auto& uint8E = tfcc::Constant<uint8_t>::getConstant("uint8_a");
    tfcc::Variable<int8_t> int8R;
    tfcc::Variable<uint8_t> uint8R;
    std::tie(int8R, std::ignore, std::ignore) = tfcc::quantization::quantize<int8_t>(a);
    std::tie(uint8R, std::ignore, std::ignore) = tfcc::quantization::quantize<uint8_t>(a);
    ASSERT_TRUE(tfcc::is_similar(int8R, int8E));
    ASSERT_TRUE(tfcc::is_similar(uint8R, uint8E));
  }
}

TEST_F(QuantizationTest, dequantize) {
  if (QuantizationTest::GetDeviceType() == tfcc::DeviceType::CUDA) {
    return;
  }

  tfcc::Variable<int8_t> a({6});
  tfcc::data::set(a, {23, -115, -128, 127, -20, -11});
  tfcc::Variable<float> minValue({1}), maxValue({1});
  tfcc::data::set(minValue, {-2.5});
  tfcc::data::set(maxValue, {3.4});

  tfcc::Variable<float> expect({6});
  tfcc::data::set(
      expect, {9.9372554e-01, -2.1992157e+00, -2.5000000e+00, 3.4000001e+00, -1.1765957e-03,
               2.0705891e-01});
  auto result = tfcc::quantization::dequantize(a, minValue, maxValue);
  result = result + (maxValue - minValue);
  expect = expect + (maxValue - minValue);
  ASSERT_TRUE(tfcc::is_similar(result, expect));
}

TEST_F(QuantizationTest, requantize) {
  if (QuantizationTest::GetDeviceType() == tfcc::DeviceType::CUDA) {
    return;
  }

  tfcc::Variable<int8_t> a({6});
  tfcc::data::set(a, {23, -115, -128, 127, -20, -11});
  tfcc::Variable<float> minInput({1}), maxInput({1}), minOutput({1}), maxOutput({1});
  tfcc::data::set(minInput, {-2.5});
  tfcc::data::set(maxInput, {3.4});
  tfcc::data::set(minOutput, {-12.7});
  tfcc::data::set(maxOutput, {24.5});

  tfcc::Variable<uint16_t> expect({6});
  tfcc::data::set(expect, {24125, 18500, 17970, 28364, 22372, 22739});

  auto result =
      tfcc::quantization::requantize<uint16_t>(a, minInput, maxInput, minOutput, maxOutput);
  ASSERT_TRUE(tfcc::is_similar(result, expect));
}

#ifdef TFCC_WITH_MKL
static inline tfcc::Variable<float> mkl_quantized_matmul(
    const tfcc::Tensor<float>& a, const tfcc::Tensor<float>& b) {
  tfcc::Variable<uint8_t> qa;
  tfcc::Variable<int8_t> qb;
  tfcc::Variable<float> minA, maxA, minB, maxB, minC, maxC;
  std::tie(qa, minA, maxA) = tfcc::quantization::quantize<uint8_t>(a);
  std::tie(qb, minB, maxB) = tfcc::quantization::quantize<int8_t>(b);
  tfcc::Variable<int32_t> ra, rb;
  ra = tfcc::mkl::quantization::reduce_sum(qa, true);
  rb = tfcc::mkl::quantization::reduce_sum(qb, false);

  tfcc::Variable<int32_t> qc;
  std::tie(qc, minC, maxC) =
      tfcc::mkl::quantization::matmul(qa, ra, minA, maxA, qb, rb, minB, maxB);

  return tfcc::quantization::dequantize(qc, minC, maxC);
}

TEST_F(QuantizationTest, mkl_matmul) {
  if (QuantizationTest::GetDeviceType() == tfcc::DeviceType::CUDA) {
    return;
  }

  tfcc::Variable<float> v1({2, 3, 2});
  tfcc::Variable<float> v2({2, 2, 1});
  {
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
    tfcc::data::set(v1, data);
  }
  tfcc::data::set(v2, {1, 2, 1, 2});
  auto result = mkl_quantized_matmul(v1, v2);
  auto vr = tfcc::data::get(result);
  ASSERT_EQ(static_cast<int>(vr[0]), 5);
  ASSERT_EQ(static_cast<int>(vr[1]), 11);
  ASSERT_EQ(static_cast<int>(vr[2]), 17);
  ASSERT_EQ(static_cast<int>(vr[3]), 5);
  ASSERT_EQ(static_cast<int>(vr[4]), 11);
  ASSERT_EQ(static_cast<int>(vr[5]), 17);
}

TEST_F(QuantizationTest, mkl_matmul2) {
  if (QuantizationTest::GetDeviceType() == tfcc::DeviceType::CUDA) {
    return;
  }

  tfcc::Variable<float> v1({3, 2});
  tfcc::Variable<float> v2({2, 2, 1});
  {
    std::vector<float> data = {
        1, 2, 3, 4, 5, 6,
    };
    tfcc::data::set(v1, data);
  }
  tfcc::data::set(v2, {1, 2, 1, 2});
  auto result = mkl_quantized_matmul(v1, v2);
  auto vr = tfcc::data::get(result);
  ASSERT_EQ(static_cast<int>(vr[0]), 5);
  ASSERT_EQ(static_cast<int>(vr[1]), 11);
  ASSERT_EQ(static_cast<int>(vr[2]), 17);
  ASSERT_EQ(static_cast<int>(vr[3]), 5);
  ASSERT_EQ(static_cast<int>(vr[4]), 11);
  ASSERT_EQ(static_cast<int>(vr[5]), 17);
}

TEST_F(QuantizationTest, mkl_matmul3) {
  if (QuantizationTest::GetDeviceType() == tfcc::DeviceType::CUDA) {
    return;
  }

  tfcc::Variable<float> v1({2, 3, 2});
  tfcc::Variable<float> v2({2, 1});
  {
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
    tfcc::data::set(v1, data);
  }
  tfcc::data::set(v2, {1, 2});
  auto result = mkl_quantized_matmul(v1, v2);
  auto vr = tfcc::data::get(result);
  ASSERT_EQ(static_cast<int>(vr[0]), 5);
  ASSERT_EQ(static_cast<int>(vr[1]), 11);
  ASSERT_EQ(static_cast<int>(vr[2]), 17);
  ASSERT_EQ(static_cast<int>(vr[3]), 5);
  ASSERT_EQ(static_cast<int>(vr[4]), 11);
  ASSERT_EQ(static_cast<int>(vr[5]), 17);
}

TEST_F(QuantizationTest, mkl_matmul4) {
  if (QuantizationTest::GetDeviceType() == tfcc::DeviceType::CUDA) {
    return;
  }

  auto scope1 = tfcc::Scope::scope("quantization");
  auto scope2 = tfcc::Scope::scope("mklquantizationmatmul");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    tfcc::Constant<uint8_t>& a = tfcc::Constant<uint8_t>::getConstant("a");
    tfcc::Constant<int8_t>& b = tfcc::Constant<int8_t>::getConstant("b");
    tfcc::Constant<float>& aMin = tfcc::Constant<float>::getConstant("aMin");
    tfcc::Constant<float>& aMax = tfcc::Constant<float>::getConstant("aMax");
    tfcc::Constant<float>& bMin = tfcc::Constant<float>::getConstant("bMin");
    tfcc::Constant<float>& bMax = tfcc::Constant<float>::getConstant("bMax");
    tfcc::Constant<int32_t>& expect = tfcc::Constant<int32_t>::getConstant("expect");

    tfcc::Variable<int32_t> ra, rb;
    ra = tfcc::mkl::quantization::reduce_sum(a, true);
    rb = tfcc::mkl::quantization::reduce_sum(b, false);

    tfcc::Variable<int32_t> result;
    std::tie(result, std::ignore, std::ignore) =
        tfcc::mkl::quantization::matmul(a, ra, aMin, aMax, b, rb, bMin, bMax);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

#endif
