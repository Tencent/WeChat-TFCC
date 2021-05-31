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

#include "environment.h"

class MathTest : public testing::Test {
  static Environment _env;

 protected:
  static void SetUpTestCase() { _env.init(); }

  static void TearDownTestCase() { _env.release(); }

  static tfcc::DeviceType GetDeviceType() { return _env.getCurrentDeviceType(); }
};

Environment MathTest::_env;

TEST_F(MathTest, broadcast) {
  // test a.shape == b.shape
  {
    tfcc::Variable<float> a({2, 1, 2});
    tfcc::data::set(a, {1, 2, 3, 4});

    tfcc::Variable<float> expect({2, 1, 2});
    tfcc::data::set(expect, {2, 4, 6, 8});
    auto result = tfcc::math::add(a, a);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }

  // test a.shape.size == b.shape.size && a.shape.size == 1
  {
    tfcc::Variable<float> a({2});
    tfcc::data::set(a, {1, 2});
    tfcc::Variable<float> b({1});
    tfcc::data::set(b, {3});

    tfcc::Variable<float> expect({2});
    tfcc::data::set(expect, {4, 5});

    auto result = tfcc::math::add(a, b);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }

  // test a.shape.size == b.shape.size && a.shape.size == 2
  {
    tfcc::Variable<float> a({2, 1});
    tfcc::data::set(a, {1, 2});
    tfcc::Variable<float> b({1, 2});
    tfcc::data::set(b, {3, 4});

    tfcc::Variable<float> expect({2, 2});
    tfcc::data::set(expect, {4, 5, 5, 6});

    auto result = tfcc::math::add(a, b);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }

  // test a.shape.size == b.shape.size && a.shape.size == 3
  {
    tfcc::Variable<float> a({2, 1, 2});
    tfcc::data::set(a, {1, 2, 3, 4});
    tfcc::Variable<float> b({1, 2, 1});
    tfcc::data::set(b, {5, 6});

    tfcc::Variable<float> expect({2, 2, 2});
    tfcc::data::set(expect, {6, 7, 7, 8, 8, 9, 9, 10});

    auto result = tfcc::math::add(a, b);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }

  // test a.shape.size == b.shape.size && a.shape.size == 4
  {
    tfcc::Variable<float> a({2, 1, 2, 1});
    tfcc::data::set(a, {1, 2, 3, 4});
    tfcc::Variable<float> b({1, 2, 1, 2});
    tfcc::data::set(b, {5, 6, 7, 8});

    tfcc::Variable<float> expect({2, 2, 2, 2});
    tfcc::data::set(expect, {6, 7, 7, 8, 8, 9, 9, 10, 8, 9, 9, 10, 10, 11, 11, 12});

    auto result = tfcc::math::add(a, b);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }

  // test a.shape.size == b.shape.size && a.shape.size == 5
  {
    tfcc::Variable<float> a({2, 1, 2, 1, 2});
    tfcc::data::set(a, {1, 2, 3, 4, 5, 6, 7, 8});
    tfcc::Variable<float> b({1, 2, 1, 2, 1});
    tfcc::data::set(b, {5, 6, 7, 8});

    tfcc::Variable<float> expect({2, 2, 2, 2, 2});
    tfcc::data::set(expect, {6,  7,  7,  8,  8,  9,  9,  10, 8,  9,  9,  10, 10, 11, 11, 12,
                             10, 11, 11, 12, 12, 13, 13, 14, 12, 13, 13, 14, 14, 15, 15, 16});

    auto result = tfcc::math::add(a, b);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }

  // test a.shape.size != b.shape.size
  {
    tfcc::Variable<float> a({2, 2, 1});
    tfcc::data::set(a, {1, 2, 3, 4});
    tfcc::Variable<float> b({2});
    tfcc::data::set(b, {5, 6});

    tfcc::Variable<float> expect({2, 2, 2});
    tfcc::data::set(expect, {6, 7, 7, 8, 8, 9, 9, 10});

    auto result = tfcc::math::add(a, b);
    auto result2 = tfcc::math::add(b, a);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
    ASSERT_TRUE(tfcc::is_similar(result2, expect));
  }
}

TEST_F(MathTest, broadcast_fail) {
  ExceptionGuard _g;
  tfcc::Exception::setStackTraceThreadLocal(false);
  // test [2, 2, 1] can not broadcast to [3, 2, 1]
  {
    tfcc::Variable<float> a({2, 2, 1});
    tfcc::Variable<float> b({3, 2, 1});

    ASSERT_THROW(tfcc::math::add(a, b), tfcc::InvalidArgumentError);
  }

  // test [2, 2, 1] can not broadcast to [3, 2]
  {
    tfcc::Variable<float> a({2, 2, 1});
    tfcc::Variable<float> b({3, 2});
    ASSERT_THROW(tfcc::math::add(a, b), tfcc::InvalidArgumentError);
  }
}

TEST_F(MathTest, add) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("add");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& b = tfcc::Constant<float>::getConstant("b");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::add(a, b);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, sub) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("sub");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& b = tfcc::Constant<float>::getConstant("b");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::sub(a, b);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, mul) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("mul");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& b = tfcc::Constant<float>::getConstant("b");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::mul(a, b);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, div) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("div");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& b = tfcc::Constant<float>::getConstant("b");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::div(a, b);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, batch_add) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("batch_add");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    std::vector<tfcc::View<float>> views;
    for (unsigned i = 0; i < a.shape(0); ++i) {
      tfcc::View<float> v(a, a.shape(), i, i + 1);
      if (v.shape().size() > 1) {
        v.squeeze({0});
      }
      views.push_back(v);
    }
    std::vector<const tfcc::Tensor<float>*> tensors;
    for (auto& v : views) {
      tensors.push_back(&v);
    }
    auto result = tfcc::math::batch_add(tensors);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, batch_mul) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("batch_mul");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    std::vector<tfcc::View<float>> views;
    for (unsigned i = 0; i < a.shape(0); ++i) {
      tfcc::View<float> v(a, a.shape(), i, i + 1);
      if (v.shape().size() > 1) {
        v.squeeze({0});
      }
      views.push_back(v);
    }
    std::vector<const tfcc::Tensor<float>*> tensors;
    for (auto& v : views) {
      tensors.push_back(&v);
    }
    auto result = tfcc::math::batch_mul(tensors);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

// a * alpha + beta
TEST_F(MathTest, transform) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("transform");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    float alpha = tfcc::Configure<float>::getConfigure("alpha");
    float beta = tfcc::Configure<float>::getConfigure("beta");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::transform(a, alpha, beta);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

// a / alpha + beta
TEST_F(MathTest, transform2) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("transform2");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    float alpha = tfcc::Configure<float>::getConfigure("alpha");
    float beta = tfcc::Configure<float>::getConfigure("beta");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::transform2(a, alpha, beta);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

// alpha / a + beta
TEST_F(MathTest, transform3) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("transform3");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    float alpha = tfcc::Configure<float>::getConfigure("alpha");
    float beta = tfcc::Configure<float>::getConfigure("beta");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::transform3(a, alpha, beta);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

// beta - a * alpha
TEST_F(MathTest, transform4) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("transform4");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    float alpha = tfcc::Configure<float>::getConfigure("alpha");
    float beta = tfcc::Configure<float>::getConfigure("beta");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::transform4(a, alpha, beta);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

// beta - a / alpha
TEST_F(MathTest, transform5) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("transform5");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    float alpha = tfcc::Configure<float>::getConfigure("alpha");
    float beta = tfcc::Configure<float>::getConfigure("beta");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::transform5(a, alpha, beta);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

// beta - alpha / a
TEST_F(MathTest, transform6) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("transform6");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    float alpha = tfcc::Configure<float>::getConfigure("alpha");
    float beta = tfcc::Configure<float>::getConfigure("beta");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::transform6(a, alpha, beta);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, min) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("min");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    uint32_t type = tfcc::Configure<uint32_t>::getConfigure("type");
    ASSERT_LT(type, 3u);
    if (type == 0) {
      auto& a = tfcc::Constant<float>::getConstant("a");
      auto& b = tfcc::Constant<float>::getConstant("b");
      auto& expect = tfcc::Constant<float>::getConstant("expect");
      auto result = tfcc::math::min(a, b);
      ASSERT_TRUE(tfcc::is_similar(result, expect));
    } else if (type == 1) {
      auto& a = tfcc::Constant<float>::getConstant("a");
      float b = tfcc::Configure<float>::getConfigure("b");
      auto& expect = tfcc::Constant<float>::getConstant("expect");
      auto result = tfcc::math::min(a, b);
      ASSERT_TRUE(tfcc::is_similar(result, expect));
    } else {
      float a = tfcc::Configure<float>::getConfigure("a");
      auto& b = tfcc::Constant<float>::getConstant("b");
      auto& expect = tfcc::Constant<float>::getConstant("expect");
      auto result = tfcc::math::min(a, b);
      ASSERT_TRUE(tfcc::is_similar(result, expect));
    }
  }
}

TEST_F(MathTest, max) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("max");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    uint32_t type = tfcc::Configure<uint32_t>::getConfigure("type");
    ASSERT_LT(type, 3u);
    if (type == 0) {
      auto& a = tfcc::Constant<float>::getConstant("a");
      auto& b = tfcc::Constant<float>::getConstant("b");
      auto& expect = tfcc::Constant<float>::getConstant("expect");
      auto result = tfcc::math::max(a, b);
      ASSERT_TRUE(tfcc::is_similar(result, expect));
    } else if (type == 1) {
      auto& a = tfcc::Constant<float>::getConstant("a");
      float b = tfcc::Configure<float>::getConfigure("b");
      auto& expect = tfcc::Constant<float>::getConstant("expect");
      auto result = tfcc::math::max(a, b);
      ASSERT_TRUE(tfcc::is_similar(result, expect));
    } else {
      float a = tfcc::Configure<float>::getConfigure("a");
      auto& b = tfcc::Constant<float>::getConstant("b");
      auto& expect = tfcc::Constant<float>::getConstant("expect");
      auto result = tfcc::math::max(a, b);
      ASSERT_TRUE(tfcc::is_similar(result, expect));
    }
  }
}

TEST_F(MathTest, sigmoid) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("sigmoid");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::sigmoid(a);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, relu) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("relu");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::relu(a);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, leaky_relu) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("leaky_relu");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    float alpha = tfcc::Configure<float>::getConfigure("alpha");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::leaky_relu(a, alpha);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, softplus) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("softplus");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::softplus(a);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, log) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("log");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::log(a);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, rsqrt) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("rsqrt");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::rsqrt(a);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, tanh) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("tanh");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::tanh(a);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, sin) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("sin");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::sin(a);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, cos) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("cos");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::cos(a);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, pow) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("pow");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    float exponent = tfcc::Configure<float>::getConfigure("exponent");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::pow(a, exponent);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, pow_v2) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("pow_v2");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& exponent = tfcc::Constant<float>::getConstant("exponent");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::pow(a, exponent);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, pow_v3) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("pow_v3");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    float a = tfcc::Configure<float>::getConfigure("a");
    auto& exponent = tfcc::Constant<float>::getConstant("exponent");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::pow(a, exponent);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, softmax) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("softmax");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    unsigned axis = tfcc::Configure<unsigned>::getConfigure("axis");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::softmax(a, axis);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, top_k) {
  if (MathTest::GetDeviceType() == tfcc::DeviceType::CUDA) {
    return;
  }
  tfcc::Variable<float> a({4, 4});
  tfcc::data::set(
      a, {1.f, 2.f, 3.f, 4.f, 4.f, 3.f, 2.f, 1.f, 1.f, 4.f, 2.f, 3.f, 2.f, 4.f, 3.f, 1.f});

  tfcc::Variable<float> expect({4, 2});
  tfcc::data::set(expect, {4.f, 3.f, 4.f, 3.f, 4.f, 3.f, 4.f, 3.f});

  tfcc::Variable<uint32_t> indices({4, 2});
  tfcc::data::set(indices, {3, 2, 0, 1, 1, 3, 1, 2});

  auto result = tfcc::math::top_k(a, 2);
  ASSERT_TRUE(tfcc::is_similar(expect, std::get<0>(result)));
  ASSERT_TRUE(tfcc::is_similar(indices, std::get<1>(result)));
}

TEST_F(MathTest, unsorted_segment_sum) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("unsorted_segment_sum");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    unsigned num = tfcc::Configure<unsigned>::getConfigure("num");
    auto& ids = tfcc::Constant<int>::getConstant("ids");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::unsorted_segment_sum(a, ids, num);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, gelu) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("gelu");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::gelu(a);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, gelu_accurate) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("gelu_accurate");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::gelu_accurate(a);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, erf) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("erf");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::erf(a);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, abs) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("abs");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::abs(a);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, asin) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("asin");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::asin(a);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, asinh) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("asinh");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::asinh(a);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, acos) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("acos");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::acos(a);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, acosh) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("acosh");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::acosh(a);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, atan) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("atan");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::atan(a);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, atanh) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("atanh");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::atanh(a);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, sign) {
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("sign");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::math::sign(a);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(MathTest, argmax) {
  if (MathTest::GetDeviceType() == tfcc::DeviceType::CUDA) {
    return;
  }
  auto scope1 = tfcc::Scope::scope("math");
  auto scope2 = tfcc::Scope::scope("argmax");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    unsigned axis = tfcc::Configure<unsigned>::getConfigure("axis");
    auto& expect = tfcc::Constant<int64_t>::getConstant("expect");
    auto result = tfcc::math::argmax(a, axis);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}