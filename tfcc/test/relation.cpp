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

class RelationTest : public testing::Test {
  static Environment _env;

 protected:
  static void SetUpTestCase() { _env.init(); }

  static void TearDownTestCase() { _env.release(); }

  static tfcc::DeviceType GetDeviceType() { return _env.getCurrentDeviceType(); }
};

Environment RelationTest::_env;

TEST_F(RelationTest, equal) {
  auto scope1 = tfcc::Scope::scope("relation");
  auto scope2 = tfcc::Scope::scope("equal");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    {
      auto& a1 = tfcc::Constant<float>::getConstant("a1");
      auto& b1 = tfcc::Constant<float>::getConstant("b1");
      auto& expect1 = tfcc::Constant<uint8_t>::getConstant("expect1");
      auto result1 = tfcc::relation::equal(a1, b1);
      ASSERT_TRUE(tfcc::is_similar(result1, expect1));

      auto a2 = tfcc::Configure<float>::getConfigure("a2");
      auto& b2 = tfcc::Constant<float>::getConstant("b2");
      auto& expect2 = tfcc::Constant<uint8_t>::getConstant("expect2");
      auto result2 = tfcc::relation::equal(a2, b2);
      ASSERT_TRUE(tfcc::is_similar(result2, expect2));

      auto& a3 = tfcc::Constant<float>::getConstant("a3");
      auto b3 = tfcc::Configure<float>::getConfigure("b3");
      auto& expect3 = tfcc::Constant<uint8_t>::getConstant("expect3");
      auto result3 = tfcc::relation::equal(a3, b3);
      ASSERT_TRUE(tfcc::is_similar(result3, expect3));
    }
    {
      auto& a1 = tfcc::Constant<float>::getConstant("a1");
      auto& b1 = tfcc::Constant<float>::getConstant("b1");
      auto& expect1 = tfcc::Constant<uint8_t>::getConstant("expect1");
      auto result1 = a1 == b1;
      ASSERT_TRUE(tfcc::is_similar(result1, expect1));

      auto a2 = tfcc::Configure<float>::getConfigure("a2");
      auto& b2 = tfcc::Constant<float>::getConstant("b2");
      auto& expect2 = tfcc::Constant<uint8_t>::getConstant("expect2");
      auto result2 = a2 == b2;
      ASSERT_TRUE(tfcc::is_similar(result2, expect2));

      auto& a3 = tfcc::Constant<float>::getConstant("a3");
      auto b3 = tfcc::Configure<float>::getConfigure("b3");
      auto& expect3 = tfcc::Constant<uint8_t>::getConstant("expect3");
      auto result3 = a3 == b3;
      ASSERT_TRUE(tfcc::is_similar(result3, expect3));
    }
  }
}

TEST_F(RelationTest, unequal) {
  auto scope1 = tfcc::Scope::scope("relation");
  auto scope2 = tfcc::Scope::scope("unequal");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    {
      auto& a1 = tfcc::Constant<float>::getConstant("a1");
      auto& b1 = tfcc::Constant<float>::getConstant("b1");
      auto& expect1 = tfcc::Constant<uint8_t>::getConstant("expect1");
      auto result1 = tfcc::relation::unequal(a1, b1);
      ASSERT_TRUE(tfcc::is_similar(result1, expect1));

      auto a2 = tfcc::Configure<float>::getConfigure("a2");
      auto& b2 = tfcc::Constant<float>::getConstant("b2");
      auto& expect2 = tfcc::Constant<uint8_t>::getConstant("expect2");
      auto result2 = tfcc::relation::unequal(a2, b2);
      ASSERT_TRUE(tfcc::is_similar(result2, expect2));

      auto& a3 = tfcc::Constant<float>::getConstant("a3");
      auto b3 = tfcc::Configure<float>::getConfigure("b3");
      auto& expect3 = tfcc::Constant<uint8_t>::getConstant("expect3");
      auto result3 = tfcc::relation::unequal(a3, b3);
      ASSERT_TRUE(tfcc::is_similar(result3, expect3));
    }
    {
      auto& a1 = tfcc::Constant<float>::getConstant("a1");
      auto& b1 = tfcc::Constant<float>::getConstant("b1");
      auto& expect1 = tfcc::Constant<uint8_t>::getConstant("expect1");
      auto result1 = a1 != b1;
      ASSERT_TRUE(tfcc::is_similar(result1, expect1));

      auto a2 = tfcc::Configure<float>::getConfigure("a2");
      auto& b2 = tfcc::Constant<float>::getConstant("b2");
      auto& expect2 = tfcc::Constant<uint8_t>::getConstant("expect2");
      auto result2 = a2 != b2;
      ASSERT_TRUE(tfcc::is_similar(result2, expect2));

      auto& a3 = tfcc::Constant<float>::getConstant("a3");
      auto b3 = tfcc::Configure<float>::getConfigure("b3");
      auto& expect3 = tfcc::Constant<uint8_t>::getConstant("expect3");
      auto result3 = a3 != b3;
      ASSERT_TRUE(tfcc::is_similar(result3, expect3));
    }
  }
}

TEST_F(RelationTest, greater) {
  auto scope1 = tfcc::Scope::scope("relation");
  auto scope2 = tfcc::Scope::scope("greater");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    {
      auto& a1 = tfcc::Constant<float>::getConstant("a1");
      auto& b1 = tfcc::Constant<float>::getConstant("b1");
      auto& expect1 = tfcc::Constant<uint8_t>::getConstant("expect1");
      auto result1 = tfcc::relation::greater(a1, b1);
      ASSERT_TRUE(tfcc::is_similar(result1, expect1));

      auto a2 = tfcc::Configure<float>::getConfigure("a2");
      auto& b2 = tfcc::Constant<float>::getConstant("b2");
      auto& expect2 = tfcc::Constant<uint8_t>::getConstant("expect2");
      auto result2 = tfcc::relation::greater(a2, b2);
      ASSERT_TRUE(tfcc::is_similar(result2, expect2));

      auto& a3 = tfcc::Constant<float>::getConstant("a3");
      auto b3 = tfcc::Configure<float>::getConfigure("b3");
      auto& expect3 = tfcc::Constant<uint8_t>::getConstant("expect3");
      auto result3 = tfcc::relation::greater(a3, b3);
      ASSERT_TRUE(tfcc::is_similar(result3, expect3));
    }
    {
      auto& a1 = tfcc::Constant<float>::getConstant("a1");
      auto& b1 = tfcc::Constant<float>::getConstant("b1");
      auto& expect1 = tfcc::Constant<uint8_t>::getConstant("expect1");
      auto result1 = a1 > b1;
      ASSERT_TRUE(tfcc::is_similar(result1, expect1));

      auto a2 = tfcc::Configure<float>::getConfigure("a2");
      auto& b2 = tfcc::Constant<float>::getConstant("b2");
      auto& expect2 = tfcc::Constant<uint8_t>::getConstant("expect2");
      auto result2 = a2 > b2;
      ASSERT_TRUE(tfcc::is_similar(result2, expect2));

      auto& a3 = tfcc::Constant<float>::getConstant("a3");
      auto b3 = tfcc::Configure<float>::getConfigure("b3");
      auto& expect3 = tfcc::Constant<uint8_t>::getConstant("expect3");
      auto result3 = a3 > b3;
      ASSERT_TRUE(tfcc::is_similar(result3, expect3));
    }
  }
}

TEST_F(RelationTest, greater_equal) {
  auto scope1 = tfcc::Scope::scope("relation");
  auto scope2 = tfcc::Scope::scope("greater_equal");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    {
      auto& a1 = tfcc::Constant<float>::getConstant("a1");
      auto& b1 = tfcc::Constant<float>::getConstant("b1");
      auto& expect1 = tfcc::Constant<uint8_t>::getConstant("expect1");
      auto result1 = tfcc::relation::greater_equal(a1, b1);
      ASSERT_TRUE(tfcc::is_similar(result1, expect1));

      auto a2 = tfcc::Configure<float>::getConfigure("a2");
      auto& b2 = tfcc::Constant<float>::getConstant("b2");
      auto& expect2 = tfcc::Constant<uint8_t>::getConstant("expect2");
      auto result2 = tfcc::relation::greater_equal(a2, b2);
      ASSERT_TRUE(tfcc::is_similar(result2, expect2));

      auto& a3 = tfcc::Constant<float>::getConstant("a3");
      auto b3 = tfcc::Configure<float>::getConfigure("b3");
      auto& expect3 = tfcc::Constant<uint8_t>::getConstant("expect3");
      auto result3 = tfcc::relation::greater_equal(a3, b3);
      ASSERT_TRUE(tfcc::is_similar(result3, expect3));
    }
    {
      auto& a1 = tfcc::Constant<float>::getConstant("a1");
      auto& b1 = tfcc::Constant<float>::getConstant("b1");
      auto& expect1 = tfcc::Constant<uint8_t>::getConstant("expect1");
      auto result1 = a1 >= b1;
      ASSERT_TRUE(tfcc::is_similar(result1, expect1));

      auto a2 = tfcc::Configure<float>::getConfigure("a2");
      auto& b2 = tfcc::Constant<float>::getConstant("b2");
      auto& expect2 = tfcc::Constant<uint8_t>::getConstant("expect2");
      auto result2 = a2 >= b2;
      ASSERT_TRUE(tfcc::is_similar(result2, expect2));

      auto& a3 = tfcc::Constant<float>::getConstant("a3");
      auto b3 = tfcc::Configure<float>::getConfigure("b3");
      auto& expect3 = tfcc::Constant<uint8_t>::getConstant("expect3");
      auto result3 = a3 >= b3;
      ASSERT_TRUE(tfcc::is_similar(result3, expect3));
    }
  }
}

TEST_F(RelationTest, less) {
  auto scope1 = tfcc::Scope::scope("relation");
  auto scope2 = tfcc::Scope::scope("less");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    {
      auto& a1 = tfcc::Constant<float>::getConstant("a1");
      auto& b1 = tfcc::Constant<float>::getConstant("b1");
      auto& expect1 = tfcc::Constant<uint8_t>::getConstant("expect1");
      auto result1 = tfcc::relation::less(a1, b1);
      ASSERT_TRUE(tfcc::is_similar(result1, expect1));

      auto a2 = tfcc::Configure<float>::getConfigure("a2");
      auto& b2 = tfcc::Constant<float>::getConstant("b2");
      auto& expect2 = tfcc::Constant<uint8_t>::getConstant("expect2");
      auto result2 = tfcc::relation::less(a2, b2);
      ASSERT_TRUE(tfcc::is_similar(result2, expect2));

      auto& a3 = tfcc::Constant<float>::getConstant("a3");
      auto b3 = tfcc::Configure<float>::getConfigure("b3");
      auto& expect3 = tfcc::Constant<uint8_t>::getConstant("expect3");
      auto result3 = tfcc::relation::less(a3, b3);
      ASSERT_TRUE(tfcc::is_similar(result3, expect3));
    }
    {
      auto& a1 = tfcc::Constant<float>::getConstant("a1");
      auto& b1 = tfcc::Constant<float>::getConstant("b1");
      auto& expect1 = tfcc::Constant<uint8_t>::getConstant("expect1");
      auto result1 = a1 < b1;
      ASSERT_TRUE(tfcc::is_similar(result1, expect1));

      auto a2 = tfcc::Configure<float>::getConfigure("a2");
      auto& b2 = tfcc::Constant<float>::getConstant("b2");
      auto& expect2 = tfcc::Constant<uint8_t>::getConstant("expect2");
      auto result2 = a2 < b2;
      ASSERT_TRUE(tfcc::is_similar(result2, expect2));

      auto& a3 = tfcc::Constant<float>::getConstant("a3");
      auto b3 = tfcc::Configure<float>::getConfigure("b3");
      auto& expect3 = tfcc::Constant<uint8_t>::getConstant("expect3");
      auto result3 = a3 < b3;
      ASSERT_TRUE(tfcc::is_similar(result3, expect3));
    }
  }
}

TEST_F(RelationTest, less_equal) {
  auto scope1 = tfcc::Scope::scope("relation");
  auto scope2 = tfcc::Scope::scope("less_equal");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    {
      auto& a1 = tfcc::Constant<float>::getConstant("a1");
      auto& b1 = tfcc::Constant<float>::getConstant("b1");
      auto& expect1 = tfcc::Constant<uint8_t>::getConstant("expect1");
      auto result1 = tfcc::relation::less_equal(a1, b1);
      ASSERT_TRUE(tfcc::is_similar(result1, expect1));

      auto a2 = tfcc::Configure<float>::getConfigure("a2");
      auto& b2 = tfcc::Constant<float>::getConstant("b2");
      auto& expect2 = tfcc::Constant<uint8_t>::getConstant("expect2");
      auto result2 = tfcc::relation::less_equal(a2, b2);
      ASSERT_TRUE(tfcc::is_similar(result2, expect2));

      auto& a3 = tfcc::Constant<float>::getConstant("a3");
      auto b3 = tfcc::Configure<float>::getConfigure("b3");
      auto& expect3 = tfcc::Constant<uint8_t>::getConstant("expect3");
      auto result3 = tfcc::relation::less_equal(a3, b3);
      ASSERT_TRUE(tfcc::is_similar(result3, expect3));
    }
    {
      auto& a1 = tfcc::Constant<float>::getConstant("a1");
      auto& b1 = tfcc::Constant<float>::getConstant("b1");
      auto& expect1 = tfcc::Constant<uint8_t>::getConstant("expect1");
      auto result1 = a1 <= b1;
      ASSERT_TRUE(tfcc::is_similar(result1, expect1));

      auto a2 = tfcc::Configure<float>::getConfigure("a2");
      auto& b2 = tfcc::Constant<float>::getConstant("b2");
      auto& expect2 = tfcc::Constant<uint8_t>::getConstant("expect2");
      auto result2 = a2 <= b2;
      ASSERT_TRUE(tfcc::is_similar(result2, expect2));

      auto& a3 = tfcc::Constant<float>::getConstant("a3");
      auto b3 = tfcc::Configure<float>::getConfigure("b3");
      auto& expect3 = tfcc::Constant<uint8_t>::getConstant("expect3");
      auto result3 = a3 <= b3;
      ASSERT_TRUE(tfcc::is_similar(result3, expect3));
    }
  }
}
