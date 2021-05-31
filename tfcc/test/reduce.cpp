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

#include "environment.h"
#include "tfcc.h"

class ReduceTest : public testing::Test {
  static Environment _env;

 protected:
  static void SetUpTestCase() { _env.init(); }

  static void TearDownTestCase() { _env.release(); }

  static tfcc::DeviceType GetDeviceType() { return _env.getCurrentDeviceType(); }
};

Environment ReduceTest::_env;

TEST_F(ReduceTest, reduce_sum) {
  auto scope1 = tfcc::Scope::scope("reduce");
  auto scope2 = tfcc::Scope::scope("reduce_sum");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    size_t keep = tfcc::Configure<size_t>::getConfigure("keep");
    tfcc::Constant<float>& inp = tfcc::Constant<float>::getConstant("input");
    tfcc::Constant<float>& expect = tfcc::Constant<float>::getConstant("expect");

    auto result = tfcc::math::reduce_sum(inp, keep);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(ReduceTest, reduce_mean) {
  auto scope1 = tfcc::Scope::scope("reduce");
  auto scope2 = tfcc::Scope::scope("reduce_mean");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    size_t keep = tfcc::Configure<size_t>::getConfigure("keep");
    tfcc::Constant<float>& inp = tfcc::Constant<float>::getConstant("input");
    tfcc::Constant<float>& expect = tfcc::Constant<float>::getConstant("expect");

    auto result = tfcc::math::reduce_mean(inp, keep);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(ReduceTest, reduce_prod) {
  auto scope1 = tfcc::Scope::scope("reduce");
  auto scope2 = tfcc::Scope::scope("reduce_prod");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    size_t keep = tfcc::Configure<size_t>::getConfigure("keep");
    tfcc::Constant<float>& inp = tfcc::Constant<float>::getConstant("input");
    tfcc::Constant<float>& expect = tfcc::Constant<float>::getConstant("expect");

    auto result = tfcc::math::reduce_prod(inp, keep);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(ReduceTest, reduce_max) {
  auto scope1 = tfcc::Scope::scope("reduce");
  auto scope2 = tfcc::Scope::scope("reduce_max");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    size_t keep = tfcc::Configure<size_t>::getConfigure("keep");
    tfcc::Constant<float>& inp = tfcc::Constant<float>::getConstant("input");
    tfcc::Constant<float>& expect = tfcc::Constant<float>::getConstant("expect");

    auto result = tfcc::math::reduce_max(inp, keep);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(ReduceTest, reduce_min) {
  auto scope1 = tfcc::Scope::scope("reduce");
  auto scope2 = tfcc::Scope::scope("reduce_min");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    size_t keep = tfcc::Configure<size_t>::getConfigure("keep");
    tfcc::Constant<float>& inp = tfcc::Constant<float>::getConstant("input");
    tfcc::Constant<float>& expect = tfcc::Constant<float>::getConstant("expect");

    auto result = tfcc::math::reduce_min(inp, keep);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(ReduceTest, reduce_any) {
  auto scope1 = tfcc::Scope::scope("reduce");
  auto scope2 = tfcc::Scope::scope("reduce_any");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    size_t keep = tfcc::Configure<size_t>::getConfigure("keep");
    tfcc::Constant<uint8_t>& inp = tfcc::Constant<uint8_t>::getConstant("input");
    tfcc::Constant<uint8_t>& expect = tfcc::Constant<uint8_t>::getConstant("expect");

    auto result = tfcc::math::reduce_any(inp, keep);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}
