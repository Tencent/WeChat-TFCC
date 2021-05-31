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
#include <algorithm>
#include <iostream>
#include <limits>

#include "tfcc.h"

#include "environment.h"

class BaseTest : public testing::Test {
  static Environment _env;

 protected:
  static void SetUpTestCase() { _env.init(); }

  static void TearDownTestCase() { _env.release(); }

  static tfcc::DeviceType GetDeviceType() { return _env.getCurrentDeviceType(); }
};

Environment BaseTest::_env;

TEST_F(BaseTest, slice) {
  auto scope1 = tfcc::Scope::scope("base");
  auto scope2 = tfcc::Scope::scope("slice");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    tfcc::Constant<float>& a = tfcc::Constant<float>::getConstant("a");
    tfcc::Constant<float>& expect = tfcc::Constant<float>::getConstant("expect");
    unsigned axis = tfcc::Configure<unsigned>::getConfigure("axis");
    unsigned start = tfcc::Configure<unsigned>::getConfigure("start");
    unsigned end = tfcc::Configure<unsigned>::getConfigure("end");
    auto result = tfcc::base::slice(a, axis, start, end);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(BaseTest, concat) {
  auto scope1 = tfcc::Scope::scope("base");
  auto scope2 = tfcc::Scope::scope("concat");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    tfcc::Constant<float>& a = tfcc::Constant<float>::getConstant("a");
    tfcc::Constant<float>& b = tfcc::Constant<float>::getConstant("b");
    tfcc::Constant<float>& expect = tfcc::Constant<float>::getConstant("expect");
    unsigned axis = tfcc::Configure<unsigned>::getConfigure("axis");
    auto result = tfcc::base::concat({&a, &b}, axis);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(BaseTest, transpose) {
  auto scope1 = tfcc::Scope::scope("base");
  auto scope2 = tfcc::Scope::scope("transpose");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    tfcc::Constant<float>& a = tfcc::Constant<float>::getConstant("a");
    tfcc::Constant<size_t>& perm = tfcc::Constant<size_t>::getConstant("perm");
    tfcc::Constant<float>& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::base::transpose(a, tfcc::data::get(perm));
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(BaseTest, stack) {
  auto scope1 = tfcc::Scope::scope("base");
  auto scope2 = tfcc::Scope::scope("stack");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    tfcc::Constant<float>& a = tfcc::Constant<float>::getConstant("a");
    tfcc::Constant<float>& b = tfcc::Constant<float>::getConstant("b");
    tfcc::Constant<float>& expect = tfcc::Constant<float>::getConstant("expect");
    unsigned axis = tfcc::Configure<unsigned>::getConfigure("axis");
    auto result = tfcc::base::stack({&a, &b}, axis);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(BaseTest, unstack) {
  auto scope1 = tfcc::Scope::scope("base");
  auto scope2 = tfcc::Scope::scope("unstack");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    tfcc::Constant<float>& a = tfcc::Constant<float>::getConstant("a");
    unsigned axis = tfcc::Configure<unsigned>::getConfigure("axis");
    auto result = tfcc::base::unstack(a, axis);
    for (unsigned i = 0; i < result.size(); ++i) {
      tfcc::Constant<float>& expect =
          tfcc::Constant<float>::getConstant("expect_" + std::to_string(i));
      ASSERT_TRUE(tfcc::is_similar(result[i], expect));
    }
  }
}

TEST_F(BaseTest, tril) {
  if (BaseTest::GetDeviceType() == tfcc::DeviceType::CUDA) {
    return;
  }

  auto scope1 = tfcc::Scope::scope("base");
  auto scope2 = tfcc::Scope::scope("tril");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    int64_t k = tfcc::Configure<int64_t>::getConfigure("k");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::base::tril(a, k);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(BaseTest, triu) {
  if (BaseTest::GetDeviceType() == tfcc::DeviceType::CUDA) {
    return;
  }

  auto scope1 = tfcc::Scope::scope("base");
  auto scope2 = tfcc::Scope::scope("triu");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    int64_t k = tfcc::Configure<int64_t>::getConfigure("k");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::base::triu(a, k);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}
