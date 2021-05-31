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

class BlasTest : public testing::Test {
  static Environment _env;

 protected:
  static void SetUpTestCase() { _env.init(); }

  static void TearDownTestCase() { _env.release(); }
};

Environment BlasTest::_env;

TEST_F(BlasTest, matmul) {
  auto scope1 = tfcc::Scope::scope("blas");
  auto scope2 = tfcc::Scope::scope("matmul");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& b = tfcc::Constant<float>::getConstant("b");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::blas::matmul(a, b);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(BlasTest, matmul_with_blas) {
  auto scope1 = tfcc::Scope::scope("blas");
  auto scope2 = tfcc::Scope::scope("matmulwithblas");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& b = tfcc::Constant<float>::getConstant("b");
    auto& c = tfcc::Constant<float>::getConstant("c");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::blas::matmul(a, b, c);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}
