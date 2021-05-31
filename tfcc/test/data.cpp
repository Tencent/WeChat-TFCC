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

class DataTest : public testing::Test {
  static Environment _env;

 protected:
  static void SetUpTestCase() { _env.init(); }

  static void TearDownTestCase() { _env.release(); }
};

Environment DataTest::_env;

TEST_F(DataTest, hostdevicetransfer) {
  tfcc::Variable<float> a({2});
  tfcc::data::set(a, {1, 2});

  auto vec = tfcc::data::get(a);
  ASSERT_EQ(vec.size(), 2u);
  ASSERT_EQ(vec[0], 1u);
  ASSERT_EQ(vec[1], 2u);
}

TEST_F(DataTest, copy) {
  tfcc::Variable<float> a({2});
  tfcc::data::set(a, {1, 2});

  auto result = tfcc::data::copy(a);
  ASSERT_TRUE(tfcc::is_similar(a, result));
}

TEST_F(DataTest, zeros) {
  tfcc::Variable<float> a({1024});
  a = a + 1.f;
  tfcc::data::zeros(a);
  auto datas = tfcc::data::get(a);
  ASSERT_EQ(datas.size(), a.size());
  for (size_t i = 0; i < datas.size(); ++i) {
    ASSERT_EQ(datas[i], 0.f);
  }
}

TEST_F(DataTest, ones) {
  tfcc::Variable<float> a({1024});
  a = a + 5.f;
  tfcc::data::ones(a);
  auto datas = tfcc::data::get(a);
  ASSERT_EQ(datas.size(), a.size());
  for (size_t i = 0; i < datas.size(); ++i) {
    ASSERT_EQ(datas[i], 1.f);
  }
}
