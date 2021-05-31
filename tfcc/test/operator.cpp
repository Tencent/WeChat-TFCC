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

class OperatorTest : public testing::Test {
  static Environment _env;

 protected:
  static void SetUpTestCase() { _env.init(); }

  static void TearDownTestCase() { _env.release(); }
};

Environment OperatorTest::_env;

TEST_F(OperatorTest, operator_add) {
  // tensor + tensor
  {
    tfcc::Variable<float> a({2, 2});
    tfcc::data::set(a, {1, 2, 3, 4});
    tfcc::Variable<float> b({1});
    tfcc::data::set(b, {1});

    tfcc::Variable<float> expect({2, 2});
    tfcc::data::set(expect, {2, 3, 4, 5});
    auto result = a + b;
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }

  // number + tensor & tensor + number
  {
    tfcc::Variable<float> a({2, 2});
    tfcc::data::set(a, {1, 2, 3, 4});

    tfcc::Variable<float> expect({2, 2});
    tfcc::data::set(expect, {2, 3, 4, 5});

    auto result1 = 1.0f + a;
    auto result2 = a + 1.0f;
    ASSERT_TRUE(tfcc::is_similar(result1, expect));
    ASSERT_TRUE(tfcc::is_similar(result2, expect));
  }
}

TEST_F(OperatorTest, operator_sub) {
  // tensor - tensor
  {
    tfcc::Variable<float> a({2, 2});
    tfcc::data::set(a, {1, 2, 3, 4});
    tfcc::Variable<float> b({1});
    tfcc::data::set(b, {1});

    tfcc::Variable<float> expect({2, 2});
    tfcc::data::set(expect, {0, 1, 2, 3});
    auto result = a - b;
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }

  // number - tensor
  {
    tfcc::Variable<float> a({2, 2});
    tfcc::data::set(a, {1, 2, 3, 4});

    tfcc::Variable<float> expect({2, 2});
    tfcc::data::set(expect, {3, 2, 1, 0});
    auto result = 4.0f - a;
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }

  // tensor - number
  {
    tfcc::Variable<float> a({2, 2});
    tfcc::data::set(a, {1, 2, 3, 4});

    tfcc::Variable<float> expect({2, 2});
    tfcc::data::set(expect, {0, 1, 2, 3});
    auto result = a - 1.0f;
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }

  // tensor - number (unsigned)
  {
    tfcc::Variable<unsigned> a({2, 2});
    tfcc::data::set(a, {1, 2, 3, 4});

    tfcc::Variable<unsigned> expect({2, 2});
    tfcc::data::set(expect, {0, 1, 2, 3});
    auto result = a - 1u;
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(OperatorTest, operator_mul) {
  // tensor * tensor
  {
    tfcc::Variable<float> a({2, 2});
    tfcc::data::set(a, {1, 2, 3, 4});
    tfcc::Variable<float> b({1});
    tfcc::data::set(b, {2});

    tfcc::Variable<float> expect({2, 2});
    tfcc::data::set(expect, {2, 4, 6, 8});
    auto result = a * b;
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }

  // number * tensor & tensor * number
  {
    tfcc::Variable<float> a({2, 2});
    tfcc::data::set(a, {1, 2, 3, 4});

    tfcc::Variable<float> expect({2, 2});
    tfcc::data::set(expect, {2, 4, 6, 8});
    auto result1 = 2.0f * a;
    auto result2 = a * 2.0f;
    ASSERT_TRUE(tfcc::is_similar(result1, expect));
    ASSERT_TRUE(tfcc::is_similar(result2, expect));
  }
}

TEST_F(OperatorTest, operator_div) {
  // tensor / tensor
  {
    tfcc::Variable<float> a({2, 2});
    tfcc::data::set(a, {1, 2, 3, 4});
    tfcc::Variable<float> b({1});
    tfcc::data::set(b, {2});

    tfcc::Variable<float> expect({2, 2});
    tfcc::data::set(expect, {0.5, 1, 1.5, 2});
    auto result = a / b;
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }

  // number / tensor
  {
    tfcc::Variable<float> a({2, 2});
    tfcc::data::set(a, {1, 2, 3, 4});

    tfcc::Variable<float> expect({2, 2});
    tfcc::data::set(expect, {12, 6, 4, 3});
    auto result = 12.0f / a;
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }

  // tensor / number
  {
    tfcc::Variable<float> a({2, 2});
    tfcc::data::set(a, {1, 2, 3, 4});

    tfcc::Variable<float> expect({2, 2});
    tfcc::data::set(expect, {0.5, 1, 1.5, 2});
    auto result = a / 2.0f;
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}
