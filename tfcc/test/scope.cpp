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
#include "fakeloader.h"

class ScopeTest : public testing::Test {
  static Environment _env;

 protected:
  static void SetUpTestCase() { _env.init(); }

  static void TearDownTestCase() { _env.release(); }
};

Environment ScopeTest::_env;

TEST_F(ScopeTest, npzdataloader) {
  tfcc::DataLoader::setGlobalDefault(
      new tfcc::NPZDataLoader(Environment::getDefaultTestDataPath() + "/npz_testfile.npz"));
  {
    auto _scopeG = tfcc::Scope::scope("model");
    auto& a = tfcc::Constant<float>::getConstant("a:0");
    tfcc::Variable<float> expect({2, 3});
    tfcc::data::set(expect, {1, 2, 3, 4, 5, 6});
    ASSERT_TRUE(tfcc::is_similar(a, expect));
  }
  {
    auto _scopeG1 = tfcc::Scope::scope("model");
    auto _scopeG2 = tfcc::Scope::scope("test");
    auto& b = tfcc::Constant<double>::getConstant("b:0");
    tfcc::Variable<double> expect({2, 4});
    tfcc::data::set(expect, {1, 2, 3, 4, 5, 6, 7, 8});
    ASSERT_TRUE(tfcc::is_similar(b, expect));
  }
  {
    auto _scopeG1 = tfcc::Scope::scope("model");
    auto _scopeG2 = tfcc::Scope::scope("test");
    auto& c = tfcc::Constant<int32_t>::getConstant("c:0");
    tfcc::Variable<int32_t> expect({2, 5});
    tfcc::data::set(expect, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    ASSERT_TRUE(tfcc::is_similar(c, expect));
  }
  {
    auto& d = tfcc::Constant<uint16_t>::getConstant("d:0");
    tfcc::Variable<uint16_t> expect({2, 1});
    tfcc::data::set(expect, {1, 2});
    ASSERT_TRUE(tfcc::is_similar(d, expect));
  }
}

TEST_F(ScopeTest, multidataloader) {
  tfcc::MultiDataLoader loader;
  tfcc::DataLoader::setGlobalDefault(&loader);
  FakeLoader fl1, fl2;
  fl1.setData("test", {1}, {1.0f});
  fl2.setData("test", {1}, {2.0f});

  loader.addLoader("f1", fl1);
  loader.addLoader("f1", fl2);
  {
    auto scopeG = tfcc::Scope::scope("f1");
    auto& d1 = tfcc::Constant<float>::getConstant("test");
    ASSERT_EQ(tfcc::data::get(d1)[0], 1.0f);
  }
  tfcc::Scope::getRootScope().removeChild("f1");

  ASSERT_EQ(loader.removeLoader("f1"), &fl1);
  {
    auto scopeG = tfcc::Scope::scope("f1");
    ExceptionGuard _g;
    tfcc::Exception::setStackTraceThreadLocal(false);
    ASSERT_ANY_THROW(tfcc::Constant<float>::getConstant("test"));
  }

  loader.addLoader("f1", fl2);
  {
    auto scopeG = tfcc::Scope::scope("f1");
    auto& d2 = tfcc::Constant<float>::getConstant("test");
    ASSERT_EQ(tfcc::data::get(d2)[0], 2.0f);
  }
}
