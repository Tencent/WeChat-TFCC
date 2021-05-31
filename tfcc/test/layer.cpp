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

class LayerTest : public testing::Test {
  static Environment _env;

 protected:
  static void SetUpTestCase() { _env.init(); }

  static void TearDownTestCase() { _env.release(); }

  static tfcc::DeviceType GetDeviceType() { return _env.getCurrentDeviceType(); }
};

Environment LayerTest::_env;

TEST_F(LayerTest, layer_normalization) {
  auto scope1 = tfcc::Scope::scope("layer");
  auto scope2 = tfcc::Scope::scope("layer_normalization");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    unsigned axis = tfcc::Configure<unsigned>::getConfigure("axis");
    tfcc::Constant<float>& a = tfcc::Constant<float>::getConstant("a");
    tfcc::Constant<float>& expect = tfcc::Constant<float>::getConstant("expect");

    auto result = tfcc::layer::layer_normalization(a, axis, true, true);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}
