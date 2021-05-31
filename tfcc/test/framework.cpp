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
#include <vector>

#include "tfcc.h"

#include "environment.h"

class FrameworkTest : public testing::Test {
  static Environment _env;

 protected:
  static void SetUpTestCase() { _env.init(); }

  static void TearDownTestCase() { _env.release(); }

  static tfcc::DeviceType GetDeviceType() { return _env.getCurrentDeviceType(); }
};

Environment FrameworkTest::_env;

TEST_F(FrameworkTest, reshape) {
  tfcc::Variable<float> expect({3, 4, 5});
  tfcc::Variable<float> a({60});
  tfcc::View<float> b(a);

  std::vector<tfcc::View<float>> list;
  list.push_back(b);
  list.push_back(b);
  list.push_back(b);

  a.reshape({3, 4, 5});
  b.reshape({3, 4, 5});

  ASSERT_EQ(a.shape(), expect.shape());
  ASSERT_EQ(b.shape(), expect.shape());

  ExceptionGuard _g;
  tfcc::Exception::setStackTraceThreadLocal(false);

  ASSERT_THROW(a.reshape({4, 4, 4}), tfcc::InvalidArgumentError);
  ASSERT_THROW(b.reshape({4, 4, 4}), tfcc::InvalidArgumentError);
}

TEST_F(FrameworkTest, expand_dims) {
  tfcc::Variable<float> expect({3, 1, 4, 5});
  tfcc::Variable<float> a({3, 4, 5});
  tfcc::View<float> b(a);

  a.expandDims(1);
  b.expandDims(1);
  ASSERT_EQ(a.shape(), expect.shape());
  ASSERT_EQ(b.shape(), expect.shape());

  ExceptionGuard _g;
  tfcc::Exception::setStackTraceThreadLocal(false);

  ASSERT_THROW(a.expandDims(5), tfcc::InvalidArgumentError);
  ASSERT_THROW(b.expandDims(5), tfcc::InvalidArgumentError);
}

TEST_F(FrameworkTest, squeeze) {
  {
    tfcc::Variable<float> expect({3, 4, 5});
    tfcc::Variable<float> a({3, 1, 4, 1, 5, 1});
    tfcc::View<float> b(a);

    a.squeeze();
    b.squeeze();
    ASSERT_EQ(a.shape(), expect.shape());
    ASSERT_EQ(b.shape(), expect.shape());
  }

  {
    tfcc::Variable<float> expect({3, 1, 4, 5});
    tfcc::Variable<float> a({3, 1, 4, 1, 5, 1});
    tfcc::View<float> b(a);

    a.squeeze({3, 5});
    b.squeeze({3, 5});
    ASSERT_EQ(a.shape(), expect.shape());
    ASSERT_EQ(b.shape(), expect.shape());

    ExceptionGuard _g;
    tfcc::Exception::setStackTraceThreadLocal(false);

    ASSERT_THROW(a.squeeze({4}), tfcc::InvalidArgumentError);
    ASSERT_THROW(b.squeeze({4}), tfcc::InvalidArgumentError);

    ASSERT_THROW(a.squeeze({0}), tfcc::InvalidArgumentError);
    ASSERT_THROW(b.squeeze({0}), tfcc::InvalidArgumentError);
  }
}

TEST_F(FrameworkTest, flexallocator) {
  if (dynamic_cast<tfcc::FlexAllocator*>(&tfcc::Session::getThreadDefault()->getAllocator()) ==
      nullptr) {
    return;
  }
  tfcc::FlexAllocator& allocator =
      *dynamic_cast<tfcc::FlexAllocator*>(&tfcc::Session::getThreadDefault()->getAllocator());
  // test limit
  allocator.setLimit(4096 * 8);
  {
    tfcc::Variable<float> x1({1024 - 64 / sizeof(float)});
    tfcc::Variable<float> x2({1024 - 64 / sizeof(float)});
    tfcc::Variable<float> x3({1024 - 64 / sizeof(float)});
    tfcc::Variable<float> x4({1024 - 64 / sizeof(float)});
    tfcc::Variable<float> x5({1024 - 64 / sizeof(float)});
    tfcc::Variable<float> x6({1024 - 64 / sizeof(float)});
    tfcc::Variable<float> x7({1024 - 64 / sizeof(float)});
    tfcc::Variable<float> x8({1024 - 64 / sizeof(float)});

    ExceptionGuard _g;
    tfcc::Exception::setStackTraceThreadLocal(false);
    ASSERT_THROW(tfcc::Variable<float>({64}), tfcc::ResourceExhaustedError);
  }
  allocator.releaseCache();
  ASSERT_EQ(allocator.used(), 0lu);

  allocator.setFlexLimit(4096 * 4);
  {
    tfcc::Variable<float> x;
    x = tfcc::Variable<float>({1024 - 64 / sizeof(float)});
    ASSERT_EQ(allocator.used(), 4096lu);

    x = tfcc::Variable<float>({2, 1024 - 64 / sizeof(float)});
    ASSERT_EQ(allocator.used(), 4096 * 3lu);

    x = tfcc::Variable<float>({3, 1024 - 64 / sizeof(float)});
    ASSERT_EQ(allocator.used(), 4096 * 5lu);

    x = tfcc::Variable<float>({3, 1024 - 64 / sizeof(float)});
    ASSERT_EQ(allocator.used(), 4096 * 6lu);

    tfcc::Variable<float> x2({4, 1024 - 64 / sizeof(float)});
    ASSERT_EQ(allocator.used(), 4096 * 7lu);
  }

  allocator.releaseCache();
  ASSERT_EQ(allocator.used(), 0lu);
  allocator.setFlexLimit(4096 * 4 - 1);
  {
    tfcc::Variable<float> x;
    x = tfcc::Variable<float>({1024 - 64 / sizeof(float)});
    ASSERT_EQ(allocator.used(), 4096lu);

    x = tfcc::Variable<float>({2, 1024 - 64 / sizeof(float)});
    ASSERT_EQ(allocator.used(), 4096 * 3lu);

    x = tfcc::Variable<float>();

    x = tfcc::Variable<float>({3, 1024 - 64 / sizeof(float)});
    ASSERT_EQ(allocator.used(), 4096 * 3lu);
  }
  allocator.setFlexLimit(0);
  allocator.setLimit(0);
}

TEST_F(FrameworkTest, getdevicetype) {
  ASSERT_EQ(tfcc::Device::getThreadDefault()->getDeviceType(), FrameworkTest::GetDeviceType());
}
