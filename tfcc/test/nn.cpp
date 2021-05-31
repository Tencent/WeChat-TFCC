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

class NNTest : public testing::Test {
  static Environment _env;

 protected:
  static void SetUpTestCase() { _env.init(); }

  static void TearDownTestCase() { _env.release(); }

  static tfcc::DeviceType GetDeviceType() { return _env.getCurrentDeviceType(); }
};

Environment NNTest::_env;

TEST_F(NNTest, conv2d) {
  tfcc::Variable<float> input({1, 3, 4, 2});
  tfcc::Variable<float> kernel({1, 2, 3, 3});
  tfcc::Variable<float> expect({1, 3, 4, 1});

  {
    std::vector<float> data = {
        1, 0.1, 2, 0.2, 3, 0.3, 4,  0.4,  5,  0.5,  6,  0.6,
        7, 0.7, 8, 0.8, 9, 0.9, 10, 0.10, 11, 0.11, 12, 0.12,
    };
    tfcc::data::set(input, data);
  }

  {
    std::vector<float> data = {
        1, 2, 3, 4, 5, 6, 7, 8, 9,

        1, 2, 3, 4, 5, 6, 7, 8, 9,
    };
    tfcc::data::set(kernel, data);
  }

  {
    std::vector<float> data = {122.1,     195.8,  238.7,     159.49998, 246,       366.68997,
                               408.36002, 261.63, 140.90001, 198.56001, 217.06999, 130.34001};
    tfcc::data::set(expect, data);
  }

  // nhwc
  {
    auto result = tfcc::nn::conv2d(input, true, kernel, 1, 1, 1, 1);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }

  // nchw
  {
    auto realInput = tfcc::base::transpose(input, {0, 3, 1, 2});
    auto realExpect = tfcc::base::transpose(expect, {0, 3, 1, 2});
    auto result = tfcc::nn::conv2d(realInput, false, kernel, 1, 1, 1, 1);
    ASSERT_TRUE(tfcc::is_similar(result, realExpect));
  }
}

TEST_F(NNTest, conv2d_with_dilation) {
  if (NNTest::GetDeviceType() == tfcc::DeviceType::CUDA) {
    return;
  }
  auto scope1 = tfcc::Scope::scope("nn");
  auto scope2 = tfcc::Scope::scope("conv");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& inputs = tfcc::Constant<float>::getConstant("inputs");
    auto& kernel = tfcc::Constant<float>::getConstant("kernel");
    unsigned padding_height = tfcc::Configure<unsigned>::getConfigure("padding_height");
    unsigned padding_width = tfcc::Configure<unsigned>::getConfigure("padding_width");
    unsigned stride_height = tfcc::Configure<unsigned>::getConfigure("stride_height");
    unsigned stride_width = tfcc::Configure<unsigned>::getConfigure("stride_width");
    unsigned dilation_height = tfcc::Configure<unsigned>::getConfigure("dilation_height");
    unsigned dilation_width = tfcc::Configure<unsigned>::getConfigure("dilation_width");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::nn::conv2d(
        inputs, false, kernel, padding_height, padding_width, stride_height, stride_width,
        dilation_height, dilation_width);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(NNTest, batch_normalization) {
  if (NNTest::GetDeviceType() == tfcc::DeviceType::CUDA) {
    return;
  }
  auto scope1 = tfcc::Scope::scope("nn");
  auto scope2 = tfcc::Scope::scope("batch_normalization");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; i++) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    auto& scale = tfcc::Constant<float>::getConstant("scale");
    auto& mean = tfcc::Constant<float>::getConstant("mean");
    auto& offset = tfcc::Constant<float>::getConstant("offset");
    auto& var = tfcc::Constant<float>::getConstant("var");
    float epsilon = tfcc::Configure<float>::getConfigure("epsilon");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::nn::batch_normalization(a, 1, scale, offset, mean, var, epsilon);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(NNTest, conv2d_same) {
  tfcc::Variable<float> input({1, 3, 4, 2});
  tfcc::Variable<float> kernel({1, 2, 2, 3});
  tfcc::Variable<float> expect({1, 3, 4, 1});

  {
    std::vector<float> data = {
        1, 0.1, 2, 0.2, 3, 0.3, 4,  0.4,  5,  0.5,  6,  0.6,
        7, 0.7, 8, 0.8, 9, 0.9, 10, 0.10, 11, 0.11, 12, 0.12,
    };
    tfcc::data::set(input, data);
  }

  {
    std::vector<float> data = {
        1, 2, 3, 4, 5, 6,

        1, 2, 3, 4, 5, 6,
    };
    tfcc::data::set(kernel, data);
  }

  {
    std::vector<float> data = {
        75.899994, 116.6,     139.7, 86.90001,  140.90001, 198.56001,
        217.06999, 130.34001, 50.1,  63.430004, 68.68,     35.350002,
    };
    tfcc::data::set(expect, data);
  }

  // nhwc
  {
    auto result = tfcc::nn::conv2d_same(input, true, kernel, 1, 1);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }

  // nchw
  {
    auto realInput = tfcc::base::transpose(input, {0, 3, 1, 2});
    auto realExpect = tfcc::base::transpose(expect, {0, 3, 1, 2});
    auto result = tfcc::nn::conv2d_same(realInput, false, kernel, 1, 1);
    ASSERT_TRUE(tfcc::is_similar(result, realExpect));
  }
}

TEST_F(NNTest, conv2d_valid) {
  tfcc::Variable<float> input({1, 3, 4, 2});
  tfcc::Variable<float> kernel({1, 2, 2, 3});
  tfcc::Variable<float> expect({1, 2, 2, 1});

  {
    std::vector<float> data = {
        1, 0.1, 2, 0.2, 3, 0.3, 4,  0.4,  5,  0.5,  6,  0.6,
        7, 0.7, 8, 0.8, 9, 0.9, 10, 0.10, 11, 0.11, 12, 0.12,
    };
    tfcc::data::set(input, data);
  }

  {
    std::vector<float> data = {
        1, 2, 3, 4, 5, 6,

        1, 2, 3, 4, 5, 6,
    };
    tfcc::data::set(kernel, data);
  }

  {
    std::vector<float> data = {
        116.6,
        139.7,
        198.56001,
        217.06999,
    };
    tfcc::data::set(expect, data);
  }

  // nhwc
  {
    auto result = tfcc::nn::conv2d_valid(input, true, kernel, 1, 1);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }

  // nchw
  {
    auto realInput = tfcc::base::transpose(input, {0, 3, 1, 2});
    auto realExpect = tfcc::base::transpose(expect, {0, 3, 1, 2});
    auto result = tfcc::nn::conv2d_valid(realInput, false, kernel, 1, 1);
    ASSERT_TRUE(tfcc::is_similar(result, realExpect));
  }
}

TEST_F(NNTest, conv2d_same_stride) {
  tfcc::Variable<float> input({1, 3, 4, 2});
  tfcc::Variable<float> kernel({1, 2, 2, 3});
  tfcc::Variable<float> expect({1, 2, 2, 1});

  {
    std::vector<float> data = {
        1, 0.1, 2, 0.2, 3, 0.3, 4,  0.4,  5,  0.5,  6,  0.6,
        7, 0.7, 8, 0.8, 9, 0.9, 10, 0.10, 11, 0.11, 12, 0.12,
    };
    tfcc::data::set(input, data);
  }

  {
    std::vector<float> data = {
        1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,
    };
    tfcc::data::set(kernel, data);
  }

  {
    std::vector<float> data = {
        116.6,
        86.90001,
        63.430004,
        35.350002,
    };
    tfcc::data::set(expect, data);
  }

  // nhwc
  {
    auto result = tfcc::nn::conv2d_same(input, true, kernel, 2, 2);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }

  // nchw
  {
    auto realInput = tfcc::base::transpose(input, {0, 3, 1, 2});
    auto realExpect = tfcc::base::transpose(expect, {0, 3, 1, 2});
    auto result = tfcc::nn::conv2d_same(realInput, false, kernel, 2, 2);
    ASSERT_TRUE(tfcc::is_similar(result, realExpect));
  }
}

TEST_F(NNTest, conv2d_backward_data) {
  tfcc::Variable<float> input({1, 2, 2, 1});
  tfcc::Variable<float> kernel({1, 2, 3, 3});
  tfcc::Variable<float> expect({1, 3, 3, 2});

  tfcc::data::set(input, {15, 0.5, 25, 0.1});

  {
    std::vector<float> data = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, -1, 2, -3, 4, -5, 6, -7, 8, -9,
    };
    tfcc::data::set(kernel, data);
  }

  {
    std::vector<float> data = {
        75,     -75, 92,  92,  2.5,  -2.5,  170,   170, 213.6,
        -213.6, 4.2, 4.2, 125, -125, 150.4, 150.4, 0.5, -0.5,
    };
    tfcc::data::set(expect, data);
  }

  // nhwc
  {
    auto result = tfcc::nn::conv2d_backward_data(input, true, kernel, 1, 1, 2, 2);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }

  // nchw
  {
    auto realInput = tfcc::base::transpose(input, {0, 3, 1, 2});
    auto realExpect = tfcc::base::transpose(expect, {0, 3, 1, 2});
    auto result = tfcc::nn::conv2d_backward_data(realInput, false, kernel, 1, 1, 2, 2);
    ASSERT_TRUE(tfcc::is_similar(result, realExpect));
  }
}

TEST_F(NNTest, one_hot) {
  {
    std::vector<unsigned> ids{3, 0, 2};
    tfcc::Variable<float> expect({3, 4});
    {
      std::vector<float> data = {0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0};
      tfcc::data::set(expect, data);
    }

    auto result = tfcc::nn::one_hot(ids, 4, 1.f, 0.f);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }

  {
    std::vector<unsigned> ids{3, 0, 2, 0, 1, 1};
    tfcc::Variable<float> expect({3, 2, 5});
    std::vector<float> data = {
        0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
    };
    tfcc::data::set(expect, data);

    auto result = tfcc::nn::one_hot(ids, {3, 2}, 5, 1.f, 0.f);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(NNTest, max_pool2d) {
  tfcc::Variable<float> input({1, 4, 4, 2});
  tfcc::Variable<float> expect({1, 3, 3, 2});

  {
    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 7.0, 6.0,
                               5.0, 4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0, 8.0, 7.0,
                               6.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    tfcc::data::set(input, data);
  }

  {
    std::vector<float> data = {
        8.0, 7.0, 6.0, 6.0, 7.0, 8.0, 8.0, 7.0, 8.0, 7.0, 8.0, 7.0, 4.0, 4.0, 8.0, 7.0, 8.0, 8.0,
    };
    tfcc::data::set(expect, data);
  }

  // nhwc
  {
    auto result = tfcc::nn::max_pool2d(input, true, 2, 2, 0, 0, 1, 1);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }

  // nchw
  {
    auto realInput = tfcc::base::transpose(input, {0, 3, 1, 2});
    auto realExpect = tfcc::base::transpose(expect, {0, 3, 1, 2});
    auto result = tfcc::nn::max_pool2d(realInput, false, 2, 2, 0, 0, 1, 1);
    ASSERT_TRUE(tfcc::is_similar(result, realExpect));
  }
}

TEST_F(NNTest, max_pool2d_valid) {
  tfcc::Variable<float> input({1, 4, 4, 2});
  tfcc::Variable<float> expect({1, 1, 1, 2});

  {
    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 7.0, 6.0,
                               5.0, 4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0, 8.0, 7.0,
                               6.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    tfcc::data::set(input, data);
  }

  tfcc::data::set(expect, {8.0, 7.0});

  // nhwc
  {
    auto result = tfcc::nn::max_pool2d_valid(input, true, 3, 3, 3, 3);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }

  // nchw
  {
    auto realInput = tfcc::base::transpose(input, {0, 3, 1, 2});
    auto realExpect = tfcc::base::transpose(expect, {0, 3, 1, 2});
    auto result = tfcc::nn::max_pool2d_valid(realInput, false, 3, 3, 3, 3);
    ASSERT_TRUE(tfcc::is_similar(result, realExpect));
  }
}

TEST_F(NNTest, max_pool2d_same) {
  tfcc::Variable<float> input({1, 4, 4, 2});
  tfcc::Variable<float> expect({1, 2, 2, 2});

  {
    std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 7.0, 6.0,
                               5.0, 4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0, 8.0, 7.0,
                               6.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    tfcc::data::set(input, data);
  }

  tfcc::data::set(expect, {8.0, 7.0, 8.0, 8.0, 8.0, 7.0, 8.0, 8.0});

  // nhwc
  {
    auto result = tfcc::nn::max_pool2d_same(input, true, 3, 3, 2, 2);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }

  // nchw
  {
    auto realInput = tfcc::base::transpose(input, {0, 3, 1, 2});
    auto realExpect = tfcc::base::transpose(expect, {0, 3, 1, 2});
    auto result = tfcc::nn::max_pool2d_same(realInput, false, 3, 3, 2, 2);
    ASSERT_TRUE(tfcc::is_similar(result, realExpect));
  }
}

TEST_F(NNTest, gather) {
  auto scope1 = tfcc::Scope::scope("nn");
  auto scope2 = tfcc::Scope::scope("gather");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    {
      auto& params = tfcc::Constant<float>::getConstant("params");
      auto& indices = tfcc::Constant<uint32_t>::getConstant("indices");
      auto& expect = tfcc::Constant<float>::getConstant("expect");
      auto result = tfcc::nn::gather(params, indices);
      ASSERT_TRUE(tfcc::is_similar(result, expect));
    }
    {
      auto& params = tfcc::Constant<float>::getConstant("params");
      auto& indices = tfcc::Constant<int32_t>::getConstant("indices");
      auto& expect = tfcc::Constant<float>::getConstant("expect");
      auto result = tfcc::nn::gather(params, indices);
      ASSERT_TRUE(tfcc::is_similar(result, expect));
    }
    {
      auto& params = tfcc::Constant<float>::getConstant("params");
      auto& indices = tfcc::Constant<uint64_t>::getConstant("indices");
      auto& expect = tfcc::Constant<float>::getConstant("expect");
      auto result = tfcc::nn::gather(params, indices);
      ASSERT_TRUE(tfcc::is_similar(result, expect));
    }
    {
      auto& params = tfcc::Constant<float>::getConstant("params");
      auto& indices = tfcc::Constant<int64_t>::getConstant("indices");
      auto& expect = tfcc::Constant<float>::getConstant("expect");
      auto result = tfcc::nn::gather(params, indices);
      ASSERT_TRUE(tfcc::is_similar(result, expect));
    }
  }
}

TEST_F(NNTest, where) {
  auto scope1 = tfcc::Scope::scope("nn");
  auto scope2 = tfcc::Scope::scope("where");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& condition = tfcc::Constant<uint8_t>::getConstant("condition");
    auto& x = tfcc::Constant<float>::getConstant("x");
    auto& y = tfcc::Constant<float>::getConstant("y");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::nn::where(condition, x, y);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(NNTest, local_response_normalization) {
  if (NNTest::GetDeviceType() == tfcc::DeviceType::CUDA) {
    return;
  }

  auto scope1 = tfcc::Scope::scope("nn");
  auto scope2 = tfcc::Scope::scope("local_response_normalization");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& a = tfcc::Constant<float>::getConstant("a");
    size_t axis = tfcc::Configure<uint64_t>::getConfigure("axis");
    float alpha = tfcc::Configure<float>::getConfigure("alpha");
    float beta = tfcc::Configure<float>::getConfigure("beta");
    float bias = tfcc::Configure<float>::getConfigure("bias");
    unsigned size = tfcc::Configure<unsigned>::getConfigure("size");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto result = tfcc::nn::local_response_normalization(a, axis, alpha, beta, bias, size);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}

TEST_F(NNTest, scatter_nd) {
  if (NNTest::GetDeviceType() == tfcc::DeviceType::CUDA) {
    return;
  }

  auto scope1 = tfcc::Scope::scope("nn");
  auto scope2 = tfcc::Scope::scope("scatter_nd");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& data = tfcc::Constant<float>::getConstant("data");
    auto& indices = tfcc::Constant<int32_t>::getConstant("indices");
    auto& updates = tfcc::Constant<float>::getConstant("updates");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    bool hasData = tfcc::Configure<int32_t>::getConfigure("has_data") != 0;
    tfcc::Variable<float> result;
    if (hasData) {
      result = tfcc::nn::scatter_nd(data, indices, updates);
    } else {
      result = tfcc::nn::scatter_nd(indices, updates, data.shape());
    }
    ASSERT_TRUE(tfcc::is_similar(result, expect));
  }
}
