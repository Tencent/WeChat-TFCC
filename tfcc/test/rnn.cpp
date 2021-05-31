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

class RNNTest : public testing::Test {
  static Environment _env;

 protected:
  static void SetUpTestCase() { _env.init(); }

  static void TearDownTestCase() { _env.release(); }

  static tfcc::DeviceType GetDeviceType() { return _env.getCurrentDeviceType(); }
};

Environment RNNTest::_env;

TEST_F(RNNTest, pytorch_grucell) {
  auto scope1 = tfcc::Scope::scope("rnn");
  auto scope2 = tfcc::Scope::scope("pytorchgrucell");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    unsigned units = tfcc::Configure<unsigned>::getConfigure("units");
    tfcc::rnn::PyTorchGRUCell<float> cell(units);
    tfcc::Constant<float>& inputs = tfcc::Constant<float>::getConstant("inputs");
    tfcc::Variable<float> state = tfcc::data::copy(tfcc::Constant<float>::getConstant("state"));
    tfcc::Constant<float>& expect = tfcc::Constant<float>::getConstant("expect");

    tfcc::Variable<float> result, nextState;
    std::tie(result, nextState) = cell(inputs, state, 0, {});

    ASSERT_TRUE(tfcc::is_similar(result, expect));
    ASSERT_TRUE(tfcc::is_similar(nextState, expect));
  }
}

TEST_F(RNNTest, pytorch_lstmcell) {
  auto scope1 = tfcc::Scope::scope("rnn");
  auto scope2 = tfcc::Scope::scope("pytorchlstmcell");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    unsigned units = tfcc::Configure<unsigned>::getConfigure("units");
    tfcc::rnn::PyTorchLSTMCell<float> cell(units);
    tfcc::Constant<float>& inputs = tfcc::Constant<float>::getConstant("inputs");
    tfcc::rnn::PyTorchLSTMCell<float>::State state;
    state.h = tfcc::data::copy(tfcc::Constant<float>::getConstant("h"));
    state.cs = tfcc::data::copy(tfcc::Constant<float>::getConstant("cs"));
    tfcc::Constant<float>& expect = tfcc::Constant<float>::getConstant("expect");
    tfcc::Constant<float>& expectCs = tfcc::Constant<float>::getConstant("ecs");

    tfcc::Variable<float> result;
    tfcc::rnn::PyTorchLSTMCell<float>::State nextState;
    std::tie(result, nextState) = cell(inputs, state, 0, {});

    ASSERT_TRUE(tfcc::is_similar(result, expect));
    ASSERT_TRUE(tfcc::is_similar(nextState.h, expect));
    ASSERT_TRUE(tfcc::is_similar(nextState.cs, expectCs));
  }
}

TEST_F(RNNTest, lstmcell) {
  auto scope1 = tfcc::Scope::scope("rnn");
  auto scope2 = tfcc::Scope::scope("lstmcell");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    unsigned units = tfcc::Configure<unsigned>::getConfigure("units");
    tfcc::rnn::LSTMCell<float> cell(units);
    tfcc::Constant<float>& inputs = tfcc::Constant<float>::getConstant("inputs");
    tfcc::rnn::LSTMCell<float>::State state;
    state.h = tfcc::data::copy(tfcc::Constant<float>::getConstant("h"));
    state.cs = tfcc::data::copy(tfcc::Constant<float>::getConstant("cs"));
    tfcc::Constant<float>& expect = tfcc::Constant<float>::getConstant("expect");
    tfcc::Constant<float>& expectCs = tfcc::Constant<float>::getConstant("ecs");

    tfcc::Variable<float> result;
    tfcc::rnn::LSTMCell<float>::State nextState;
    std::tie(result, nextState) = cell(inputs, state, 0, {});

    ASSERT_TRUE(tfcc::is_similar(result, expect));
    ASSERT_TRUE(tfcc::is_similar(nextState.h, expect));
    ASSERT_TRUE(tfcc::is_similar(nextState.cs, expectCs));
  }
}

TEST_F(RNNTest, grucell) {
  auto scope1 = tfcc::Scope::scope("rnn");
  auto scope2 = tfcc::Scope::scope("grucell");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    unsigned units = tfcc::Configure<unsigned>::getConfigure("units");
    tfcc::rnn::GRUCell<float> cell(units);
    tfcc::Constant<float>& inputs = tfcc::Constant<float>::getConstant("inputs");
    tfcc::Variable<float> state = tfcc::data::copy(tfcc::Constant<float>::getConstant("state"));
    tfcc::Constant<float>& expect = tfcc::Constant<float>::getConstant("expect");

    tfcc::Variable<float> result, nextState;
    std::tie(result, nextState) = cell(inputs, state, 0, {});

    ASSERT_TRUE(tfcc::is_similar(result, expect));
    ASSERT_TRUE(tfcc::is_similar(nextState, expect));
  }
}

TEST_F(RNNTest, lstm_forward) {
  auto scope1 = tfcc::Scope::scope("rnn");
  auto scope2 = tfcc::Scope::scope("lstm_forward");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& inputs = tfcc::Constant<float>::getConstant("inputs");
    auto& iKernel = tfcc::Constant<float>::getConstant("i_kernel");
    auto& hKernel = tfcc::Constant<float>::getConstant("h_kernel");
    auto& bias = tfcc::Constant<float>::getConstant("bias");
    auto& ih = tfcc::Constant<float>::getConstant("ih");
    auto& ic = tfcc::Constant<float>::getConstant("ic");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto& h = tfcc::Constant<float>::getConstant("h");
    auto& c = tfcc::Constant<float>::getConstant("c");
    tfcc::Variable<float> result;
    tfcc::Variable<float> rh;
    tfcc::Variable<float> rc;
    std::tie(result, rh, rc) = tfcc::rnn::lstm_forward(inputs, iKernel, hKernel, bias, ih, ic);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
    ASSERT_TRUE(tfcc::is_similar(rh, h));
    ASSERT_TRUE(tfcc::is_similar(rc, c));
  }
}

TEST_F(RNNTest, lstm_backward) {
  auto scope1 = tfcc::Scope::scope("rnn");
  auto scope2 = tfcc::Scope::scope("lstm_backward");
  size_t count = tfcc::Configure<uint64_t>::getConfigure("__count__");
  for (size_t i = 0; i < count; ++i) {
    auto scope3 = tfcc::Scope::scope(std::to_string(i));
    auto& inputs = tfcc::Constant<float>::getConstant("inputs");
    auto& iKernel = tfcc::Constant<float>::getConstant("i_kernel");
    auto& hKernel = tfcc::Constant<float>::getConstant("h_kernel");
    auto& bias = tfcc::Constant<float>::getConstant("bias");
    auto& ih = tfcc::Constant<float>::getConstant("ih");
    auto& ic = tfcc::Constant<float>::getConstant("ic");
    auto& expect = tfcc::Constant<float>::getConstant("expect");
    auto& h = tfcc::Constant<float>::getConstant("h");
    auto& c = tfcc::Constant<float>::getConstant("c");
    tfcc::Variable<float> result;
    tfcc::Variable<float> rh;
    tfcc::Variable<float> rc;
    std::tie(result, rh, rc) = tfcc::rnn::lstm_backward(inputs, iKernel, hKernel, bias, ih, ic);
    ASSERT_TRUE(tfcc::is_similar(result, expect));
    ASSERT_TRUE(tfcc::is_similar(rh, h));
    ASSERT_TRUE(tfcc::is_similar(rc, c));
  }
}
