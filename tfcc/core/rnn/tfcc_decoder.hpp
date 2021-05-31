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

#pragma once

#include "tfcc_decoder.h"

#include "framework/tfcc_scope.h"

namespace tfcc {
namespace rnn {

template <class D>
std::tuple<std::vector<typename D::Output>, typename D::State> dynamic_decoder(
    D& decoder, size_t maxLoop, const std::string& name) {
  auto scopeG = Scope::scope(name);
  auto _tmp = decoder.initialize();
  bool& finished = std::get<0>(_tmp);
  typename D::Input& input = std::get<1>(_tmp);
  typename D::State& state = std::get<2>(_tmp);
  std::vector<typename D::Output> outputs;
  for (size_t i = 0; i < maxLoop && !finished; ++i) {
    auto _rs = decoder.step(i, input, state, session);
    typename D::Output& output = std::get<0>(_rs);
    state = std::move(std::get<1>(_rs));
    input = std::move(std::get<2>(_rs));
    finished = std::get<3>(_rs);
    outputs.emplace_back(std::move(output));
  }
  return std::make_pair(std::move(outputs), std::move(state));
}

}  // namespace rnn
}  // namespace tfcc
