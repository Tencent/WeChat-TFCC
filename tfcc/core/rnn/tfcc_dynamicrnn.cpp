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

#include "tfcc_dynamicrnn.h"

#include "rnn/tfcc_lstmcell.h"

namespace tfcc {
namespace rnn {

template std::tuple<Variable<float>, typename LSTMCell<float>::State> dynamic_rnn(
    LSTMCell<float>&, const Tensor<float>&, typename LSTMCell<float>::State&&, size_t, size_t,
    const std::string&);

template std::tuple<
    Variable<float>, Variable<float>, typename LSTMCell<float>::State,
    typename LSTMCell<float>::State>
bidirectional_dynamic_rnn(
    LSTMCell<float>&, LSTMCell<float>&, const Tensor<float>&, typename LSTMCell<float>::State&&,
    typename LSTMCell<float>::State&&, size_t, size_t, const std::string&);

}  // namespace rnn
}  // namespace tfcc
