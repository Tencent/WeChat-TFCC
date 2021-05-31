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

#include "tfcc_runtime/proto/common.pb.h"

namespace tfcc {
namespace runtime {

template <class T>
bool get_value(const tfcc::runtime::common::Value& value, T& result);

std::string get_protobuf_type_url(const ::google::protobuf::Message& message);

}  // namespace runtime
}  // namespace tfcc
