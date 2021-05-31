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

#include "commutils.h"

#include "google/protobuf/any.pb.h"
#include "google/protobuf/message.h"

#include "tfcc_runtime/framework/types.h"

namespace tfcc {
namespace runtime {

template <class T>
bool get_value(const tfcc::runtime::common::Value& value, T& result) {
  switch (value.source_case()) {
    case tfcc::runtime::common::Value::kInt64Value:
      result = static_cast<T>(value.int64_value());
      break;
    case tfcc::runtime::common::Value::kUint64Value:
      result = static_cast<T>(value.uint64_value());
      break;
    case tfcc::runtime::common::Value::kFloatValue:
      result = static_cast<T>(value.float_value());
      break;
    case tfcc::runtime::common::Value::kDoubleValue:
      result = static_cast<T>(value.double_value());
      break;
    default:
      return false;
  }
  return true;
}

std::string get_protobuf_type_url(const ::google::protobuf::Message& message) {
  ::google::protobuf::Any value;
  value.PackFrom(message);
  return value.type_url();
}

#define DEFINE_FUNC(type) \
  template bool get_value(const tfcc::runtime::common::Value& value, type& result);

TFCC_RUNTIME_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace runtime
}  // namespace tfcc
