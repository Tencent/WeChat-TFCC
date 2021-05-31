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

#include "unknowsymbolerror.h"

namespace tfcc {
namespace runtime {

UnknowSymbolError::UnknowSymbolError(const std::string& name)
    : _reason("name: " + name + " not found in symbol map") {}

const char* UnknowSymbolError::reason() const noexcept { return _reason.c_str(); }

}  // namespace runtime
}  // namespace tfcc
