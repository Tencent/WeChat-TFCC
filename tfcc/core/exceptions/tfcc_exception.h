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

#include <exception>
#include <string>
#include <vector>

namespace tfcc {

class Exception : public std::exception {
  mutable std::string _stackInfo;
  mutable std::string _parentInfo;
  mutable std::string _errorInfo;

 private:
  static thread_local bool _stackTraceEnabled;

 public:
  Exception();

  const char* what() const noexcept override;

  virtual const char* reason() const noexcept;
  void setParent(const std::exception& e) noexcept;

 public:
  static void setStackTraceThreadLocal(bool enable);
  static bool isStackTraceEnabled();
};

}  // namespace tfcc
