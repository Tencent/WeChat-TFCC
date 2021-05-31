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

#include "tfcc_exception.h"

#ifndef WITHOUT_BOOST
#include <sstream>
#include <boost/stacktrace.hpp>
#endif

namespace tfcc {

#ifndef WITHOUT_BOOST
thread_local bool Exception::_stackTraceEnabled = true;
#else
thread_local bool Exception::_stackTraceEnabled = false;
#endif

Exception::Exception() {
#ifndef WITHOUT_BOOST
  if (_stackTraceEnabled) {
    std::stringstream ss;
    ss << boost::stacktrace::stacktrace();
    _stackInfo = ss.str();
  }
#else
  if (_stackTraceEnabled) {
    _stackInfo = "can't open stacktrace. rebuild tfcc with USE_BOOST=yes please.\n";
  }
#endif
}

const char* Exception::what() const noexcept {
  if (_errorInfo.size() == 0) {
    _errorInfo = reason();
    if (_stackInfo.size() > 0) {
      _errorInfo += "\n";
      _errorInfo += _stackInfo;
    }

    if (_parentInfo.size() > 0) {
      _errorInfo += "\nCause by:\n";
      _errorInfo += _parentInfo;
    }
  }

  return _errorInfo.c_str();
}

const char* Exception::reason() const noexcept { return "Unknown exception"; }

void Exception::setParent(const std::exception& e) noexcept {
  _parentInfo = e.what();
  _errorInfo = std::string();
}

void Exception::setStackTraceThreadLocal(bool enable) { _stackTraceEnabled = enable; }

bool Exception::isStackTraceEnabled() { return _stackTraceEnabled; }

}  // namespace tfcc
