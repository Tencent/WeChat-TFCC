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

#include <string>
#include <vector>

#include "tfcc.h"

class Environment {
  tfcc::Device* _device;
  tfcc::Session* _session;
  tfcc::DeviceType _currentDeviceType;

  static tfcc::DeviceType _defaultDeviceType;
  static tfcc::DataLoader* _defaultTestDataLoader;
  static std::string _defaultTestDataPath;

 public:
  Environment() : _device(nullptr), _session(nullptr) {}
  ~Environment() { release(); }

  void init();
  void release();

  tfcc::DeviceType getCurrentDeviceType() { return _currentDeviceType; }

  static void setDefaultDeviceType(tfcc::DeviceType deviceType) { _defaultDeviceType = deviceType; }

  static void setDefaultTestDataPath(const std::string& path);
  static std::string getDefaultTestDataPath();
};

class ExceptionGuard {
  bool _enable;

 public:
  ExceptionGuard() : _enable(tfcc::Exception::isStackTraceEnabled()) {}

  ~ExceptionGuard() { tfcc::Exception::setStackTraceThreadLocal(_enable); }
};

template <class T>
std::vector<T> makeSequence(T start, T skip, size_t count) {
  std::vector<T> result;
  for (size_t i = 0; i < count; ++i) {
    result.push_back(start);
    start += skip;
  }
  return result;
}

template <class T>
std::string vectorToString(const std::vector<T>& vec) {
  std::string result;
  result += "[ ";
  for (T v : vec) {
    result += std::to_string(v) + " ";
  }
  result += "]";
  return result;
}

std::vector<size_t> revert_transpose_perm(std::vector<size_t> perm);
