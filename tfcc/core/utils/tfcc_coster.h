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

#include <algorithm>
#include <chrono>
#include <cstdint>

namespace tfcc {

class Duration {
  std::chrono::steady_clock::duration du;

 public:
  explicit Duration(std::chrono::steady_clock::duration d) : du(d) {}

  unsigned long microseconds() const {
    unsigned long cost = std::chrono::duration_cast<std::chrono::microseconds>(du).count();
    return std::max(0lu, cost);
  }

  unsigned long milliseconds() const {
    unsigned long cost = std::chrono::duration_cast<std::chrono::milliseconds>(du).count();
    return std::max(0lu, cost);
  }

  unsigned long seconds() const {
    unsigned long cost = std::chrono::duration_cast<std::chrono::seconds>(du).count();
    return std::max(0lu, cost);
  }

  unsigned long minutes() const {
    unsigned long cost = std::chrono::duration_cast<std::chrono::minutes>(du).count();
    return std::max(0lu, cost);
  }

  unsigned long hours() const {
    unsigned long cost = std::chrono::duration_cast<std::chrono::hours>(du).count();
    return std::max(0lu, cost);
  }
};

class Coster {
  std::chrono::system_clock::time_point start;

 public:
  Coster() : start(std::chrono::system_clock::now()) {}

  void reset() { start = std::chrono::system_clock::now(); }

  Duration lap() const {
    auto now = std::chrono::system_clock::now();
    return Duration(now - start);
  }
};

}  // namespace tfcc
