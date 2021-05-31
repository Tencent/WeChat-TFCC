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

#include <limits>
#include <ostream>
#include <unordered_map>

namespace tfcc {

struct _MKLTaskStatisticsInfo {
  size_t count;
  uint64_t maxCost;
  uint64_t minCost;
  uint64_t totalCost;

  _MKLTaskStatisticsInfo()
      : count(0), maxCost(0), minCost(std::numeric_limits<uint64_t>::max()), totalCost(0) {}
};

class MKLTaskStatistics {
  std::unordered_map<std::string, _MKLTaskStatisticsInfo> _infoMap;

 public:
  /**
   * Statistics task info.
   * @param name Task name.
   * @param microseconds The task cost (microsecond).
   */
  void statistics(const std::string& name, uint64_t microseconds);

  /**
   * Print statistics info to std::cout.
   */
  void print() const;

  /**
   * Print statistics info to stream.
   * @param stream to print.
   */
  void print(std::ostream& stream) const;

  /**
   * Clear statistics info.
   */
  void clear();
};

}  // namespace tfcc
