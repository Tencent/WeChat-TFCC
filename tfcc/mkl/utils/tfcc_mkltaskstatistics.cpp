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

#include "tfcc_mkltaskstatistics.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace tfcc {

static std::string _make_ms_string(uint64_t microseconds) {
  std::string result = std::to_string(microseconds % 1000);
  if (result.size() < 3) {
    result.insert(0, 3 - result.size(), '0');
  }
  result = std::to_string(microseconds / 1000) + "." + result;
  return result;
}

static std::string _make_ratio_string(double ratio) {
  ratio *= 10000;
  ratio = round(ratio);
  std::string result = std::to_string(static_cast<size_t>(ratio));
  if (result.size() >= 5) {
    return "99.99%";
  }
  result.insert(0, 4lu - result.size(), '0');
  result.insert(2, 1, '.');
  return result + "%";
}

static std::string _format_to_size(std::string str, size_t len) {
  if (str.size() < len) {
    str.append(len - str.size(), ' ');
  }
  return str;
}

void MKLTaskStatistics::statistics(const std::string& name, uint64_t microseconds) {
  _MKLTaskStatisticsInfo& info = _infoMap[name];
  ++info.count;
  info.maxCost = std::max(info.maxCost, microseconds);
  info.minCost = std::min(info.minCost, microseconds);
  info.totalCost += microseconds;
}

void MKLTaskStatistics::print() const { this->print(std::cout); }

void MKLTaskStatistics::print(std::ostream& stream) const {
  struct StringInfo {
    std::string name;
    std::string count;
    std::string costRatio;
    std::string totalCost;
    std::string avgCost;
    std::string minCost;
    std::string maxCost;
    uint64_t sortKey = std::numeric_limits<uint64_t>::max();
  };
  StringInfo header;
  header.name = "Name";
  header.count = "Call count";
  header.costRatio = "Cost ratio";
  header.totalCost = "Total cost(ms)";
  header.avgCost = "Average cost(ms)";
  header.minCost = "Minimum cost(ms)";
  header.maxCost = "Maximum cost(ms)";

  std::vector<StringInfo> infoList;
  size_t maxStrNameLen = header.name.size();
  size_t maxStrCountLen = header.count.size();
  size_t maxStrTotalCostLen = header.totalCost.size();
  size_t maxStrCostRatioLen = header.costRatio.size();
  size_t maxStrAvgCostLen = header.avgCost.size();
  size_t maxStrMinCostLen = header.minCost.size();
  size_t maxStrMaxCostLen = header.maxCost.size();

  auto tmpInfoMap = _infoMap;

  uint64_t taskTotalCost = 0;
  for (auto& pair : tmpInfoMap) {
    taskTotalCost += pair.second.totalCost;
  }

  infoList.push_back(header);
  for (auto& pair : tmpInfoMap) {
    StringInfo info;
    info.name = pair.first;
    info.count = std::to_string(pair.second.count);
    info.costRatio = _make_ratio_string(
        static_cast<double>(pair.second.totalCost) / static_cast<double>(taskTotalCost));
    info.totalCost = _make_ms_string(pair.second.totalCost);
    info.avgCost = _make_ms_string(pair.second.totalCost / pair.second.count);
    info.minCost = _make_ms_string(pair.second.minCost);
    info.maxCost = _make_ms_string(pair.second.maxCost);
    info.sortKey = pair.second.totalCost;

    maxStrNameLen = std::max(maxStrNameLen, info.name.size());
    maxStrCountLen = std::max(maxStrCountLen, info.count.size());
    maxStrTotalCostLen = std::max(maxStrTotalCostLen, info.totalCost.size());
    maxStrCostRatioLen = std::max(maxStrCostRatioLen, info.costRatio.size());
    maxStrAvgCostLen = std::max(maxStrAvgCostLen, info.avgCost.size());
    maxStrMinCostLen = std::max(maxStrMinCostLen, info.minCost.size());
    maxStrMaxCostLen = std::max(maxStrMaxCostLen, info.maxCost.size());

    infoList.push_back(std::move(info));
  }

  std::sort(infoList.begin() + 1, infoList.end(), [](const StringInfo& i1, const StringInfo& i2) {
    return i1.sortKey > i2.sortKey;
  });
  stream << "Task total cost: " << _make_ms_string(taskTotalCost) << std::endl;
  for (auto& info : infoList) {
    if (info.name == "") {
      continue;
    }
    stream << _format_to_size(info.name, maxStrNameLen + 4)
           << _format_to_size(info.count, maxStrCountLen + 4)
           << _format_to_size(info.costRatio, maxStrCostRatioLen + 4)
           << _format_to_size(info.totalCost, maxStrTotalCostLen + 4)
           << _format_to_size(info.avgCost, maxStrAvgCostLen + 4)
           << _format_to_size(info.minCost, maxStrMinCostLen + 4)
           << _format_to_size(info.maxCost, maxStrMaxCostLen + 4) << std::endl;
  }
}

void MKLTaskStatistics::clear() { _infoMap.clear(); }

}  // namespace tfcc
