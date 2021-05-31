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

#include "tfcc_npzdataloader.h"

#include <utility>

#include "exceptions/tfcc_exception.h"
#include "exceptions/tfcc_runtimeerror.h"
#include "utils/tfcc_commutils.h"

namespace tfcc {

NPZDataLoader::NPZDataLoader(const std::string& path) {
  std::map<std::string, std::string> nameMap = unzip_from_path(path);
  bool enable = Exception::isStackTraceEnabled();
  Exception::setStackTraceThreadLocal(false);
  for (const auto& kv : nameMap) {
    if (kv.first.size() < 4) {
      continue;
    }
    if (kv.first.substr(kv.first.size() - 4) != ".npy") {
      continue;
    }
    std::string name = kv.first.substr(0, kv.first.size() - 4);
    try {
      auto result = parse_npy(kv.second);
      _dataMap[name] = std::move(result);
    } catch (std::exception& e) {
    }
  }
  Exception::setStackTraceThreadLocal(enable);
}

NPZDataLoader::NPZDataLoader(const char* data, size_t dataLen) {
  std::string buffer(data, dataLen);
  std::map<std::string, std::string> nameMap = unzip(buffer);
  bool enable = Exception::isStackTraceEnabled();
  Exception::setStackTraceThreadLocal(false);
  for (const auto& kv : nameMap) {
    if (kv.first.size() < 4) {
      continue;
    }
    if (kv.first.substr(kv.first.size() - 4) != ".npy") {
      continue;
    }
    std::string name = kv.first.substr(0, kv.first.size() - 4);
    try {
      auto result = parse_npy(kv.second);
      _dataMap[name] = std::move(result);
    } catch (std::exception& e) {
    }
  }
  Exception::setStackTraceThreadLocal(enable);
}

const std::tuple<Shape, std::string, std::string>& NPZDataLoader::loadData(
    const std::string& name) const {
  auto it = _dataMap.find(name);
  if (it == _dataMap.end()) {
    throw RuntimeError("Name: [" + name + "] not found in data loader");
  }
  return it->second;
}

std::vector<std::tuple<std::string, std::string>> NPZDataLoader::getAllNames() const {
  std::vector<std::tuple<std::string, std::string>> result;
  for (auto kv : _dataMap) {
    result.emplace_back(kv.first, std::get<1>(kv.second));
  }
  return result;
}

bool NPZDataLoader::hasData(const std::string& name) const {
  auto it = _dataMap.find(name);
  return it != _dataMap.end();
}

}  // namespace tfcc
