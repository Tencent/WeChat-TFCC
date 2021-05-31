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

#include "tfcc_multidataloader.h"

#include <mutex>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_runtimeerror.h"

namespace tfcc {

MultiDataLoader::MultiDataLoader() : _loaderMap(new std::map<std::string, const DataLoader*>) {}

void MultiDataLoader::addLoader(std::string prefix, const DataLoader& loader) {
  if (prefix == "") {
    throw InvalidArgumentError("invalid prefix");
  }
  if (prefix[prefix.size() - 1] != '/') {
    prefix += '/';
  }

  while (true) {
    std::shared_ptr<std::map<std::string, const DataLoader*>> loaderMap;
    {
      std::lock_guard<SpinLock> lck(_mtx);
      loaderMap = _loaderMap;
    }
    if (loaderMap->find(prefix) != loaderMap->end()) {
      break;
    }
    std::shared_ptr<std::map<std::string, const DataLoader*>> newLoaderMap(
        new std::map<std::string, const DataLoader*>);
    *newLoaderMap = *loaderMap;
    newLoaderMap->insert(std::make_pair(prefix, &loader));
    {
      std::lock_guard<SpinLock> lck(_mtx);
      if (loaderMap != _loaderMap) {
        continue;
      }
      _loaderMap = newLoaderMap;
      break;
    }
  }
}

const DataLoader* MultiDataLoader::removeLoader(std::string prefix) {
  if (prefix == "") {
    throw InvalidArgumentError("invalid prefix");
  }
  if (prefix[prefix.size() - 1] != '/') {
    prefix += '/';
  }
  const DataLoader* loader = nullptr;
  while (true) {
    loader = nullptr;
    std::shared_ptr<std::map<std::string, const DataLoader*>> loaderMap;
    {
      std::lock_guard<SpinLock> lck(_mtx);
      loaderMap = _loaderMap;
    }
    if (loaderMap->find(prefix) == loaderMap->end()) {
      break;
    }
    std::shared_ptr<std::map<std::string, const DataLoader*>> newLoaderMap(
        new std::map<std::string, const DataLoader*>);
    *newLoaderMap = *loaderMap;
    auto it = newLoaderMap->find(prefix);
    loader = it->second;
    newLoaderMap->erase(it);
    {
      std::lock_guard<SpinLock> lck(_mtx);
      if (loaderMap != _loaderMap) {
        continue;
      }
      _loaderMap = newLoaderMap;
      break;
    }
  }
  return loader;
}

const std::tuple<Shape, std::string, std::string>& MultiDataLoader::loadData(
    const std::string& name) const {
  std::shared_ptr<std::map<std::string, const DataLoader*>> loaderMap;
  {
    std::lock_guard<SpinLock> lck(_mtx);
    loaderMap = _loaderMap;
  }
  size_t pos = name.find('/');
  if (pos == std::string::npos) {
    throw RuntimeError("Name: [" + name + "] not found in data loader");
  }
  std::string prefix = name.substr(0, pos + 1);
  std::string realName = name.substr(pos + 1);
  auto it = loaderMap->find(prefix);
  if (it == loaderMap->end()) {
    throw RuntimeError("Name: [" + name + "] not found in data loader");
  }
  return it->second->loadData(realName);
}

std::vector<std::tuple<std::string, std::string>> MultiDataLoader::getAllNames() const {
  std::shared_ptr<std::map<std::string, const DataLoader*>> loaderMap;
  {
    std::lock_guard<SpinLock> lck(_mtx);
    loaderMap = _loaderMap;
  }
  std::vector<std::tuple<std::string, std::string>> result;
  for (auto& kv : *loaderMap) {
    auto names = kv.second->getAllNames();
    for (auto& name : names) {
      result.emplace_back(kv.first + std::get<0>(name), std::get<1>(name));
    }
  }
  return result;
}

bool MultiDataLoader::hasData(const std::string& name) const {
  std::shared_ptr<std::map<std::string, const DataLoader*>> loaderMap;
  {
    std::lock_guard<SpinLock> lck(_mtx);
    loaderMap = _loaderMap;
  }
  size_t pos = name.find('/');
  if (pos == std::string::npos) {
    return false;
  }
  std::string prefix = name.substr(0, pos + 1);
  std::string realName = name.substr(pos + 1);
  auto it = loaderMap->find(prefix);
  if (it == loaderMap->end()) {
    return false;
  }
  return it->second->hasData(realName);
}

}  // namespace tfcc
