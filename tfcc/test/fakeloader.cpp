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

#include "fakeloader.h"

#include <stdexcept>

FakeLoader::FakeLoader() {}

void FakeLoader::setData(std::string name, tfcc::Shape shape, std::vector<float> data) {
  std::string strData(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
  _dataMap[name] = std::make_tuple(shape, "f4", strData);
}

void FakeLoader::clear() { _dataMap.clear(); }

const std::tuple<tfcc::Shape, std::string, std::string>& FakeLoader::loadData(
    const std::string& name) const {
  auto it = _dataMap.find(name);
  if (it == _dataMap.end()) {
    throw std::runtime_error("Name: [" + name + "] not found in data loader");
  }
  return it->second;
}

std::vector<std::tuple<std::string, std::string>> FakeLoader::getAllNames() const {
  std::vector<std::tuple<std::string, std::string>> result;
  for (auto kv : _dataMap) {
    result.emplace_back(kv.first, std::get<1>(kv.second));
  }
  return result;
}

bool FakeLoader::hasData(const std::string& name) const {
  auto it = _dataMap.find(name);
  return it != _dataMap.end();
}
