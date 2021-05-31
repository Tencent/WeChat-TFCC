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

#include <map>
#include <string>
#include <vector>

#include "tfcc.h"

class FakeLoader : public tfcc::DataLoader {
  std::map<std::string, std::tuple<tfcc::Shape, std::string, std::string>> _dataMap;

 public:
  FakeLoader();

  void setData(std::string name, tfcc::Shape shape, std::vector<float> data);
  void clear();
  const std::tuple<tfcc::Shape, std::string, std::string>& loadData(
      const std::string& name) const override;
  std::vector<std::tuple<std::string, std::string>> getAllNames() const override;
  bool hasData(const std::string& name) const override;
};
