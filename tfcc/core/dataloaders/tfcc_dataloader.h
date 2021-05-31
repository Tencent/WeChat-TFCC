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
#include <tuple>

#include "framework/tfcc_shape.h"

namespace tfcc {

class DataLoader {
  static DataLoader* _defaultDataLoader;

 public:
  DataLoader() {}
  virtual ~DataLoader() {}

  /**
   * tuple(shape, type, data)
   * type is one of 'i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8', 'f4', 'f8'
   */
  virtual const std::tuple<Shape, std::string, std::string>& loadData(
      const std::string& name) const = 0;

  /**
   * Get all names the DataLoader have.
   * @return A vector of std::tuple<name, type>. type is one of 'i1', 'i2', 'i4', 'i8', 'u1', 'u2',
   * 'u4', 'u8', 'f4', 'f8'
   */
  virtual std::vector<std::tuple<std::string, std::string>> getAllNames() const = 0;

  /**
   * Check for data in the DataLoader.
   * @param name The name of data.
   * @return Has or not.
   */
  virtual bool hasData(const std::string& name) const = 0;

 public:
  static DataLoader* getGlobalDefault();
  static void setGlobalDefault(DataLoader* dataLoader);
};

}  // namespace tfcc
