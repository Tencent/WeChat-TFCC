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
#include <memory>
#include <string>

#include "dataloaders/tfcc_dataloader.h"
#include "utils/tfcc_spinlock.h"

namespace tfcc {

/**
 * A thread-safe DataLoader that can combine other loaders.
 */
class MultiDataLoader : public DataLoader {
  mutable SpinLock _mtx;
  std::shared_ptr<std::map<std::string, const DataLoader*>> _loaderMap;

 public:
  MultiDataLoader();

  /**
   * Add a DataLoader to this instance.
   * If the prefix has existed in this instance, the function will return immediately without doing
   * anything.
   * @param prefix The sub loader prefix.
   * @param loader The sub loader.
   */
  void addLoader(std::string prefix, const DataLoader& loader);

  /**
   * Remove a DataLoader from this instance.
   * @param prefix The sub loader prefix.
   * @return If remove successfully, the function return a pointer which DataLoader has been
   * removed, otherwise return a nullptr.
   */
  const DataLoader* removeLoader(std::string prefix);

  const std::tuple<Shape, std::string, std::string>& loadData(
      const std::string& name) const override;
  std::vector<std::tuple<std::string, std::string>> getAllNames() const override;
  bool hasData(const std::string& name) const override;
};

}  // namespace tfcc
