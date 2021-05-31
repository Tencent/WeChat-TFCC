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

#include "tfcc_configure.h"

#include <mutex>

#include "dataloaders/tfcc_dataloader.h"
#include "exceptions/tfcc_runtimeerror.h"
#include "framework/tfcc_scope.h"
#include "framework/tfcc_types.h"
#include "utils/tfcc_commutils.h"

namespace tfcc {

template <class T>
Configure<T>& Configure<T>::getDefault() {
  static Configure<T> configure;
  return configure;
}

template <class T>
Configure<T>::Configure() {}

template <class T>
T Configure<T>::getConfigure(const std::string& name, bool& existed) {
  Configure<T>& configure = getDefault();
  std::lock_guard<SpinLock> lck(configure._mtx);
  auto key = std::make_tuple(&Scope::getCurrentScope(), name);
  auto it = configure._dataMap.find(key);
  if (it != configure._dataMap.end()) {
    existed = true;
    return it->second;
  }

  std::string fullName = Scope::getCurrentScope().getFullName() + name;
  std::string type, data;
  Shape shape;
  std::vector<T> realData;
  DataLoader* dataLoader = DataLoader::getGlobalDefault();
  if (!dataLoader->hasData(fullName)) {
    existed = false;
    return T();
  }
  std::tie(shape, type, data) = dataLoader->loadData(fullName);
  realData = transfer_string_data<T>(type, data);
  if (realData.size() != 1) {
    throw RuntimeError("DataLoader return a invalid configure. Constant name: " + fullName);
  }
  configure._dataMap[key] = realData[0];
  existed = true;
  return realData[0];
}

template <class T>
T Configure<T>::getConfigure(const std::string& name, T defaultValue) {
  bool existed = false;
  T value = getConfigure(name, existed);
  if (!existed) {
    value = defaultValue;
  }
  return value;
}

template <class T>
T Configure<T>::getConfigure(const std::string& name) {
  bool existed = false;
  T value = getConfigure(name, existed);
  if (!existed) {
    throw RuntimeError(
        "Name: [" + Scope::getCurrentScope().getFullName() + name + "] not found in data loader");
  }
  return value;
}

template <class T>
void Configure<T>::removeConfigures(const Scope* scope) {
  Configure<T>& configure = getDefault();
  std::lock_guard<SpinLock> lck(configure._mtx);
  auto it = configure._dataMap.begin();
  while (it != configure._dataMap.end()) {
    if (std::get<0>(it->first) != scope) {
      ++it;
      continue;
    }
    it = configure._dataMap.erase(it);
  }
}

#define DEFINE_FUNC(type) template class Configure<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
