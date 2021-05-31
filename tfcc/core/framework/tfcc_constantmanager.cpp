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

#include "tfcc_constantmanager.h"

#include <mutex>
#include <utility>

#include "framework/tfcc_constant.h"
#include "framework/tfcc_types.h"

namespace tfcc {

class Scope;
template <class T>
class Constant;

template <class T>
ConstantManager<T>::ConstantManager() {}

template <class T>
ConstantManager<T>::~ConstantManager() {
  _constants.clear();
}

template <class T>
Constant<T>& ConstantManager<T>::getConstant(const Scope* scope, const std::string& name) {
  std::lock_guard<SpinLock> lck(_mtx);
  auto it = _constants.find(std::make_tuple(scope, name));
  if (it != _constants.end()) {
    return *it->second;
  }

  Constant<T>* result = new Constant<T>(_allocator);
  std::unique_ptr<Constant<T>> constant(result);
  _names.insert(std::make_pair(scope, name));
  _constants.insert(std::make_pair(std::make_tuple(scope, name), std::move(constant)));
  return *result;
}

template <class T>
void ConstantManager<T>::removeConstants(const Scope* scope) {
  while (true) {
    std::lock_guard<SpinLock> lck(_mtx);
    auto it = _names.find(scope);
    if (it == _names.end()) {
      return;
    }
    _constants.erase(std::make_tuple(scope, it->second));
    _names.erase(it);
  }
  _allocator.releaseCache();
}

#define DEFINE_FUNC(type) template class ConstantManager<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
