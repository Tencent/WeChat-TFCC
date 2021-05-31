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

#include "tfcc_scope.h"

#include "utils/tfcc_commutils.h"

#include <mutex>
#include <utility>

namespace tfcc {

ScopeGuard::ScopeGuard() : _valid(true) {}

ScopeGuard::ScopeGuard(ScopeGuard&& guard) noexcept : _valid(guard._valid) { guard._valid = false; }

ScopeGuard::~ScopeGuard() {
  if (_valid) {
    Scope::popScope();
  }
}

Scope Scope::_root(nullptr, "");
thread_local Scope* Scope::_currentScope = &Scope::_root;

Scope::Scope(Scope* parent, const std::string& name) : _parent(parent), _name(name) {}

Scope::~Scope() { release_constants(this); }

Scope& Scope::getChild(const std::string& name) {
  std::lock_guard<SpinLock> lck(_mtx);
  auto it = _children.find(name);
  if (it != _children.end()) {
    return *it->second;
  }
  Scope* p = new Scope(this, name);
  _children[name] = std::unique_ptr<Scope>(p);
  return *p;
}

std::string Scope::getFullName() const {
  if (_parent == nullptr) {
    return "";
  }
  return _parent->getFullName() + _name + "/";
}

void Scope::removeChild(const std::string& name) {
  std::unique_ptr<Scope> p;
  {
    std::lock_guard<SpinLock> lck(_mtx);
    auto it = _children.find(name);
    if (it == _children.end()) {
      return;
    }
    p = std::move(it->second);
    _children.erase(it);
  }
}

void Scope::pushScope(const std::string& name) {
  Scope& s = _currentScope->getChild(name);
  _currentScope = &s;
}

void Scope::popScope() { _currentScope = _currentScope->getParent(); }

ScopeGuard Scope::scope(const std::string& name) {
  pushScope(name);
  return ScopeGuard();
}

}  // namespace tfcc
