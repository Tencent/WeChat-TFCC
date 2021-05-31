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
#include <tuple>

#include "utils/tfcc_spinlock.h"

namespace tfcc {

template <class T>
class Constant;

template <class T>
class Configure;

class Device;

class Scope;

class ScopeGuard {
  bool _valid;

 public:
  ScopeGuard();
  ScopeGuard(ScopeGuard&& guard) noexcept;
  ScopeGuard(const ScopeGuard& guard) = delete;
  ~ScopeGuard();

  ScopeGuard& operator=(ScopeGuard&& guard) = delete;
  ScopeGuard& operator=(const ScopeGuard& guard) = delete;
};

class Scope {
  Scope* _parent;
  std::string _name;
  SpinLock _mtx;
  std::map<std::string, std::unique_ptr<Scope>> _children;

 private:
  static Scope _root;
  static thread_local Scope* _currentScope;

 private:
  Scope(Scope* parent, const std::string& name);

  Scope& getChild(const std::string& name);
  Scope* getParent() { return _parent; }

 public:
  Scope(const Scope&) = delete;
  Scope(Scope&&) = delete;
  ~Scope();

  Scope& operator=(const Scope&) = delete;
  Scope& operator=(Scope&&) = delete;

  const Scope* getParent() const { return _parent; }

  const std::string& getName() const { return _name; }

  std::string getFullName() const;

  /**
   * Remove child scope and it's Constant.
   * Be carefully to use this function.
   * @param name The child scope name.
   */
  void removeChild(const std::string& name);

 public:
  static void pushScope(const std::string& name);
  static void popScope();
  static ScopeGuard scope(const std::string& name);
  static Scope& getCurrentScope() { return *_currentScope; }
  static void setCurrentScope(Scope& s) { _currentScope = &s; }
  static Scope& getRootScope() { return _root; }
};

}  // namespace tfcc
