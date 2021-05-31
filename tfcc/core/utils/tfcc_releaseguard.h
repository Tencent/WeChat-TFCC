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

#include <utility>

template <class Func>
class ReleaseGuard {
  Func _func;
  bool _valid;

 public:
  explicit ReleaseGuard(Func func) : _func(func), _valid(true) {}
  ~ReleaseGuard() {
    if (_valid) {
      _func();
    }
  }

  ReleaseGuard(const ReleaseGuard&) = delete;
  ReleaseGuard(ReleaseGuard&& g) : _func(std::move(g._func)), _valid(true) { g._valid = false; }

  ReleaseGuard& operator=(const ReleaseGuard&) = delete;
  ReleaseGuard& operator=(ReleaseGuard&&) = delete;
};

/**
 * Create a ReleaseGuard
 * @param func A callable object without argument. The function will call before the guard is
 * destroyed.
 * @return The guard object.
 */
template <class Func>
inline ReleaseGuard<Func> make_release_guard(Func func) {
  return ReleaseGuard<Func>(func);
}
