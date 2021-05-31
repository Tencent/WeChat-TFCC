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

#include <cstddef>
#include <initializer_list>
#include <utility>
#include <vector>

namespace tfcc {

class Shape {
  std::vector<unsigned> _s;
  unsigned _area;

 public:
  Shape() : _area(0) {}

  Shape(std::initializer_list<unsigned> il) : _s(il), _area(1) {
    if (_s.size() == 0) {
      _area = 0;
    }
    for (unsigned l : _s) {
      _area *= l;
    }
  }

  Shape(std::vector<unsigned> il) : _s(std::move(il)), _area(1) {
    if (_s.size() == 0) {
      _area = 0;
    }
    for (unsigned l : _s) {
      _area *= l;
    }
  }

  Shape(const Shape& s) = default;
  Shape(Shape&& s) noexcept : _s(std::move(s._s)), _area(s._area) { s._area = 0; }

  Shape& operator=(const Shape& s) = default;
  Shape& operator=(Shape&& s) noexcept {
    _s = std::move(s._s);
    _area = s._area;
    s._area = 0;
    return *this;
  }

  unsigned operator[](size_t i) const { return _s[i]; }

  size_t size() const { return _s.size(); }

  unsigned area() const { return _area; }

  const std::vector<unsigned>& toVector() const { return _s; }

  friend bool operator==(const Shape& a, const Shape& b) {
    if (a.size() != b.size()) {
      return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
      if (a[i] != b[i]) {
        return false;
      }
    }
    return true;
  }

  friend bool operator!=(const Shape& a, const Shape& b) { return !(a == b); }
};

}  // namespace tfcc
