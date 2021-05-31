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

#include "framework/tfcc_shape.h"

namespace tfcc {

class Device;

template <class T>
class Tensor {
 protected:
  Shape _shape;

 public:
  virtual ~Tensor() {}

  /**
   * @warning Don't Use this function directly! Please use tfcc::data::get instead, If you want get
   * the value of the tensor.
   */
  virtual const T* data() const = 0;
  virtual void sync() const = 0;
  virtual bool available() const = 0;

  unsigned size() const { return _shape.area(); }

  const Shape& shape() const { return _shape; }

  unsigned shape(size_t n) const { return _shape[n]; }

 protected:
  Tensor() {}
  explicit Tensor(const Shape& s) : _shape(s) {}
  explicit Tensor(Shape&& s) : _shape(std::move(s)) {}
  Tensor(const Tensor&) = delete;
  Tensor(Tensor&& v) noexcept = default;

  Tensor& operator=(const Tensor&) = delete;
  Tensor& operator=(Tensor&&) noexcept = default;
};

}  // namespace tfcc
